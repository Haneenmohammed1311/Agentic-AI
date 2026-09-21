## RAG Gist

This branch builds a complete Retrieval Augmented Generation pipeline, split into two files: `ingestion.py`, which prepares the knowledge base once, and `main.py`, which answers questions using that knowledge base, shown three different ways.

### Stack used in this branch, and why it differs from the instructor's

The instructor's original version uses `OpenAIEmbeddings`, `ChatOpenAI`, and `PineconeVectorStore`. This version swaps each of those for a free, mostly local equivalent, since `app.pinecone.io` was unreachable from my network, and to avoid burning through paid API usage while iterating repeatedly.

| Original | Used here | Why |
|---|---|---|
| `OpenAIEmbeddings` | `HuggingFaceEmbeddings` | Runs locally, no API key, no cost, downloads once and caches |
| `ChatOpenAI` | `ChatGoogleGenerativeAI` (Gemini) | Free tier, used carefully due to a 20 request per day limit |
| `PineconeVectorStore` | `Chroma` | Fully local vector store, no account or network needed |

---

### Part 1, `ingestion.py`, building the knowledge base

This file runs once, whenever the source document changes. It is not part of answering questions, it prepares the data questions will later be answered from.

**1. `TextLoader`**
```python
loader = TextLoader("mediumblog1.txt", encoding="utf-8")
documents = loader.load()
```
`TextLoader` reads a plain text file from disk and wraps its content in LangChain's internal `Document` object. `encoding="utf-8"` tells it how to interpret the file's bytes as text, important for files with special characters. `.load()` returns a list of `Document` objects, one per file loaded.

**2. `CharacterTextSplitter`**
```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
```
- `chunk_size=1000` → the maximum number of characters allowed in a single chunk.
- `chunk_overlap=200` → how many characters from the end of one chunk are repeated at the start of the next chunk, this prevents a sentence or fact from being cut cleanly in half between two chunks with no shared context.
- `.split_documents()` takes the whole loaded document and breaks it into many smaller `Document` chunks, this is what actually gets embedded and stored, not the original whole file.

**3. `HuggingFaceEmbeddings`**
```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
This loads a small, local embedding model. `model_name` points to a specific pretrained model on HuggingFace's hub, downloaded once and cached locally afterward. This object does not embed anything yet by itself, it is a tool passed into the vector store, which calls it internally on every chunk.

**4. `Chroma.from_documents`**
```python
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
```
- `texts` → the list of chunked documents from step 2.
- `embeddings` → the embedding model from step 3, used internally on every chunk.
- `persist_directory="./chroma_db"` → the local folder where the vectors and their original text get saved to disk, so they persist after the script ends.

This single line does embedding and storage together, every chunk gets converted into a vector, and both the vector and its original text get written to `./chroma_db`.

---

### Part 2, `main.py`, answering questions three ways

All three implementations use the same `embeddings`, `llm`, `vectorstore`, `retriever`, and `prompt_template`, defined once near the top of the file. Only how they are wired together differs.

**Shared setup**

```python
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```
This reopens the exact same local store `ingestion.py` created, no re-embedding happens here, it just reads what was already saved to disk.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```
`.as_retriever()` wraps the vector store in a standard LangChain `Retriever` object. `search_kwargs={"k": 3}` means every search returns the top 3 closest matching chunks, `k` is the count of results, not a distance or score.

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```
Takes a list of `Document` objects, retrieved chunks, and joins just their text content, `doc.page_content` specifically, ignoring metadata, into one single string, separated by blank lines, ready to be inserted into a prompt.

---

#### Implementation 0, raw LLM, no RAG at all

```python
llm_response = llm.invoke([HumanMessage(content=query)])
```
The question goes straight to the model with **no retrieval, no context, no document knowledge involved at all**. This exists purely as a baseline for comparison, to show what the model would answer relying only on what it already learned during training.

In your actual run, the model still answered correctly about Pinecone here, because Pinecone is a well known public tool the model already learned about during training. This baseline becomes far more revealing once you ask something specific to your own private document that the model could not possibly know from training alone, that is the scenario where RAG's value becomes obvious rather than subtle.

---

#### Implementation 1, manual RAG, without LCEL

```python
def retrieval_chain_without_lcel(query: str) -> str:
    docs = retriever.invoke(query)
    context = format_docs(docs)
    message = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(message)
    return response.content
```

Five explicit, ordered steps, each one a separate line, each variable feeding into the next by hand.

1. `retriever.invoke(query)` → embeds the query internally, finds the 3 closest chunks, returns them as a list of `Document` objects.
2. `format_docs(docs)` → flattens that list into one plain text string.
3. `prompt_template.format_messages(context=context, question=query)` → fills the prompt template's placeholders with the real context and question, producing the final message ready to send.
4. `llm.invoke(message)` → sends that filled prompt to Gemini, gets back a response object.
5. `return response.content` → pulls just the answer text out of that response object.

**Why this matters in your actual output.** `response.content` here returned the raw structured list, `[{'type': 'text', 'text': '...'}]`, exactly the same issue you first hit back in the hello world lesson. This function does nothing to clean that up, it returns whatever `response.content` happens to be, structured or plain, depending on the model.

---

#### Implementation 2, RAG with LCEL

```python
retrieval_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
)
```

This is the piece worth understanding slowly, since it looks compressed but is doing exactly the same five steps as Implementation 1, just declared as a pipeline instead of written as sequential lines.

**Reading it from the inside out, not top to bottom, is the easiest way to understand it.**

`itemgetter("question")` → this chain is eventually called with a dictionary, `{"question": "..."}`. `itemgetter("question")` is a small function that, given that dictionary, pulls out just the value behind the `"question"` key, the plain query string.

`| retriever` → that extracted question string is piped into the retriever, exactly like `retriever.invoke(query)` in the manual version, producing the list of top 3 matching `Document` chunks.

`| format_docs` → that list of chunks is piped into `format_docs`, flattening it into one context string, exactly like step 2 in the manual version.

So the whole line `itemgetter("question") | retriever | format_docs` is a **self contained mini pipeline**, question in, formatted context string out. This mini pipeline becomes the value assigned to a new key called `context`.

**`RunnablePassthrough.assign(context=...)`** → this is the part that connects that mini pipeline back to the full input. `RunnablePassthrough` means "pass the original input through unchanged." `.assign(context=...)` means "and also add a new key called `context`, computed by the pipeline above." So if the chain was called with `{"question": "what is a vector store"}`, after this step the data flowing through becomes `{"question": "what is a vector store", "context": "...the retrieved text..."}`, both keys now present together.

**`| prompt_template`** → that combined dictionary, now containing both `question` and `context`, matches exactly what `prompt_template` expects to fill in its `{question}` and `{context}` placeholders, so it fills them in automatically.

**`| llm`** → the filled prompt is sent to Gemini, same as before.

**`| StrOutputParser()`** → this is the step that explains the clean output you saw. `StrOutputParser` is a small, standard LCEL component whose entire job is to take whatever the model returned and extract just the plain text answer from it, regardless of whether the model's raw response was already a plain string or a structured list like Gemini's. This is exactly why **Implementation 2 printed a clean sentence while Implementation 1 printed the raw list**, Implementation 1 has no equivalent cleanup step, it just returns `response.content` directly, whatever shape that happens to be.

**The arrow shaped summary of the whole chain:**
{"question": "..."}
-> RunnablePassthrough.assign adds "context" by running:
question -> retriever -> format_docs
-> {"question": "...", "context": "..."}
-> prompt_template fills in both placeholders
-> llm generates a response
-> StrOutputParser extracts clean text
-> final answer string


Every `|` in this whole chain means exactly the same thing, take whatever came out of the step on the left, and feed it directly into the step on the right as its input.

---

### Comparing what you actually saw across all three

| | Implementation 0 | Implementation 1 | Implementation 2 |
|---|---|---|---|
| Uses retrieved documents | No | Yes | Yes |
| Answer source | Model's training data only | Retrieved context plus model | Retrieved context plus model |
| Output cleanliness | Raw structured list | Raw structured list | Clean plain text, thanks to `StrOutputParser` |
| Code style | One line | Five explicit steps | One declarative pipeline |

### Key lesson from this branch

`StrOutputParser` is not just a cosmetic convenience, it is solving a real, recurring problem you have hit multiple times in this course, some models, Gemini specifically, return `response.content` as a structured list rather than plain text, and every manual implementation you write has to remember to handle that itself. LCEL's parser components exist specifically so you do not have to remember that fix in every single function you write, you add it once, at the end of the chain, and it applies automatically no matter what shape the model's raw output happens to take.

### Setup

```bash
git checkout project/rag-gist
uv sync
```

Environment variables needed in `.env`.
GOOGLE_API_KEY=your_key_here
No `PINECONE_API_KEY` or `INDEX_NAME` needed, since Chroma needs neither.

Run ingestion first, once, whenever the source document changes.
```bash
uv run python ingestion.py
```

Then run the retrieval script as many times as needed.
```bash
uv run python main.py
```