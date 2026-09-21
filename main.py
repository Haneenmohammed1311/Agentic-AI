import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter # for sorting the list of tuples based on the second element (similarity score)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_chroma import Chroma

load_dotenv()
print("Loading documents from text file...")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # for embedding generation

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash") # for LLM responses

vectorstore = Chroma(persist_directory="./chroma_db",
                      embedding_function=embeddings) # for loading the vectorstore from the database from ingestion.py

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
# Return VectorStoreRetriever initialized from this VectorStore, k: is the number of documents to return from the retriever


# RAG, and A is Augmentation. The prompt template is used to format the input to the LLM.
prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions based on the context provided.
{context}
Question: {question}
please answer the question as truthfully as possible using the provided context,
 and if the answer is not contained within the text below, say "I don't know."
"""
)


##############
## 1) Manually implemented retrieval of RAG chain without using LLM, just retrieves documents and formats them.
##############
def format_docs(docs):
    """"
    Format the retrieved documents into a string for the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str) -> str:
    """
    Simple retrieval chain without using LLM, just retrieves documents and formats them.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No built-in error handling
    - No built-in caching or memory
"""
    # step 1: Retrieve relevant documents based on the query
    docs = retriever.invoke(query)
    #step 2: Format the retrieved documents into a string for the prompt
    context = format_docs(docs)
    #step 3: Format the prompt with the context and query
    message = prompt_template.format_messages(context=context, question=query)
    #step 4: Invoke the LLM with the formatted prompt and return the response
    response = llm.invoke(message)
    #step 5: Return the response content
    return response.content

#################
## 2) Implemented retrieval of RAG chain using LLM with LCEL (LangChain Expression Language)
#################
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    # Step 1: Create a retriever that retrieves relevant documents based on the query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 2: Create a chain that retrieves documents and formats them into a string for the prompt
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain





if __name__ == "__main__":
    print("Ready to answer questions. Type 'exit' to quit.")

    query = "What is the pinecoe in machine learning?"

# ##########
# # Option 0: Answer the question using the LLM without using llm
# ###########
print("=="*27)
print ("Answering question without using RAG Only LLm...")
llm_response = llm.invoke([HumanMessage(content=query)])
print(f"LLM Response: {llm_response.content}")
print("=="*27)

# ###########
# # Option 1: Answer the question using the LLM without using llm
# ###########
print("\n" + "=" * 70)
print("IMPLEMENTATION 1: Without LCEL")
print("=" * 70)
result_without_lcel = retrieval_chain_without_lcel(query)
print("\nAnswer:")
print(result_without_lcel)


##########
# Option 2: Answer the question using the LLM with LCEL
###########

print("IMPLEMENTATION 2: With LCEL - Better Approach")
print("=" * 70)
print("=" * 70)

chain_with_lcel = create_retrieval_chain_with_lcel()
result_with_lcel = chain_with_lcel.invoke({"question": query})
print("\nAnswer:")
print(result_with_lcel)