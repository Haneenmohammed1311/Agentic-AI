# Agentic AI Learning Journey

My hands-on journey learning to build AI agents with LangChain and LangGraph: tools, agent loops, RAG, state graphs, and production concerns, in Python.

I am working through Eden Marco's LangChain and LangGraph course on Udemy, and I rebuild every lesson myself. This repository is the journal: one branch per lesson, plus my theory notes.

## The course

| Item | Details |
|---|---|
| Course | [LangChain and LangGraph course by Eden Marco on Udemy](https://www.udemy.com/course/langchain/?couponCode=MT261019G3) |
| Status | In progress |

> **Featured project:** [Agentic RAG with LangGraph](https://github.com/Haneenmohammed1311/Agentic-RAG-LangGraph). A RAG system built in three stages (naive, corrective, adaptive) as a LangGraph state machine, with a router, document grading, web search fallback, and hallucination and answer checks.

## Lessons, one branch each

Each lesson lives on its own branch, so `main` stays a clean starting point and every project keeps its own history and code.

| Branch | What it covers |
|---|---|
| [project/hello-world](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/hello-world) | First LLM chain, prompt templates, basic LangChain wiring, LangSmith tracing |
| [project/search-agent](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/search-agent) | Search agent evolving from a custom tool to built-in `TavilySearch`, using `create_agent`, with structured Pydantic outputs |
| [project/agents-under-the-hood](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/agents-under-the-hood) | The same agent loop rebuilt three ways: LangChain tool calling, raw Ollama function calling, and a raw ReAct prompt with regex parsing, to see what `create_agent` hides |
| [project/rag-gist](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/rag-gist) | A full RAG pipeline: ingestion into a vector store, then the same retrieval logic implemented manually and with LCEL |

## External repos

Some lessons are separate repositories instead of branches, matching how the course itself organizes them.

| Repo | What it covers |
|---|---|
| [Agentic RAG](https://github.com/Haneenmohammed1311/agentic-rag-langgraph) | Naive, corrective, and adaptive RAG as one LangGraph graph, with a local Chroma store, Tavily web search, and an LLM grading pipeline |
| [Documentation Helper](https://github.com/Haneenmohammed1311/Documentation-Helper) | A Streamlit RAG chat app over LangChain's own documentation, with Tavily crawling, Chroma, and an agent that cites its sources |
| [LangGraph Course](https://github.com/Haneenmohammed1311/LangGrapgh-Course) | LangGraph from first principles. `main` is a ReAct tool-calling agent, [project/reflection-agent](https://github.com/Haneenmohammed1311/LangGrapgh-Course/tree/project/reflection-agent) is a self-critiquing generate-and-reflect loop, and [project/reflexion-agent](https://github.com/Haneenmohammed1311/LangGrapgh-Course/tree/project/reflexion-agent) is a research agent that critiques itself and revises with cited web search |


## Notes

Theory chapters, written for review, with diagrams:

- [Prompt theory and context engineering](notes/prompt-theory-and-context-engineering.md)
- [LLM production landscape and course wrap-up](notes/llm-production-landscape-and-course-wrapup.md)

## Tech stack

- **Python**, 3.13 in the recent projects
- **LangChain** and **LangGraph** for agent orchestration
- **uv** for dependency and environment management
- **LLMs**: Google Gemini, and local Ollama models (`llama3.2:3b`, `qwen3:0.6b`, `qwen3:1.7b`), swapped depending on the lesson, rate limits, and whether the goal is speed or fully offline testing
- **Chroma** as a local vector store, used in place of Pinecone
- **HuggingFace `sentence-transformers`** for local embeddings, used in place of OpenAI embeddings
- **Tavily** for web search and crawling
- **Pydantic** for structured model output
- **Streamlit** for the Documentation Helper interface
- **LangSmith** for tracing, with one project per lesson to keep traces separate
- **pytest** for chain tests

## How to run a lesson

```bash
git checkout project/<lesson-name>
uv sync
```

Each branch's README lists its own run commands and environment variables, because the entry points differ between lessons.

## Why the stack differs from the course

The course is built around OpenAI for embeddings and chat, and Pinecone as the vector store. Two practical constraints shaped different choices here.

`app.pinecone.io` was unreachable from my network. I confirmed it across multiple days, networks, and devices, while Pinecone's marketing site loaded fine, which points to a routing issue outside my control. Chroma replaced it: it is fully local, needs no account, and has the same LangChain interface, so the RAG logic needed no real changes.

OpenAI's embedding and chat APIs need paid billing. Gemini's free tier and local Ollama models fill the same roles at no cost. The trade-offs are Gemini's daily request quota and the fact that small local models follow multi-step tool instructions less reliably than a stronger model.

## Challenges solved along the way

- Gemini often returns `response.content` as a list of blocks instead of a string. I saw why LCEL's `StrOutputParser` handles this automatically while manual code needs an `isinstance` check.
- Gemini's free tier has a daily request cap. I switched to local Ollama models during heavy debugging.
- `init_chat_model` guesses `google_vertexai` for Gemini model names, which needs Google Cloud credentials. Passing `model_provider="google_genai"` explicitly fixed it.
- A raw ReAct agent returned a wrong answer without crashing. The parser checked for `Final Answer` before checking for a real `Action`, so the model could skip a tool call it should have made.
- Gemini rejects a conversation that ends with an assistant message, and rejects system messages that are not first. The reflection agent needed a role swap, and the reflexion agent needed a single system message.
- `pyproject.toml` errors from `uv`: a missing `[project]` table, and `[tool.uv]` placed so the fields below it were misread.
- I diagnosed the Pinecone connectivity issue by ruling out DNS, ping, and browser causes one at a time.
- I rebuilt the same agent loop at three levels of abstraction to see exactly what `create_agent` does under a single `.invoke()` call.

