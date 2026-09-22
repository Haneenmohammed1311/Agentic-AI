# Agentic AI Learning Journey

This repository documents my hands on journey learning to build AI Agents with LangChain and LangGraph, covering RAG, Tools, MCP, and production ready agentic systems in Python.

Each lesson lives on its own branch, so `main` stays as a clean starting point while every project has its own isolated history and code.

## Structure

## Structure

| Branch | What it covers |
|---|---|
| [project/hello-world](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/hello-world) | First LLM chain, prompt templates, basic LangChain wiring, LangSmith tracing |
| [project/search-agent](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/search-agent) | Search agent evolving from a custom tool to built-in `TavilySearch`, using `create_agent`, structured Pydantic outputs |
| [project/agents-under-the-hood](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/agents-under-the-hood) | The same agent loop rebuilt three ways, LangChain tool calling, raw Ollama function calling, and a raw ReAct prompt with regex parsing, to see what `create_agent` hides |
| [project/rag-gist](https://github.com/Haneenmohammed1311/Agentic-AI/tree/project/rag-gist) | A full RAG pipeline, ingestion into a vector store, then the same retrieval logic implemented manually and again with LCEL |

More branches will be added as I progress through the course.

## Tech stack

- **Python 3.10+**
- **LangChain** and **LangGraph** for agent orchestration
- **uv** for dependency and environment management
- **LLM providers used across branches**, Google Gemini, local Ollama models (`llama3.2:3b`, `qwen3:0.6b`, `qwen3:1.7b`), swapped in and out depending on the lesson's needs, rate limits, and whether the goal is speed or full offline testing
- **Chroma**, a local vector store, used in place of Pinecone throughout
- **HuggingFace `sentence-transformers`**, used locally for embeddings in place of OpenAI's embedding API
- **LangSmith** for tracing and debugging agent and chain runs, each branch uses its own project name to keep traces separated

## How to run any branch

```bash
git checkout project/hello-world
uv sync
uv run python src/langchain_course/main.py
```

Each branch's own README has the exact run commands and required environment variables for that lesson specifically.

## Why the tech stack differs from the instructor's course

The original course is built around OpenAI, for both embeddings and chat, and Pinecone as the vector store. Two practical constraints shaped different choices here.

`app.pinecone.io` was consistently unreachable from my network, confirmed across multiple days, multiple networks, and multiple devices, while Pinecone's marketing site loaded fine, pointing to a routing issue outside my control rather than anything fixable on my end. Chroma, a fully local vector store needing no account or network access at all, replaced it, with identical LangChain interfaces so the actual RAG logic needed no real changes.

OpenAI's embeddings and chat API both require paid billing with no meaningful free tier. Gemini's free tier and local Ollama models cover the same role at no cost, with the tradeoff of Gemini's daily request quota, and small local models occasionally following multi step tool instructions less reliably than a stronger model would.

## Challenges solved along the way

- Fixed `response.content` returning a structured list instead of plain text with newer Gemini models, and saw firsthand why LCEL's `StrOutputParser` solves this automatically while manual implementations do not
- Diagnosed and resolved a package name collision between a local `schema.py` file and an installed PyPI package also named `schema`
- Hit and worked around Gemini's free tier daily rate limit by switching to local Ollama models during heavy debugging
- Debugged a raw ReAct agent that returned a wrong final answer without crashing, tracing it back to a parsing order bug, checking for `Final Answer` text before checking for a genuine `Action`, allowing the model to skip a tool call it should have made
- Diagnosed a persistent Pinecone connectivity issue by ruling out DNS, ping, and browser level causes one at a time, down to a specific subdomain timing out consistently across networks and devices
- Rebuilt the same agent loop at three different levels of abstraction to understand exactly what `create_agent` and LangChain's helper functions do underneath a single `.invoke()` call

## Repository history

This repository originally began as a fork of the course's own repository. Its git history has since been fully detached, every branch now contains only commits authored directly for this personal learning project, with no shared commit history, automation branches, or content inherited from the original course repository.