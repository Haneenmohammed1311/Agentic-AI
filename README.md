# Agentic AI Learning "Udemy Course" 🦜🔗

This repository documents my hands-on journey learning to build AI Agents with LangChain and LangGraph, covering RAG, Tools, MCP, and production-ready agentic systems in Python.

Built while working through a Udemy course on Agentic AI. All code here is my own implementation and experimentation as I learn. 

Each lesson lives on its own branch, so `main` stays as a clean starting point while every project has its own isolated history and code.

## 🌱 Structure

| Branch | What it covers |
|---|---|
| `project/hello-world` | First LLM chain: prompt templates, basic LangChain wiring, LangSmith tracing |
| `project/search-agent` | Search agent using LangChain's `create_agent` interface, tool calling, structured outputs |

More branches will be added as I progress through the course.

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangChain** and **LangGraph** for agent orchestration
- **uv** for dependency and environment management
- **Google Gemini** (free tier) and **Ollama** (local models) as LLM providers
- **LangSmith** for tracing and debugging agent runs

## ▶️ How to run any branch

```bash
git checkout project/hello-world
uv sync
uv run python src/langchain_course/main.py
```

