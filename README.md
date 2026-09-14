# Lesson 2: Modern Search Agent 🔎

Building a search agent using LangChain's `create_agent` interface, progressing from a custom tool to a built-in integration with structured output.

## 🎯 What this covers

This lesson evolves through three stages:

1. **Custom tool with `@tool`** wrote a placeholder `search` function decorated with `@tool`, wired it into `create_agent`, and learned that the agent decides whether to call a tool based entirely on its docstring description.
2. **Built-in integration with `TavilySearch`** replaced the custom placeholder with LangChain's real `TavilySearch` tool for live web search, removing boilerplate.
3. **Structured output with Pydantic** defined an `AgentResponse` schema (with nested `Source` objects) and used `response_format` so the agent returns validated, typed data instead of free-form text.
## 📦 About structured output

Normally, the agent just replies with plain text. With `response_format`, you give it a schema (like `AgentResponse`), and it returns your data already organized into that shape instead of a paragraph you'd have to parse yourself.

## 🛠️ Setup

1. `.env` file needs:
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_tavily_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=langchain-course-search-agent

2. Install dependencies:
```bash
   uv sync
```
3. Run:
```bash
   uv run python src/langchain_course/main.py
```

## 💡 What I learned

- `create_agent` expects input as `{"messages": [...]}`, not a bare list of messages
- Tool descriptions directly control whether the agent decides to call them vague or mismatched docstrings mean the agent skips the tool entirely
- `result["messages"][-1]` gives the final answer after any tool-calling steps; `result["structured_response"]` gives the typed Pydantic object when `response_format` is set


## 🧠 Tech used

- `langchain.agents.create_agent`
- `langchain_tavily.TavilySearch`
- Pydantic `BaseModel` for structured responses
- Gemini (cloud) and Ollama (local) as swappable LLM backends