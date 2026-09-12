# Lesson 1: LangChain Hello World 👋

The first project in my Agentic AI learning journey — a basic LLM chain using LangChain's modern LCEL (LangChain Expression Language) syntax.

## 🎯 What this covers

- Setting up a `ChatGoogleGenerativeAI` model (Gemini, free tier)
- Building a prompt template with `PromptTemplate`
- Chaining prompt → model together with the `|` pipe operator
- Handling the response object and extracting readable text
- Adding LangSmith tracing to observe the chain's execution

## 🛠️ Setup

1. Create a `.env` file with:
GOOGLE_API_KEY=your_key_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=langchain-course
2. Install dependencies:
```bash
   uv sync
```
3. Run:
```bash
   uv run python src/langchain_course/main.py
```

## 💡 What I learned

- The modern LCEL pipe syntax replaces the older `LLMChain` class
- Newer Gemini models return `response.content` as a structured list, not a plain string
- LangSmith requires no code changes, just environment variables, to trace every chain run