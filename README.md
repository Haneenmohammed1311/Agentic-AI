## Agents Under the Hood

This branch takes the same shopping assistant agent from the search agent branch and rebuilds it three times, each time removing one layer of the framework, so the loop underneath `create_agent` becomes visible instead of hidden.

### The shared setup

All three scripts answer the same question using the same two tools, so the only thing changing between files is how the loop itself is built.

Question asked in every script: What is the price of a laptop after applying a gold discount?

`get_product_price(product)` looks up a price from a small catalog. Laptop is priced at 1299.99.

`apply_discount(price, discount_tier)` applies a named tier. Gold is 23 percent off, bronze 5, silver 12. The numbers are unusual on purpose, so the model cannot guess the final answer without actually calling the tools.

### File 1, `1_agent_loop_langchain_tool_calling.py`

This is the LangChain version, and the baseline the other two files are compared against.

`init_chat_model(f"ollama:{MODEL}")` connects to a local model through LangChain's provider agnostic interface. Changing providers later means changing one string.

The `tool` decorator turns a plain function into something the model can call. It reads the function's type hints and docstring and builds the JSON schema for you automatically.

`bind_tools()` attaches that schema to the model so every request sent to it includes what tools exist and what arguments each expects.

`llm_with_tools.invoke(messages)` sends the conversation and gets back either plain text or a request to call a tool. The model only ever asks for a tool to run, it never runs it.

`ToolMessage` carries a tool's result back into the conversation, tagged with the id of the tool call it answers, so the model can match results to requests.

What to take away from this file: this is what "it just works" looks like, and it works because LangChain is quietly building schemas, tracking ids, and formatting messages for you the whole time.

### File 2, `2_agent_loop_raw_function_calling.py`

Same agent, LangChain removed, calling `ollama` directly.

The JSON schema that `tool` built automatically in file 1 is written here by hand, as a plain dictionary with `name`, `description`, and `parameters`. Seeing it written out shows how much `tool` was actually doing.

`ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)` is the raw API call `invoke()` was wrapping in file 1.

`tool_call.function.name` and `tool_call.function.arguments` use dot notation instead of dictionary lookups, because Ollama's client returns typed objects, not plain dicts, unlike the LangChain version.

Tool results go back in as plain dictionaries, `{"role": "tool", "tool_name": tool_name, "content": str(result)}`. Ollama specifically requires the `tool_name` field here, since there is no id based matching the way `ToolMessage` uses.

What went wrong while building this file, and why it matters: `gemma3:270m` rejected the request outright, since it has no tool calling support at all, that is a model limitation, not a bug. `llama3.2:3b` does support tool calling but was unreliable, in testing it sometimes called `apply_discount` before `get_product_price`, skipping a required argument or passing a placeholder value instead of a real price. Switching to `qwen3:0.6b`, which Ollama's own documentation uses in every tool calling example, made this reliable. Wrapping the actual tool call in a try except block was the other fix, so a bad argument gets reported back to the model as text instead of crashing the script outright.

What to take away from this file: removing LangChain does not remove the underlying mechanism, structured tool calls still exist at the API level, LangChain was just formatting them for you. The real fragility in this file came from the model, not from writing the loop by hand.

### File 3, `3_raw_react_prompt.py`

Same agent again, no LangChain and no structured tool calling at all. This is the ReAct pattern, how agents were built before models had a built in concept of tool calling.

`inspect.signature()` and `inspect.getdoc()` read a function's parameters and docstring at runtime. They build the plain text tool descriptions that get written directly into the prompt, since there is no schema step anymore.

The prompt itself is the mechanism. It asks the model to write its reasoning as text, following a strict Thought, Action, Action Input, Observation format, repeating until it writes Final Answer.

`options={"stop": ["\nObservation"]}` tells Ollama to stop generating right after the model writes the word Observation, so the real tool result can be inserted instead of letting the model invent one.

A scratchpad string replaces the message list from files 1 and 2. Each loop iteration appends the model's output and the real observation onto this one growing string, and the whole thing is resent as a single prompt every iteration.

`re.search()` with patterns for Action and Action Input parses the model's raw text, since there is no structured field to read anymore.

What went wrong while building this file, and why it matters, this is the most important part of the whole branch. First attempt, the model wrote `Action: get_product_price(product="laptop")`, function call style on one line, instead of the two separate lines the prompt asked for. Adding one worked example directly into the prompt fixed this, showing that a single example is often worth more than paragraphs of instructions when steering a model through free text.

Second attempt, the model wrote an Action line for apply_discount and a Final Answer line in the same single response, before the tool had actually run. The parsing code checked for Final Answer first, found a match, and returned it immediately, so it returned 1299.99, the original undiscounted price, as if it were the final answer, without ever calling apply_discount. The fix is to always check for a real Action before ever trusting a Final Answer, since a genuine final answer should only ever appear in an output that contains no action, meaning it came after a real observation was already available.

What to take away from this file: raw text based agents are not just less convenient than structured tool calling, they can fail silently. A structured `tool_calls` field either exists or it does not, there is no ambiguity. A block of text can contain what looks like two different, contradictory decisions at once, and your parsing logic has to actively decide which one to believe. That decision is part of building the agent correctly, not a detail to skip.

### Comparing the three loops

| | File 1 | File 2 | File 3 |
|---|---|---|---|
| Reasoning shows up as | structured `tool_calls` | structured `tool_calls` | plain text |
| Parsed with | `ai_message.tool_calls` | `message.tool_calls[0].function` | regex |
| History stored as | message list, `ToolMessage` | message list, plain dicts | one scratchpad string |
| Finish detected by | tool_calls is empty | tool_calls is empty | text contains Final Answer |

### Setup

```bash
git checkout project/agents-under-the-hood
uv sync
```

```bash
ollama pull llama3.2:3b
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
```

```bash
uv run python 1_agent_loop_langchain_tool_calling.py
uv run python 2_agent_loop_raw_function_calling.py
uv run python 3_raw_react_prompt.py
```

### What this branch actually teaches

Structured tool calling, whether through LangChain or raw through an SDK, is reliable because the model's output shape is enforced by the API itself. Raw text based ReAct prompting has no such enforcement, the model can produce anything, including text that looks correct but hides a logic error, like a final answer written before the tool that should have produced it ever ran. Understanding this file by file is exactly what makes it possible to debug a framework like LangChain later, since every abstraction it offers maps onto something you have now built and broken by hand at least once.