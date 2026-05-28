# Week 12 — Agentic AI

## What this lab covers

This lab introduces **AI agents** — systems where a language model doesn't just respond, but decides what to do, calls tools, observes the results, and reasons its way to an answer in a loop. This is the foundation of modern agentic applications.

You will work with two notebooks that implement the **same agent** at two levels of abstraction. By the end, you will understand what an agent actually is under the hood, and how frameworks like LangGraph abstract away the loop so you can focus on logic rather than plumbing.

### The two notebooks

1. **Simple ReAct Agent from Scratch** — Build the Reasoning + Acting loop entirely in raw Python using the OpenAI API directly. No frameworks. Every step of the loop is visible.
2. **Simple ReAct Agent with LangGraph** — Rebuild the same agent using LangChain tools and LangGraph's state machine. Compare how the framework handles routing, tool binding, and message state.

### The ReAct pattern

```
User question
      │
      ▼
  ┌──────┐    Thought: do I need a tool?
  │ LLM  │ ─────────────────────────────→  if yes: emit Action
  └──────┘
      │
      ▼ (if tool called)
  ┌──────────┐
  │   Tool    │  ── runs the function, returns result
  └──────────┘
      │
      ▼
  Observation fed back to LLM → next Thought
      │
      ▼ (when LLM emits Final Answer)
  Response to user
```

The agent in both notebooks has two tools: a `calculate` tool (safe arithmetic evaluator) and an `average_dog_weight` tool (a toy lookup function). These are intentionally simple so the focus stays on the agent loop, not the tools themselves.

---

## Prerequisites

### API key

Both notebooks require an **OpenAI API key**. Copy the provided `.env_example` to `.env` and fill in your key:

```bash
cp .env_example .env
# then edit .env and set OPENAI_API_KEY="sk-..."
```

The notebooks load the key automatically via `python-dotenv`.

### Software

| Tool | Version |
|------|---------|
| Python | 3.10+ |

### Python packages

**Notebook 1 (from scratch):**
```bash
pip install openai httpx python-dotenv
```

**Notebook 2 (LangGraph):**
```bash
pip install langchain langchain-openai langgraph python-dotenv
```

Or run the install cells at the top of each notebook (already included).

---

## Setup

### 1. Set up Python with pyenv

We recommend **Python 3.11**:

```bash
pyenv install 3.11.13
pyenv local 3.11.13
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myvenv
source myvenv/bin/activate   # Windows: myvenv\Scripts\activate
pip install openai httpx python-dotenv langchain langchain-openai langgraph
```

### 3. Configure your API key

```bash
cp .env_example .env
# Edit .env: OPENAI_API_KEY="sk-..."
```

### 4. Run the notebooks in order

- **Simple_Agent_from_Scratch.ipynb** — start here; no framework dependencies
- **Simple_Agent_with_LangGraph.ipynb** — run after, compare the two approaches

---

## Lab structure

```
Week12_Agentic_AI/
├── README.md                            ← you are here
├── .env_example                         ← copy to .env and add your API key
├── Simple_Agent_from_Scratch.ipynb      ← Lab 12a: ReAct loop in raw Python
└── Simple_Agent_with_LangGraph.ipynb    ← Lab 12b: same agent via LangGraph
```

## Notes

- A local/free **Ollama** option is described at the end of notebook 12b if you don't have an OpenAI key.
- The agent loop in notebook 12a is intentionally verbose — every decision point is printed so you can follow the reasoning step by step.
- Notebook 12b produces the same behaviour with ~60% less boilerplate, which illustrates why frameworks exist.
