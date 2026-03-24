# Week 5 — LoRA Fine-Tuning & Retrieval-Augmented Generation (RAG)

## What this lab covers

This lab walks through two techniques that are central to working with Large Language Models in practice: **fine-tuning** and **RAG**. Rather than treating them as isolated topics, the lab builds them on top of each other so you can directly see what each one brings to the table.

You will work with a single model and a single dataset throughout both notebooks. By the end, you'll have compared the same model in three configurations — and you'll understand *why* each step matters.

### The three stages

1. **Vanilla model** — Ask the base model science questions. Most commonly it will not know how to answer them properly.
2. **Fine-tuned model (LoRA)** — Train the model on science Q&A pairs. It learns the format and picks up some knowledge.
3. **Fine-tuned model + RAG** — Give the model access to a search engine full of relevant passages. Now it can look things up before answering.

The same set of test questions will be used at every stage, so the comparison is direct.

---

## Architecture overview

There are **two separate models** in this lab, and they do very different jobs:

### Generator — Qwen2.5-0.5B

- **What it is:** A 500-million parameter language model released by Alibaba in 2024. Decoder-only transformer (same family as GPT-4, LLaMA, Claude — just much smaller).
- **What it does:** Generates text. Given a prompt, it produces a continuation. This is the model we fine-tune and the model that ultimately answers our questions.
- **Why this one:** It's small enough to fine-tune on a laptop CPU with LoRA in about 15–20 minutes, but modern and capable enough that the fine-tuning actually produces visible improvements. Apache 2.0 license.
- **HuggingFace ID:** `Qwen/Qwen2.5-0.5B`

### Embedder — all-MiniLM-L6-v2

- **What it is:** A 22-million parameter sentence-transformer model. It maps text into 384-dimensional vectors.
- **What it does:** Converts text passages (and queries) into numerical vectors that capture semantic meaning. Similar passages end up close together in vector space. This is what powers the "retrieval" part of RAG.
- **Why this one:** Tiny, fast (embeds thousands of passages in seconds on CPU), and well-established. It doesn't generate any text — it just measures similarity.
- **HuggingFace ID:** `sentence-transformers/all-MiniLM-L6-v2`

### How they work together

```
User asks a question
        │
        ▼
   ┌─────────────┐
   │  Embedder    │  ── converts the question into a vector
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  OpenSearch  │  ── finds the most similar passages in the index
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Generator   │  ── reads the retrieved passages + question, writes an answer
   └─────────────┘
```

---

## Dataset — SciQ

**SciQ** is a science question-answering dataset from AI2 (Allen Institute for AI), available on HuggingFace.

Each entry contains:
- A **question** (e.g., "What type of organism is commonly used in preparation of foods such as bread and yogurt?")
- A **correct answer** (e.g., "microorganism")
- A **support passage** — a short paragraph that contains the information needed to answer the question

We use the dataset in two ways:
- **Q&A pairs** (question + answer) go into LoRA fine-tuning — this teaches the model how to answer science questions.
- **Support passages** get embedded and stored in OpenSearch — this becomes the knowledge base that the RAG system searches at query time.

This dual use is what ties the whole lab together.

---

## Prerequisites

### Software

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.10+ | Everything |
| Docker + Docker Compose | Recent | Running OpenSearch |
| ~3 GB free disk space | — | Model weights + Docker image |

### Python packages

All dependencies are listed in the project's `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Key packages for this week:
- `transformers` — loading and running Qwen2.5
- `peft` — LoRA adapter configuration and training
- `trl` — supervised fine-tuning trainer
- `datasets` — loading SciQ from HuggingFace
- `sentence-transformers` — embedding model
- `opensearch-py` — Python client for OpenSearch
- `torch` — PyTorch (CPU)

---

## Setup

### 1. Start OpenSearch

From the directory that contains the docker-compose.yml file execute:

```bash
docker-compose up -d
```

Wait about 30 seconds, then verify it's running:

```bash
curl http://localhost:9202
```

You should see a JSON response with the cluster name and version. The Dashboards UI is available at `http://localhost:5603` if you want to poke around.

### 2. Set up Python with pyenv

We recommend **Python 3.11** — it has the best compatibility across PyTorch, transformers, and the rest of the ML stack. If you don't have it installed yet:

```bash
pyenv install 3.11.13
```

From the **project root** directory, lock this repo to 3.11:

```bash
pyenv local 3.11.13
```

This creates a `.python-version` file so anyone who clones the repo gets the right version automatically (assuming they have pyenv set up).

### 3. Create a virtual environment and install dependencies

Still from the project root:

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

The `myvenv/` folder is already in `.gitignore`, so it won't end up in the repo.

### 4. Run the notebooks in order

- **lab5a_LoRA_FineTuning.ipynb** — fine-tunes the model and saves the adapter to `models/`
- **lab5b_RAG_System.ipynb** — loads the fine-tuned model, builds the RAG pipeline, runs the final comparison

Lab 5b depends on the output of Lab 5a (the saved LoRA adapter), so run them in order.

---

## Shutting down

When you're done:

```bash
docker-compose down
```

Add `-v` if you also want to delete the indexed data:

```bash
docker-compose down -v
```

---

## Lab structure

```
Week5_LLM_FineTuning_RAG/
├── README.md                        ← you are here
├── docker-compose.yml               ← OpenSearch + Dashboards
├── lab5a_LoRA_FineTuning.ipynb      ← Part 1: fine-tuning with LoRA
├── lab5b_RAG_System.ipynb           ← Part 2: RAG system + 3-way comparison
└── models/                            ← saved LoRA adapter (created by lab5a)
```

## Hardware notes

Everything here runs on CPU. No GPU required. The fine-tuning step is the slowest part — expect around 15–20 minutes on a modern laptop. All other steps (embedding, indexing, retrieval, generation) complete in seconds.
