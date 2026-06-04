# Week 11 — LLM from Scratch

## What this lab covers

Modern language models feel like black boxes — until you build one yourself. This lab takes you from a raw text file to a working GPT, implementing every piece by hand in PyTorch: the tokenizer, the attention mechanism, the transformer block, the training loop, and the sampling code. Nothing is imported from a model library; every forward pass is yours.

You start with a bigram baseline — the simplest possible language model — and progressively add self-attention, multi-head attention, feedforward layers, and positional encodings until you have a small GPT (≈1–3M parameters) trained on TinyShakespeare. The second half of the lab introduces every modern transformer upgrade (RMSNorm, SwiGLU, FlashAttention, RoPE) one at a time, runs a controlled ablation to show what each one actually contributes, then repeats the experiment on a Greek corpus to test transfer. The lab closes by fine-tuning a real 124M-parameter GPT-2 on Greek with two training recipes side by side.

---

## What this lab covers — notebook detail

### lab11_llm_from_scratch.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| 0 · Install dependencies | One-cell environment setup; CPU and GPU build options |
| 1 · Setup | Device detection, seed, TinyShakespeare download |
| 2 · Character-level tokenizer | Vocabulary from raw chars; encode/decode round-trip |
| 3 · Train/val split + batching | `get_batch` for random context windows; next-token targets |
| 4 · Bigram baseline | `nn.Embedding`-only LM; untrained vs trained generation |
| 5 · Self-attention from scratch | Q, K, V projections; causal mask; scaled dot-product formula |
| 6 · Multi-head attention | Parallel heads; concat + project back to `n_embd` |
| 7 · FeedForward | Per-token MLP with 4× expansion and GELU |
| 8 · Transformer block (pre-norm) | Residual connections + LayerNorm around attention and FFN |
| 9 · GPT model | Token + positional embeddings → N blocks → unembedding; weight tying |
| 10 · Training loop | AdamW + gradient clipping; train/val loss tracking |
| 11 · Loss curve | Matplotlib visualisation of training progress |
| 12 · Sampling | Greedy, temperature, top-k comparison |
| §S1 — RMSNorm | Drop-in LayerNorm replacement used by the Llama family |
| §S2 — SwiGLU FFN | Gated activation with three linear projections (Llama-style) |
| §S3 — FlashAttention | `F.scaled_dot_product_attention`; auto-dispatches on Ampere+ |
| §S4 — Top-p (nucleus) sampling | Nucleus cutoff that adapts to distribution shape |
| §S5 — KV cache | Illustrative single-head cache logic |
| §S6 — Attention map visualisation | Which positions does layer 0, head 0 attend to? |
| §S7 — RoPE | Rotary positional embedding; `precompute_rope_cache` + `apply_rope` |
| §S8 — BPE tokenizer | tiktoken GPT-2 encoding; vocab size vs context efficiency |
| §S9 — Save / load checkpoint | `torch.save` + `torch.load` round-trip |
| §13 · Ablation comparison | Seven variants (baseline → all-modern → best-config) on identical compute; val-loss chart + summary table |
| §14 · Greek dataset + continued pretraining | Byte-level tokenizer; baseline vs modern config vs English-pretrained continued training on Greek; transfer-learning demo |
| §15 · Fine-tune GPT-2 (124M) on Greek | Load HF `gpt2`; measure baseline Greek loss; vanilla vs modern training recipe; Greek generation samples |
| §S10–S20 — More to try | Mixed precision, GQA, MoE, curriculum learning, distillation, DPO, and more |

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| PyTorch | ≥ 2.3 | Tensor ops, autograd, `scaled_dot_product_attention` |
| NumPy | ≥ 1.26 | Array utilities |
| Matplotlib | ≥ 3.7 | Loss curves, attention maps |
| Transformers | ≥ 4.40 | Loading GPT-2 weights for §15 |
| Datasets | ≥ 2.18 | Greek Wikipedia corpus for §14 |
| Accelerate | ≥ 0.30 | Mixed-precision helpers (§15) |
| tiktoken | ≥ 0.7 | BPE tokenizer stretch goal (§S8) |
| Jupyter | — | Running the notebook |

GPU is optional. Defaults are sized for laptop CPU (~10–15 min for the main training loop). GPU configs are commented in the training cell. §15 on a CPU uses `gpt2` at reduced `max_iters`; swap `HF_MODEL = "distilgpt2"` to stay comfortable.

---

## Setup

### 1. Pin the Python version

From the `Week11_LLM_from_Scratch/` directory:

```bash
pyenv local 3.11.13
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Jupyter and open the notebook

```bash
jupyter notebook lab11_llm_from_scratch.ipynb
```

---

## Lab structure

```
Week11_LLM_from_Scratch/
├── README.md                    ← you are here
├── requirements.txt             ← all Python dependencies
└── lab11_llm_from_scratch.ipynb ← char-level GPT → ablation → Greek transfer → GPT-2 fine-tune
```

## Hardware notes

CPU is sufficient for sections 1–12 and all stretch goals up to §S9. Expect 10–15 minutes for the main 3000-iteration training run on a modern laptop CPU. The §13 ablation trains seven model variants (1500 iterations each) — roughly 20–40 minutes on CPU, 2–3 minutes on a GPU. §14 adds two more byte-level runs. §15 (GPT-2 fine-tune) takes ~3 minutes on an RTX-class GPU; CPU users should reduce `HF_ITERS` or switch to `distilgpt2`.
