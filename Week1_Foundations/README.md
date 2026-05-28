# Week 1 — Computational Foundations

## What this lab covers

Before any model can be trained, the data has to exist somewhere in memory and move through hardware. This lab builds intuition for *how* that actually works — not as a theoretical exercise, but through direct experimentation with PyTorch tensors.

You will get hands-on with the low-level mechanics that every deep learning framework rests on: how tensors are laid out in memory, why broadcasting works (and when it silently does the wrong thing), how PyTorch tracks operations to compute gradients automatically, and how to move computation between CPU and GPU. Everything here underpins the more complex models you'll build in later weeks.

---

## What this lab covers — notebook detail

### lab1_tensors.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Memory layout & row-major order | How multidimensional arrays are stored as a flat sequence of bytes; what strides are and why they matter for performance |
| Dense vs sparse tensors | When sparsity is a first-class concern; how PyTorch represents and operates on sparse formats |
| Broadcasting | The rules that let tensors of different shapes interact; common pitfalls and how to debug shape mismatches |
| Autograd & computational graphs | How PyTorch builds a dynamic computation graph as you write operations; how `.backward()` walks it to compute gradients |
| GPU vs CPU device management | How to create tensors on a specific device, move them, and check where they live; what happens when two tensors are on different devices |

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| PyTorch | ≥ 2.0.0 | Tensor operations and autograd |
| Jupyter | — | Running the notebook |

No GPU required. Everything in this lab runs on CPU.

---

## Setup

### 1. Pin the Python version

From the `Week1_Foundations/` directory:

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
jupyter notebook lab1_tensors.ipynb
```

---

## Lab structure

```
Week1_Foundations/
├── README.md               ← you are here
└── lab1_tensors.ipynb      ← tensor fundamentals: memory, broadcasting, autograd, devices
```

## Hardware notes

No GPU required. All experiments run on CPU. The sections that demonstrate device management will detect whether a GPU is available and gracefully fall back to CPU if not.
