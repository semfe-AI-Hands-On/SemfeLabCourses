# Week 2 — Mathematical Foundations of Deep Learning

## What this lab covers

Deep learning is applied mathematics. This lab makes that concrete by working through the geometric and algebraic ideas that keep coming up when you read papers, debug training runs, or try to understand why a model fails. Rather than working through proofs, you will build interactive visualisations and run experiments that let you *see* the phenomena directly.

By the end of the lab you will have an intuition for why high-dimensional spaces behave so counter-intuitively, what weight matrices actually do to data, and why vanishing gradients are an inevitable consequence of stacking nonlinearities — not a bug that clever engineers just forgot to fix.

---

## What this lab covers — notebook detail

### lab2_math_foundations.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Curse of dimensionality | How the volume of a unit hypersphere collapses to zero as dimensions grow; why this breaks nearest-neighbour search in high-dimensional spaces |
| Distance collapse — the soap bubble paradox | How, in very high dimensions, almost all points in a ball concentrate near the surface; what this means for distance-based algorithms |
| Weight matrix transformations | Visualising what a learned weight matrix does to input data — rotations, scalings, projections; building geometric intuition for linear layers |
| Why nonlinearities are necessary | Demonstrating that stacking linear layers without activations collapses to a single linear transformation; what nonlinearities add |
| Gradient flow & vanishing gradients | How gradients shrink as they propagate back through many layers; why sigmoid saturates and ReLU was a practical fix |

Interactive `ipympl` widgets are used throughout — you can drag sliders to change dimensionality or layer depth and watch the plots update in real time.

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| NumPy | ≥ 1.24.0 | Numerical computation |
| SciPy | ≥ 1.10.0 | Special functions (gamma, etc.) |
| Matplotlib | ≥ 3.7.0 | Plots |
| ipympl | ≥ 0.9.0 | Interactive Jupyter widgets |
| PyTorch | ≥ 2.0.0 | Gradient-flow demonstrations |
| Jupyter | — | Running the notebook |

No GPU required. Everything in this lab runs on CPU.

---

## Setup

### 1. Pin the Python version

From the `Week2_Mathematics/` directory:

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
jupyter notebook lab2_math_foundations.ipynb
```

> **Note on interactive widgets:** The `ipympl` backend (`%matplotlib widget`) is required for the interactive plots. If you see a static image instead of an interactive widget, make sure `ipympl` installed correctly and that you are running inside Jupyter (not JupyterLab without the extension).

---

## Lab structure

```
Week2_Mathematics/
├── README.md                    ← you are here
└── lab2_math_foundations.ipynb  ← dimensionality, weight matrices, nonlinearities, gradients
```

## Hardware notes

No GPU required. All computations are CPU-based numpy and scipy. The PyTorch sections that demonstrate gradient flow are lightweight and run in seconds on any modern laptop.
