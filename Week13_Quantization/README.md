# Week 13 — Quantization

## What this lab covers

Modern LLMs are too large to run as-is on most hardware. Quantization solves this by reducing the number of bits used to store each weight — but naive rounding destroys accuracy whenever a single outlier forces the scale far above the typical weight magnitude. This lab walks you through three progressively more powerful strategies for dealing with that problem, building every piece from scratch in NumPy.

You start with the simplest possible scheme — Round-to-Nearest (RTN) — and watch it fail catastrophically in the presence of a planted outlier. You then implement SVD separation (the core idea behind SVDQuant), which isolates the outlier direction and keeps it in full precision. Next you implement Hadamard rotation (the key step in QuIP#), which spreads the outlier energy uniformly across all weights so no single entry dominates the scale. The lab closes with GPTQ, the production standard for post-training quantization: a column-by-column algorithm that uses the second-order Hessian of the calibration data to compensate for each rounding error as it is introduced.

---

## What this lab covers — notebook detail

### lab13_quantization.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Setup | NumPy + Matplotlib configuration; `np.set_printoptions` |
| **Part 1 — RTN and the Outlier Problem** | |
| TODO 1 · `quantize_rtn` | Symmetric per-tensor INT-b quantization: scale, clip, round, dequantize |
| TODO 2 · Apply RTN to W | Measure Euclidean output error; visualise original vs quantized weight matrix |
| TODO 3 · `svd_separate` | Rank-k outlier extraction via full SVD; quantize the clean residual; combine outputs |
| Extension 1 | Repeat SVD separation on a random matrix with no planted outlier; compare error-vs-rank curves |
| **Part 2 — Hadamard Rotation** | |
| Pre-written · `build_hadamard` | Recursive normalized Hadamard matrix; `H @ H.T = I` |
| TODO 4 · Verify orthonormality | Check `H @ H.T ≈ I` for n = 4, 256, 1024 |
| TODO 5 · Apply Hadamard rotation | Compute `W' = WH`, `x' = H.T x`; measure max-abs reduction; compare scales |
| TODO 6 · Quantize rotated matrix | Full comparison table: Naive RTN vs Hadamard + INT4 vs SVD rank-1/2 |
| TODO 7 · QuIP# random sign matrices | Add `D2 = diag(±1)` before the Hadamard; 50-seed experiment for mean ± std |
| Extension 2 | Repeat QuIP# on a 64×64 matrix with 3 planted outlier columns; variance vs matrix size |
| **Part 3 — GPTQ-Style Optimization** | |
| TODO 8 · Column-by-column RTN | Per-column scale; no error compensation; error trajectory per column |
| TODO 9 · `gptq_quantize` | Hessian `H = 2XX.T`; `H_inv`; column-by-column rounding with error propagation to remaining columns |
| TODO 10 · Error trajectory plot | Side-by-side per-column RTN vs GPTQ; mark outlier columns |
| Extension 3 | Does column processing order matter? Sort by `H_inv` diagonal; compare left-to-right vs most-sensitive-first vs least-sensitive-first |

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| NumPy | ≥ 1.26 | All quantization math, SVD, linear algebra |
| Matplotlib | ≥ 3.7 | Weight heatmaps, histograms, error trajectories |
| Jupyter | — | Running the notebook |

No GPU required. No quantization library is used. Every operation is plain NumPy.

---

## Setup

### 1. Pin the Python version

From the `Week13_Quantization/` directory:

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
jupyter notebook lab13_quantization.ipynb
```

---

## Lab structure

```
Week13_Quantization/
├── README.md                    ← you are here
├── requirements.txt             ← all Python dependencies
└── lab13_quantization.ipynb    ← RTN → SVD separation → Hadamard rotation → GPTQ
```
