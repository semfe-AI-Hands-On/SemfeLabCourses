# Week 4 — Neural Networks from Scratch

## What this lab covers

Every time you call `loss.backward()` in PyTorch, a chain of matrix multiplications and partial derivatives runs automatically. This lab makes you implement all of that by hand — in plain NumPy, with no deep learning framework. The goal is not to build a better training loop than PyTorch. The goal is to demystify what PyTorch is actually doing, so that when something goes wrong in a real model you know where to look.

By the end of the lab you will have built a two-layer neural network (784 → 10 → 10) that trains on MNIST and reaches a meaningful accuracy — and you will have written every line of the forward pass, the backward pass, and the gradient update yourself.

---

## What this lab covers — notebook detail

### lab04_NN_from_Scratch.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Architecture design | Choosing layer sizes for MNIST (28×28 input → 784 flattened features → two layers → 10 class outputs) |
| Forward propagation | Matrix multiply → add bias → activation, repeated per layer; computing the final class scores |
| ReLU activation | Implementing ReLU as a simple elementwise threshold; why it avoids the vanishing gradient problem that sigmoid suffers from |
| Softmax & cross-entropy loss | Converting raw scores to a probability distribution; computing the loss that measures how wrong the predictions are |
| Backpropagation by hand | Deriving and implementing the gradient of the loss with respect to every weight and bias in the network using the chain rule |
| Gradient descent | Updating weights by stepping in the direction that reduces the loss; effect of learning rate |
| Training on MNIST | Loading the dataset, running the training loop, tracking loss and accuracy across epochs |

The network architecture used throughout:

```
Input (784)
    │
    ▼
┌──────────────┐
│  Dense layer  │  W1 (784×10) + b1 (10,)
└──────┬───────┘
       │  ReLU
       ▼
┌──────────────┐
│  Dense layer  │  W2 (10×10) + b2 (10,)
└──────┬───────┘
       │  Softmax
       ▼
  Output (10 classes)
```

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| NumPy | ≥ 1.24.0 | All matrix operations — no PyTorch used |
| Pandas | ≥ 2.0.0 | Loading the MNIST CSV |
| Matplotlib | ≥ 3.7.0 | Plotting loss curves and sample digits |
| Jupyter | — | Running the notebook |

No GPU required. The network is small enough to train in seconds on CPU.

---

## Setup

### 1. Pin the Python version

From the `Week4_NN_DeepLearning/` directory:

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
jupyter notebook lab04_NN_from_Scratch.ipynb
```

---

## Lab structure

```
Week4_NN_DeepLearning/
├── README.md                    ← you are here
└── lab04_NN_from_Scratch.ipynb  ← two-layer NN in pure NumPy: forward pass, backprop, MNIST
```

## Hardware notes

No GPU required. The entire network is implemented in NumPy and trains on MNIST in seconds on any modern laptop CPU. This is intentional — the point of this lab is understanding the mechanics, not maximising throughput.
