# Week 3 — Classical ML Algorithms

## What this lab covers

Neural networks are not the only tool in the box, and often not the right one. This lab covers the classical supervised and unsupervised algorithms that practitioners still reach for daily — and that form the conceptual bedrock for understanding what neural networks do differently.

The four notebooks are designed to be worked through in order. Each one focuses on a different family of problems: predicting a continuous value, classifying examples, finding natural groups, and compressing or visualising high-dimensional data. You will fit models, inspect decision boundaries, tune hyperparameters, and evaluate results using standard metrics throughout.

---

## What this lab covers — notebook detail

### lab3a_Regression.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Linear regression | Ordinary least squares from scikit-learn; interpreting coefficients |
| Ridge & Lasso | L2 and L1 regularisation; shrinkage effects on coefficients; feature selection with Lasso |
| Polynomial regression | Feature expansion to fit nonlinear relationships; the bias–variance tradeoff |
| Overfitting & hyperparameter tuning | Using cross-validation to select regularisation strength; diagnosing underfitting vs overfitting |

### lab3b_Classiffication.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| Logistic regression | Probabilistic binary and multiclass classification |
| Decision trees | Recursive partitioning; visualising learned trees; max-depth as a regulariser |
| Support Vector Machines | Hard and soft margin; kernel trick for nonlinear boundaries |
| k-Nearest Neighbours | Instance-based learning; effect of k on decision boundaries |
| Decision boundary visualisation | Side-by-side comparison of how each algorithm carves up feature space |

### lab3c_Clustering.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| K-Means | Centroid-based clustering; inertia; choosing k with the elbow method |
| DBSCAN | Density-based clustering; handling noise and irregular shapes; eps and min_samples |
| Hierarchical clustering | Agglomerative linkage; reading dendrograms |
| Evaluation metrics | Silhouette score and adjusted Rand index for comparing cluster quality |

### lab3d_DImensionality_Reduction.ipynb

| Section | What you build / explore |
|---------|--------------------------|
| PCA | Linear projection onto principal components; explained variance; scree plots |
| t-SNE | Nonlinear dimensionality reduction for visualisation; perplexity parameter |
| UMAP | Faster alternative to t-SNE; preserving global structure |
| Comparing methods | Visualising the same high-dimensional dataset with all three methods side by side |

---

## Prerequisites

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.11 | Everything |
| NumPy | ≥ 1.24.0 | Numerical arrays |
| Pandas | ≥ 2.0.0 | Data loading and manipulation |
| Matplotlib | ≥ 3.7.0 | Plots and decision boundary visualisation |
| scikit-learn | ≥ 1.3.0 | All ML algorithms and metrics |
| Jupyter | — | Running the notebooks |

No GPU required. Everything in this lab runs on CPU.

---

## Setup

### 1. Pin the Python version

From the `Week3_MLAlgorithms/` directory:

```bash
pyenv local 3.11.13
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Jupyter and open the notebooks

```bash
jupyter notebook
```

Run the notebooks in order: `lab3a` → `lab3b` → `lab3c` → `lab3d`. Each is self-contained, but following the order reinforces the progression from supervised to unsupervised to representation learning.

---

## Lab structure

```
Week3_MLAlgorithms/
├── README.md                            ← you are here
├── lab3a_Regression.ipynb               ← linear, ridge, lasso, polynomial regression
├── lab3b_Classiffication.ipynb          ← logistic regression, trees, SVM, k-NN
├── lab3c_Clustering.ipynb               ← K-Means, DBSCAN, hierarchical clustering
└── lab3d_DImensionality_Reduction.ipynb ← PCA, t-SNE, UMAP
```

## Hardware notes

No GPU required. scikit-learn runs efficiently on CPU for the dataset sizes used here. The most compute-intensive step is t-SNE on large datasets — if it feels slow, reduce the sample size in that cell.
