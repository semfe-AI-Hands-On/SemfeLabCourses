# Week 8 — Customer Ranking with Learning to Rank

## What this lab covers

This lab introduces **Learning to Rank (LTR)** — a supervised machine learning paradigm where the goal is not to predict a value but to order items by relevance. It is the backbone of recommendation systems, search engines, and customer prioritisation in marketing.

You will work with a single notebook and a real-world retail dataset. By the end, you will have engineered features from raw transaction records, trained a LightGBM ranking model, tuned it with Bayesian optimisation (Hyperopt), and produced a ranked list of customers by predicted revenue potential.

### The three stages

1. **Feature engineering** — Derive recency, frequency, and monetary features from raw transactions. Apply chronological train/validation/test splits to avoid data leakage.
2. **LightGBM ranker** — Train a gradient-boosted ranking model with `rank_xendcg` objective. Convert the continuous target into ordinal relevance labels via quantile binning.
3. **Hyperopt tuning** — Run 20 trials of Bayesian optimisation (TPE algorithm) over the hyperparameter space, then retrain on the best configuration and score the test set.

---

## Dataset

**Large Retail Dataset for EDA** — available on [Kaggle](https://www.kaggle.com/datasets/utkalk/large-retail-data-set-for-eda).

> **`retail_data.csv` is NOT included in this repository.** The file is 518 MB and exceeds GitHub's file size limit. You must download it manually before running the notebook.

### How to get the data

1. Go to: https://www.kaggle.com/datasets/utkalk/large-retail-data-set-for-eda
2. Download `retail_data.csv`
3. Place it in this folder (`Week8_Customer_Ranking/`) next to the notebook

The notebook loads it with a relative path (`CSV_PATH = "retail_data.csv"`), so it must be in the same directory.

| Property | Value |
|----------|-------|
| File size | ~518 MB |
| Rows | ~1,000,000 |
| Columns | 78 |
| Key columns | `customer_id`, `transaction_date`, `total_sales`, `product_category`, `age`, `income_bracket` |

---

## Architecture overview

```
Raw retail transactions (retail_data.csv)
         │
         ▼
┌──────────────────┐
│ Feature Engineering │  ── recency, frequency, monetary, category features
└────────┬─────────┘
         │ Chronological 70/15/15 split
         ▼
┌──────────────────┐
│  Quantile Binning │  ── total_sales → relevance labels 0–4
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LightGBM Ranker  │  ── rank_xendcg objective, trained per customer group
└────────┬─────────┘
         │ Hyperopt (20 TPE trials)
         ▼
┌──────────────────┐
│  Final Model      │  ── retrained on best HP, scores test set
└────────┬─────────┘
         │
         ▼
   Ranked customer list (results/ranked_scores.csv)
```

---

## Prerequisites

### Software

| Tool | Version | What for |
|------|---------|----------|
| Python | 3.10+ | Everything |

### Python packages

Install dependencies with:

```bash
pip install lightgbm hyperopt pandas numpy matplotlib seaborn scikit-learn
```

Or run the install cell at the top of the notebook (already included, commented out).

Key packages:
- `lightgbm` — gradient-boosted ranking model
- `hyperopt` — Bayesian hyperparameter optimisation (TPE)
- `pandas` / `numpy` — data manipulation
- `matplotlib` / `seaborn` — visualisation
- `scikit-learn` — preprocessing utilities

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
pip install lightgbm hyperopt pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the notebook

```bash
jupyter notebook ltr_lightgbm_hyperopt_lab.ipynb
```

The `retail_data.csv` file must be in the same directory as the notebook. The notebook loads it with a relative path.

---

## Lab structure

```
Week8_Customer_Ranking/
├── README.md                          ← you are here
├── ltr_lightgbm_hyperopt_lab.ipynb    ← main lab notebook
└── retail_data.csv                    ← NOT in repo — download from Kaggle (518 MB)
```

## Hardware notes

Everything runs on CPU. No GPU required. The Hyperopt search (20 trials) typically completes in 5–15 minutes on a modern laptop. Increase `N_EVALS` in Section 5 for a more thorough search.
