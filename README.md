# Enterprise Fraud Detection System

A production-ready fraud detection system for mobile money transactions using the PaySim synthetic dataset. The pipeline processes over **6.3 million transactions**, engineers behavioral and balance-based features, and trains a **LightGBM classifier** to detect fraudulent activity under extreme class imbalance (~0.13% fraud rate).

The system is fully reproducible, tracked with MLflow, and includes SHAP-based explainability for model transparency.

---

## Group Members

- Israel De La Mothe — 816037345  
- Kieron Seepersad — 816041436  
- Dylan Sinkia — 816042623  
- Tyrese Des Vignes — 816042764  

---

## Overview

This project simulates a real-world fraud detection pipeline:

- Large-scale tabular data processing (6M+ rows)
- Heavy feature engineering (balance behavior signals)
- Imbalanced classification (fraud detection)
- Model tracking with MLflow
- Explainability using SHAP
- Optional deployment via Streamlit dashboard

---

## ⚙️ Setup

### Requirements
- Python 3.10+
- pip

### Create and activate virtual environment

**Windows:**
```bash
py -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set up Jupyter kernel (optional)
```bash
pip install ipykernel
python -m ipykernel install --user --name=fraud-env --display-name "Python (fraud-env)"
```

---

## Pipeline

### 1. Data Ingestion
- **Dataset:** PaySim (synthetic financial transactions)
- **Size:** 6,362,620 rows
- **Fraud rate:** ~0.13%

### 2. Key Insights from EDA
- Fraud occurs only in `TRANSFER` and `CASH_OUT`
- **Strongest fraud signal:**
  - Sender balance becomes exactly zero
- **Fraud often:**
  - Transfers entire balance
  - Goes to empty destination accounts

### 3. Feature Engineering

We created high-signal behavioral features.

#### Most Important Features
- `exact_drain` → entire balance transferred
- `orig_zeroed` → balance becomes zero
- `orig_balance_error` → balance inconsistency
- `amount_ratio_orig` → % of balance transferred

#### Other Features
- Transaction type encoding
- Log-transformed balances
- Time features (hour/day)
- Account type indicators

**Total features:** ~25–30

### 4. Model Training

**Algorithm:** LightGBM

**Why:**
- Fast on large tabular data
- Handles imbalance well
- Strong performance on structured datasets

**Key configuration:**
- `scale_pos_weight = neg / pos`
- `metric = "average_precision"`
- `learning_rate = 0.05`
- `num_leaves = 127`

### 5. Handling Class Imbalance
- **Fraud rate:** ~0.13%
- **Techniques used:**
  - `scale_pos_weight`
  - PR-AUC as main metric

### 6. Threshold Optimization

Instead of using 0.5:
- Compute precision-recall curve
- Select threshold maximizing F1 score

### 7. Model Performance

Expected results:

| Metric     | Value              |
|------------|--------------------|
| ROC-AUC    | ~0.97 – 0.99       |
| PR-AUC     | ~0.85 – 0.95       |
| F1 Score   | High (threshold-dependent) |

> **Note:** Extremely high scores (e.g., >0.99 PR-AUC) may indicate overly simplified or synthetic patterns.

### 8. Explainability (SHAP)

SHAP explains model predictions.

**Top features:**
- `exact_drain`
- `orig_zeroed`
- `orig_balance_error`
- `amount_ratio_orig`
- `is_risky_type`

### 9. MLflow Tracking

Tracks:
- Parameters
- Metrics
- Model artifacts

**Run:**
```bash
mlflow ui
```

### 10. Streamlit Dashboard

**Launch:**
```bash
streamlit run fraud_dashboard.py
```

**Features:**
- Real-time fraud prediction
- Model insights
- Threshold tuning

---

## Visualizations

Generated plots:
- Precision-Recall Curve
- Score Distribution (log scale)
- Confusion Matrix
- Feature Importance
- SHAP Summary Plot

---

## Reproducibility

- Random seed: 42
- Stratified splits
- Deterministic pipeline

---

## Dataset

- **Name:** PaySim
- **Source:** Kaggle
- **Size:** ~500MB

**Includes:**
- Transaction type
- Amount
- Balances
- Fraud label

---

## Key Learnings

- Fraud detection is highly imbalanced
- Feature engineering is critical
- Balance-based signals dominate model performance
- PR-AUC is more informative than accuracy

---

## Limitations

- Synthetic dataset (not fully realistic)
- Static fraud patterns (no concept drift)
- Real-world systems require monitoring and retraining

---

## Future Improvements

- Time-series behavioral modeling
- Graph-based fraud detection
- Ensemble methods
- Real-time deployment optimization

---

##  AI Tools Used

- **ChatGPT** — debugging & explanations
- **Claude** — documentation refinement
- **Copilot** — code completion

All outputs were reviewed and validated manually.

---

## Project Structure

```
fraud-detection/
│
├── docs/
├── .gitignore
├── Project_4_Enterprise_Fraud_Detection_Platform.ipynb
├── README.md
├── app.py
├── fraud_dashboard.py
├── mlflow_logging_code.py
├── requirements.txt
├── test_api.py
```

## Acknowledgments

- PaySim dataset creators
- LightGBM
- SHAP
- scikit-learn
- MLflow
