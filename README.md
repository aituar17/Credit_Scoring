# 💳 Credit Risk Scoring Model (Default Prediction)

## 📘 Overview
This project builds a **credit default scoring model** using the well-known **Default of Credit Card Clients** dataset from UCI.
It demonstrates the complete lifecycle of a risk analytics pipeline:
- **Data cleaning & preprocessing**
- **Imbalanced classification modeling**
- **Threshold-tuned evaluation (credit risk perspective)**
- **Global & local explainability using SHAP**
- **Risk-aligned insight extraction**
The goal is to predict whether a client will **default on their next payment**, while ensuring the model remains **transparent, interpretable, and regulator-friendly**.

## 🧩 Objectives
- Predict **next-month payment default** for credit card clients.
- Explore demographic + financial behavior data (limits, bill statements, repayment history).
- Handle strong **class imbalance** using:
  - `class_weight = "balanced"`
  - Precision-Recall analysis
- Compare **interpretable** (Logistic Regression) and **non-linear** (Tree-based) models.
- Provide **SHAP-based explainability**:
  - Global feature drivers
  - Feature interactions
  - Local risk explanations (TP / FP / FN / TN)

## 📊 Dataset
- **Source:** [Default of Credit Card Clients — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Size:** 30,000 observations
- **Target:** `default.payment.next.month` (0 = no default, 1 = default)
- **Features:** 23 variables including:
  - **Demographics:** `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - **Financial capacity:** `LIMIT_BAL`
  - **Payment history (6 months):** `PAY_0`, `PAY_2`, … `PAY_6`
  - **Bill statements (6 months):** `BILL_AMT1`–`BILL_AMT6`
  - **Previous payments (6 months):** `PAY_AMT1`–`PAY_AMT6`

### Key Characteristics
- Highly **imbalanced** target (approx. 22% default).
- Strongly skewed financial values.
- Payment delay variables (`PAY_X`) are the dominant signal.

## 🧠 Project Workflow
### 1️⃣ Exploratory Data Analysis & Preprocessing
📄 `notebooks/01_eda_preprocessing.ipynb`

Key steps:
- Inspected distributions of financial variables.
- Treated zero-inflation & skewness (logs where needed).
- Checked correlations among repayment, bill, and payment variables.
- Engineered:
  - **Debt ratio features**
  - **Payment behavior indicators**
- Split into train/test (stratified).
- Prepared preprocessing pipeline:
  - Standard scaling
  - No one-hot encoding needed (most features numeric)
 
Saved cleaned dataset → `data/df_cleaned.csv`

### 2️⃣ Modeling
📄 `notebooks/02_modeling.ipynb`

Approach:
- Built unified `sklearn` pipeline:
  - `ColumnTransformer` → **Scaler**
  - Classifier (`LogisticRegression` or `RandomForestClassifier`)
- Performed grid search on:
  - Regularization (LogReg)
  - Number of trees (RF)
  - Class weighting

#### 🔍 Performance Metrics (threshold-free)
| **Model** | **ROC-AUC**| **PR-AUC** | **Notes** |
| ------------- | ------------- | ------------- | ------------- |
| **Logistic Regression** | **0.771**| 0.532 | Transparent baseline |
| **Random Forest** | **0.767**| 0.545 | Captures non-linear relations. Slightly higher PR-AUC |

#### 📌 Threshold Analysis (credit-risk oriented)
Optimized threshold based on:
- Maximizing **Recall** (avoid missing defaulters)
- Balanced **Recall–Precision** tradeoff
- Minimized financial loss (banking perspective)

Saved artifacts:
- `models/artifacts/credit_model.joblib`
- `models/artifacts/metadata.json`

### 3️⃣ Explainability
📄 `notebooks/03_explainability.ipynb`

Explainability is crucial in credit scoring for compliance with **regulation, fairness, and model risk management**.

✔ **Global SHAP**
- SHAP summary (bar)
- Beeswarm
- Top feature contributions
- Dependence plots for most influential variables (`PAY_0`, `PAY_2`, `BILL_AMT1`, etc.)

✔ **Local SHAP**
Produced waterfall plots for:
- **TP** (correctly identified defaulter)
- **FP** (wrongly flagged as default → fairness concern)
- **FN** (missed defaulter → financial risk)
- **TN** (correct non-defaulter)
These help auditors, risk managers, and regulators understand **WHY** the model made a particular prediction.

## 📈 Key Findings
### 🔍 Top Drivers of Default
*(Based on SHAP mean absolute values computed on the test set)*
| **Rank** | **Feature**| **Effect on Default Risk** |
| ------------- | ------------- | ------------- |
| 1 | `PAY_1 = 2` | Recent 2-month delay → **strongly increases** risk |
| 2 | `PAY_1 = 0` | On-time behavior → **reduces** risk |
| 3 | `PAY_2 = 2` | Past delays (August) → **increases** risk |
| 4 | `LIMIT_BAL` & `log_limit_bal` | Higher credit limit → **reduces** risk |
| 5 | `PAY_3 = 2`, `PAY_4 = 2`, `PAY_5 = 2` | Consistent historical delays → **increases** risk |
| 6 | `PAY_AMT1` (and logs) | Larger recent repayments → **reduces** risk |
| 7 | `BILL_AMT1`, `BILL_AMT2` | Higher outstanding bills → **increases** risk |

**Interpretation:**
Default is driven primarily by **recent payment delay**, **historical delinquency**, and indicators of **financial stress** (high bills + low repayments).
Credit limits and repayment amounts provide strong stabilizing signals.

## 🧩 Visualizations

## 🧮 Repository Structure
```plaintext
credit_scoring_model/
├── data/
│   ├── default_of_credit_card_clients.csv
│   └── df_preprocessed.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_summary_bar.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_dependence_*.png
│   └── local/
│       ├── shap_local_TN.png
│       ├── shap_local_FP.png
│       ├── shap_local_FN.png
│       └── shap_local_TP.png
├── models/
│   └── artifacts/
│       ├── credit_model.joblib
│       └── metadata.json
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── requirements.txt
└── README.md
```
