# 💳 Credit Default Prediction (Risk Scoring)

## 📘 Overview
This project builds a **credit default prediction model** on the well-known **UCI “Default of Credit Card Clients” dataset**.
It demonstrates a full **end-to-end scoring workflow** — from **EDA and preprocessing**, through **model training and tuning**, to **explainability using SHAP**.

The focus is on **risk modeling**, where both **predictive performance** and **interpretability** are essential for use in **banking**, **consumer credit**, and **regulatory reporting**.

## 🧩 Objectives
- Predict whether a credit card client will **default next month**.
- Build a proper **scoring pipeline** using scikit-learn transformers.
- Handle **imbalanced classes** using:
  - downsampling experiments
  - class weights
- Evaluate threshold-based classification metrics relevant for **risk decisioning**.
- Explain model behavior via:
  - **Global SHAP** (feature importance & beeswarm)
  - **Feature dependence plots**
  - **Local SHAP waterfall plots (TP / FP / FN / TN)**

## 📊 Dataset
- **Source:** [Default of Credit Card Clients — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Rows:** 30,000 clients
- **Features:** 23 predictors
- **Target:** `default.payment.next.month` (binary 0/1)

### Key Variables
| **Type** | **Example Features**|
| ------------- | ------------- | 
| **Demographics**| `SEX`, `AGE`, `EDUCATION`, `MARRIAGE` |
| **Credit Limits** | `LIMIT_BAL` |
| **Repayment Status History**| `PAY_0` ... `PAY_6` |
| **Bill Statement Amounts**| `BILL_AMT1` ... `BILL_AMT6` |
| **Previous Payment Behavior**| `PAY_AMT1` ... `PAY_AMT6` |

## 🧠 Project Workflow
### 1️⃣ Exploratory Data Analysis & Preprocessing
*Notebook:* `01_eda_preprocessing.ipynb`

Tasks performed:
- Loaded and inspected the raw dataset.
- Checked distributions, target imbalance, and missing values.
- Detected strong right-skew in many monetary variables.
- Converted categorical variables to correct types.
- Split dataset:
  - **Train:** 80%
  - **Test:** 20%
  - Using **stratified sampling** due to imbalance.
- Built preprocessing:
  - Standard scaling for numeric features
  - No one-hot encoding needed (categoricals are ordinal)
  - Complete transformer wrapped inside a **pipeline**

 Outputs:
 - Cleaned, transformed data
 - Feature list
 - Stored preprocessing pipeline for reuse

### 2️⃣ Modeling
*Notebook:* `02_modeling.ipynb`

Models evaluated:
- Logistic Regression (baseline)
- Random Forest
- 
