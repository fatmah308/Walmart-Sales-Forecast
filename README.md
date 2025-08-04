# 🛍️ Walmart Sales Forecasting (Linear Regression + LightGBM)

This repository presents a predictive modeling pipeline for forecasting weekly sales for Walmart departments using historical data and engineered time-based features. We utilize both a baseline **Linear Regression** and a more powerful **LightGBM** model.

---

## 📌 Objective

- Predict weekly sales for each (Store, Dept, Date) combination.
- Perform feature engineering including time-based and lag features.
- Compare performance of Linear Regression and LightGBM.
- Generate a formatted CSV submission for final predictions.

---

## 📂 Dataset Overview

- **train.csv**: Historical sales data including store, department, weekly sales, and holiday flag.
- **test.csv**: Data without the `Weekly_Sales` column, for prediction.
- **features.csv**: Additional economic and promotional data (e.g., CPI, Unemployment, Markdowns).
- **stores.csv**: Store-level metadata like store type and size.

---

## ⚙️ Workflow

### 🔹 1. Data Preprocessing

- Date columns converted to `datetime`.
- Datasets merged: `train`, `test` with `features` and `stores`.
- Missing values handled:
  - Markdown columns filled with `0`.
  - CPI and Unemployment forward filled.

### 🔹 2. Feature Engineering

- Extracted time features: Year, Month, Week, Day, MonthStart/End.
- Lag features:
  - **Lag-1 Weekly Sales** per (Store, Dept)
  - **Rolling Mean of 4 weeks** (excluding current week)

### 🔹 3. Model Training

- Split based on date:
  - Train set: Before **July 2012**
  - Validation set: From **July 2012** onward
- Models Used:
  - ✅ Linear Regression (baseline)
  - ✅ LightGBM Regressor (main model)

### 🔹 4. Evaluation

- Metric: **RMSE (Root Mean Squared Error)**
- Visualization of actual vs predicted weekly sales.

### 🔹 5. Prediction & Submission

- Lag features in test set filled with `0` (due to unavailability).
- Predictions made using the LightGBM model.
- Submission CSV generated with:


---

## 🛠️ Dependencies

```bash
pandas
numpy
matplotlib
scikit-learn
lightgbm
