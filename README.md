# 📊 Customer Churn Prediction (Project)

A full end-to-end Machine Learning project predicting telecom customer churn and explaining why customers leave.  
Built with **Python, Scikit-Learn, and Streamlit**, this project demonstrates clean data pipelines, model interpretability, and deployable ML applications.

---

## 🚀 Demo
- 👉 **Live App:** churnapp-swayamkumar1.streamlit.app
- 👉 **Repo:** https://github.com/SwayamKumar1/Churn-Predictor/blob/main/churn_predictor.py

---

## 🧠 Project Overview

This project predicts whether a telecom customer will churn based on usage and contract features, and identifies the key drivers behind churn.

**Business Goal:**  
Help telecom companies retain customers by targeting high-risk profiles with proactive offers or contract changes.

---

## 🏗️ Tech Stack

- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Streamlit, Joblib  
- **Model:** Logistic Regression (with balanced classes & threshold optimization)  
- **Deployment:** Streamlit Cloud / GitHub Pages  

---

## 🧩 Workflow

1. **Data Preprocessing**
   - Cleaned categorical & numeric data  
   - One-hot encoding & standard scaling via `ColumnTransformer`

2. **Model Training**
   - Logistic Regression (`class_weight='balanced'`)
   - 5-fold cross-validation  
   - ROC-AUC ≈ 0.83 ± 0.006  

3. **Threshold Optimization**
   - Best F1 at threshold = 0.60 (precision = 0.55, recall = 0.73)

4. **Interpretability**
   - Extracted coefficients to rank churn drivers  
   - Visualized top positive & negative factors  

5. **Deployment**
   - Streamlit web app with interactive inputs + predictions  
   - Model and artifacts stored via `joblib`

---

## 📈 Key Metrics

| Metric | CV Mean ± Std | Holdout |
|---------|---------------|----------|
| ROC-AUC | 0.838 ± 0.006 | 0.832 |
| F1 (optimized) | 0.631 |  |

---

## 🔍 Top Insights (Feature Importance)

| Factor | Impact | Explanation |
|--------|---------|-------------|
| **Contract = Month-to-month** | ↑ Churn Risk | Short-term users are least loyal. |
| **InternetService = Fiber optic** | ↑ Churn Risk | Higher cost plans → more switching. |
| **PaymentMethod = Electronic check** | ↑ Churn Risk | Manual payments increase attrition. |
| **Contract = Two year** | ↓ Churn Risk | Long-term commitments build retention. |
| **tenure (high)** | ↓ Churn Risk | Loyal customers rarely leave. |

---

## 🧰 Files Included

| File | Purpose |
|------|----------|
| `app.py` | Streamlit demo application |
| `artifacts/churn_lr_pipeline.joblib` | Trained model pipeline |
| `artifacts/feature_coefficients.csv` | Model coefficients for interpretation |
| `artifacts/feature_importance.png` | Feature importance chart |
| `churn_predictor.py` | Training script |
| `requirements.txt` | Project dependencies |

---

## 🧾 How to Run Locally

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Run demo
streamlit run app.py
💼 About This Project
Created by Swayam Kumar as part of a professional AI/ML learning journey to build deployable ML solutions and launch a freelancing career.

Use-Case: Customer Churn Prediction
Goal: Showcase full pipeline skills – from data cleaning → modeling → deployment.

```
🌐 Connect

GitHub: https://github.com/SwayamKumar1

Email: [swayamk270@gmail.com]
# Churn-Predictor
