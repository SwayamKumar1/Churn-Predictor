# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ---- Config ----
MODEL_PATH = Path("artifacts/churn_lr_pipeline.joblib")
COEF_IMG = Path("artifacts/feature_importance.png")
COEF_CSV = Path("artifacts/feature_coefficients.csv")

st.set_page_config(page_title="Churn Predictor Demo", layout="centered")

st.title("Customer Churn Predictor — Demo")
st.markdown(
    """
    Upload your cleaned single-row customer info or use the form below.
    Model: Logistic Regression pipeline (OneHot + StandardScaler).  
    Default threshold set to 0.60 (optimized for F1).
    """
)

# Load model (handle missing file nicely)
if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run training and save artifacts first.")
    st.stop()

pipe = joblib.load(MODEL_PATH)

# Sidebar: threshold and quick info
st.sidebar.header("Decision Threshold")
threshold = st.sidebar.slider("Churn probability threshold", 0.0, 1.0, 0.60, 0.01)

st.sidebar.markdown("**Model metrics (holdout)**")
st.sidebar.markdown("- ROC-AUC: 0.832")
st.sidebar.markdown("- Best F1 @ threshold 0.60")

# Input form
st.header("Customer input")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        InternetService = st.selectbox("Internet Service", options=["fiber optic","dsl","no"])
        Contract = st.selectbox("Contract", options=["month-to-month","one year","two year"])
        PaymentMethod = st.selectbox("Payment Method", options=[
            "electronic check",
            "mailed check",
            "bank transfer (automatic)",
            "credit card (automatic)"
        ])
    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=1e6, value=float(tenure*MonthlyCharges), step=1.0)
    submitted = st.form_submit_button("Predict")

# Predict on the provided single-row record
if submitted:
    # Build dataframe row in the expected schema / order
    X_row = pd.DataFrame([{
        "InternetService": InternetService,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])
    # Probability / prediction
    prob = pipe.predict_proba(X_row)[:,1][0]
    pred = int(prob >= threshold)

    st.metric(label="Churn probability", value=f"{prob:.3f}")
    st.markdown(f"**Predicted class (threshold {threshold:.2f})**: {'CHURN' if pred==1 else 'KEEP'}")

    # Simple explanation: top positive drivers from coef CSV
    if COEF_CSV.exists():
        coef_df = pd.read_csv(COEF_CSV)
        st.subheader("Top drivers (model Coefficients)")
        top_pos = coef_df.sort_values("Coefficient", ascending=False).head(5)
        top_neg = coef_df.sort_values("Coefficient", ascending=True).head(5)
        st.write("**Increase churn (top)**")
        st.table(top_pos.reset_index(drop=True))
        st.write("**Reduce churn (top)**")
        st.table(top_neg.reset_index(drop=True))

# Show feature importance image
st.markdown("---")
st.header("Feature importance")
if COEF_IMG.exists():
    st.image(str(COEF_IMG), use_column_width=True)
else:
    st.info("Feature importance image not found. Save artifacts first.")

st.markdown("---")
st.caption("This demo uses a pre-trained Logistic Regression pipeline. It's intended for demonstration and portfolio use only.")



