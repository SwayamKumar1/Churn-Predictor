import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import joblib
import os

df = pd.read_csv('TelcoData.csv')

df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.lower().str.strip()

df['MultipleLines'] = df['MultipleLines'].replace({'no phone service': 'no'})

cols_to_replace = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in cols_to_replace:
    df[col] = df[col].replace({'no internet service': 'no'})

bin_cols = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'Churn'
]
for col in bin_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['gender'] = df['gender'].map({'male': 1, 'female': 0})

cat_features = ['InternetService', 'Contract', 'PaymentMethod']
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
target = "Churn"

X = df[cat_features + num_features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

num_pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=False))
])

preprocess = ColumnTransformer([
    ("cat", cat_pipeline, cat_features),
    ("num", num_pipeline, num_features)
   ],
   remainder = 'drop'
)

logreg = LogisticRegression(
    class_weight='balanced', max_iter=1000, solver='liblinear', penalty='l2',
    C=1.0, random_state=42
)

pipe = Pipeline([
    ('preprocess', preprocess),
    ('Logreg', logreg)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(
    pipe, X_train, y_train, cv=cv, scoring='roc_auc'
)

print(f"CV ROC-AUC: mean={cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)


print("\nHoldout ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def threshold_sweep(y_true, y_prob, thresholds=np.linspace(0.1, 0.9, 17)):
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        results.append({
            "threshold": round(t, 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        })
    return pd.DataFrame(results)

tuning_df = threshold_sweep(y_test, y_prob)
print(tuning_df.sort_values("f1", ascending=False).head(10))

best_t = tuning_df.loc[tuning_df["f1"].idxmax(), "threshold"]
print(f"\n🔹 Best F1 threshold: {best_t:.2f}")

y_pred_opt = (y_prob >= best_t).astype(int)

ohe = pipe.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot']
cat_out = list(ohe.get_feature_names_out(cat_features))
final_features = cat_out + num_features
coefs = pipe.named_steps['Logreg'].coef_.ravel()
coef_df = pd.DataFrame({
    "Feature": final_features,
    "Coefficient": coefs
}).sort_values("Coefficient", ascending=False)

print(coef_df.head(10))
print(coef_df.tail(10))

os.makedirs("artifacts", exist_ok=True)
plt.figure(figsize=(10, 6))
con_cat = pd.concat([coef_df.head(7), coef_df.tail(7)])
plt.barh(con_cat['Feature'] , con_cat['Coefficient'], color='skyblue')
plt.title('Top Positive and Negative drivers of customer Churn')
plt.xlabel("Coefficient Value(impact on Churn Probability)")
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("artifacts/feature_importance.png")
plt.show()
plt.close()

joblib.dump(pipe, "artifacts/churn_lr_pipeline.joblib")


coef_df.to_csv("artifacts/feature_coefficients.csv", index=False)


print("\nOptimized Threshold Classification Report:\n", classification_report(y_test, y_pred_opt, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_opt))
print("Saved: artifacts/churn_lr_pipeline.joblib")
print("Saved: artifacts/feature_coefficients.csv")

print("Saved: artifacts/feature_importance.png")
