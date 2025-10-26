# src/train.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import pandas as pd
import joblib
from pathlib import Path

df = pd.read_csv("data/plat/invoices_plat.csv")

# Features available at issuance only
X = df[["countryCode", "InvoiceAmount", "PaperlessBill", "InvoiceDay", "InvoiceMonth"]].copy()
y = df["PaidLate"].astype(int)

num_cols = ["InvoiceAmount"]
cat_cols = ["countryCode", "PaperlessBill"]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")

pipe = Pipeline([("pre", pre), ("clf", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR AUC:", round(average_precision_score(y_test, y_prob), 4))

Path("data/models").mkdir(parents=True, exist_ok=True)
# Save whole pipeline so Flask can load once and score
joblib.dump(pipe, "data/models/risk_model.pkl")
print("Saved model to data/models/risk_model.pkl")
