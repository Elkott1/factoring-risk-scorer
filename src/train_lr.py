from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

df = pd.read_csv("data/plat/invoices_plat.csv")

X = df[[
    "countryCode",
    "InvoiceAmount",
    "PaperlessBill",
    "TotalInvoices",
    "PaidLateCount",
    "LateRatio"
]].copy()
y = df["PaidLate"].astype(int)

num_cols = ["InvoiceAmount", "TotalInvoices", "PaidLateCount", "LateRatio"]
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

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/risk_model.pkl")

print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR AUC:", round(average_precision_score(y_test, y_prob), 4))

ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(pipe, X_test, y_test)
plt.title("ROC Curve")
plt.show()


print("Saved model to models/risk_model.pkl")
