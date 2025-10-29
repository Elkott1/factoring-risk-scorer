import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

df = pd.read_csv("data/plat/invoices_plat.csv")

X = df[[
    "countryCode",
    "InvoiceAmount",
    "PaperlessBill",
    "TotalInvoices",
    "PaidLateCount",
    "LateRatio"
]]
y = df["PaidLate"].astype(int)

num_cols = ["InvoiceAmount", "TotalInvoices", "PaidLateCount", "LateRatio"]
cat_cols = ["countryCode", "PaperlessBill"]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", pre),
    ("rf", rf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/rf_model.pkl")


print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR AUC:", round(average_precision_score(y_test, y_prob), 4))

ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(pipe, X_test, y_test)
plt.title("Random Forest ROC Curve")
plt.show()



print("Saved Random Forest model to models/rf_model.pkl")
