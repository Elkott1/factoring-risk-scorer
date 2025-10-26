import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

# load your gold dataset
df = pd.read_csv("data/plat/invoices_plat.csv")

# define features and target
X = df[["countryCode", "InvoiceAmount", "PaperlessBill", "InvoiceDay", "InvoiceMonth"]]
y = df["PaidLate"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))

# save model
joblib.dump(model, "data/models/rf_model.pkl")
print("Saved Random Forest model to data/models/rf_model.pkl")
