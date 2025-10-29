from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/rf_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    expected_keys = ["countryCode", "InvoiceAmount", "PaperlessBill", "TotalInvoices", "PaidLateCount", "LateRatio"]

    for key in expected_keys:
        if key not in data:
            return jsonify({"error": f"Missing key: {key}"}), 400

    df = pd.DataFrame([data])

    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "PaidLate_Prediction": int(prediction),
        "Probability": round(float(probability), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
