A machine learning project for predicting the probability that an invoice will be paid late. Built with Python, scikit-learn, and Flask.
factoring-risk-scorer/
│
├── data/
│   ├── raw/              # Original invoices data
│   ├── silver/           # Cleaned data
│   ├── plat/             # Final feature set for training
│   ├── gold/             # Final output data
│
├── models/               # Saved trained models (.pkl)
│
│
├── notebooks/            # Exploratory notebooks
│
├── src/
│   ├── data_preparation.ipynb  # Data cleaning and feature creation
│   ├── train_lr.py             # Logistic Regression training
│   ├── train_rf.py             # Random Forest training
│   ├── api.py                  # Flask API for predictions
│
└── README.md

Setup

Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

Install dependencies
pip install -r requirements.txt


Train the model
python src/train_lr.py
or
python src/train_rf.py

Trained models will be saved in the models/ folder.

Running the API

Start the Flask server:
python src/api.py

You should see:
Running on http://127.0.0.1:5000

Using Postman

Send a POST request to:
http://127.0.0.1:5000/predict

Set Headers:
Content-Type: application/json

Set Body (raw JSON):
{
  "countryCode": "391",
  "InvoiceAmount": 250.75,
  "PaperlessBill": "Electronic",
  "TotalInvoices": 8,
  "PaidLateCount": 2,
  "LateRatio": 0.25
}

Expected Response:
{
    "PaidLate_Prediction": 0,
    "Probability": 0.34
}

Model Outputs

Confusion Matrix and ROC Curve are saved in the reports/ folder.

Model file: models/risk_model.pkl

Notes

Logistic Regression (train_lr.py) is the main production model.

Random Forest (train_rf.py) is used for performance comparison.

The API loads the trained model and provides real time scoring through Flask.
