from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "status": "success"
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input into DataFrame (IMPORTANT for pipeline)
        input_data = pd.DataFrame([{
            "Gender": data["Gender"],
            "Age": data["Age"],
            "TenureMonths": data["TenureMonths"],
            "ContractType": data["ContractType"],
            "MonthlyCharges": data["MonthlyCharges"],
            "TotalCharges": data["TotalCharges"],
            "InternetService": data["InternetService"],
            "SupportTickets": data["SupportTickets"],
            "PaymentMethod": data["PaymentMethod"]
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return jsonify({
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Invalid input format"
        })

if __name__ == "__main__":
    app.run(debug=True)
