# ANN-case-study-customer-churn-prediction
📌 Overview

Customer churn prediction is a critical task for businesses aiming to retain customers and reduce revenue loss. This project uses an Artificial Neural Network (ANN) built with TensorFlow and Keras to predict whether a customer is likely to leave a service.

The model analyzes customer data and identifies patterns that indicate churn behavior.

🎯 Objective
Predict whether a customer will churn (Yes/No)
Identify key factors influencing churn
Help businesses improve customer retention strategies
📂 Dataset

The dataset contains customer-related information such as:

👤 Demographics (gender, senior citizen, etc.)
📅 Account details (tenure, contract type)
📡 Services subscribed (internet, phone, streaming)
💳 Billing details (monthly charges, total charges)
🔑 Target Variable:
Churn → Yes (1), No (0)
⚙️ Methodology
🔹 Data Preprocessing
Removed unnecessary columns
Converted categorical data using encoding
Applied feature scaling using StandardScaler
🔹 Model Building
Artificial Neural Network (ANN)
2 Hidden Layers
Activation Functions:
ReLU (hidden layers)
Sigmoid (output layer)
🔹 Model Training
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 20–50
🧠 Model Architecture
Input Layer → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)
📈 Results
✅ Accuracy: ~80–85%
📊 Good performance on classification task
🔍 Key Insights:
Customers with shorter tenure are more likely to churn
Higher monthly charges increase churn probability
Month-to-month contracts have higher churn rates
💼 Business Impact

This model can help organizations:

Identify high-risk customers early
Improve customer retention strategies
Reduce marketing and acquisition costs
Enhance customer satisfaction
🚀 Future Improvements
Hyperparameter tuning for better accuracy
Implement Dropout to reduce overfitting
Add model explainability (SHAP, LIME)
Deploy using web frameworks (FastAPI/Streamlit)
