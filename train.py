import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully")
print(df.head())

# =========================
# 2. Split Features & Target
# =========================
X = df.drop(["Churn", "CustomerID"], axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 3. Define Columns
# =========================
categorical_features = [
    "Gender",
    "ContractType",
    "InternetService",
    "PaymentMethod"
]

numeric_features = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "SupportTickets"
]

# =========================
# 4. Preprocessing Pipeline
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# =========================
# 5. Model Pipeline
# =========================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 7. Train Model
# =========================
model.fit(X_train, y_train)

# =========================
# 8. Evaluation
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 9. Save Model (.pkl)
# =========================
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as churn_model.pkl")
