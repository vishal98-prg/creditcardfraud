import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load test dataset
df = pd.read_csv("credit_card_fraud.csv")

# One-hot encode categorical columns
if 'TransactionType' in df.columns:
    df = pd.get_dummies(df, columns=['TransactionType'], drop_first=True)
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Feature Engineering
threshold = df['Amount'].quantile(0.90)
df['HighRiskTransaction'] = (df['Amount'] > threshold).astype(int)

# Split features and labels
X = df.drop(columns=['IsFraud'])
y = df['IsFraud']

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Make Predictions
y_pred = model.predict(X)

# Print Evaluation Metrics
print("\n🔹 Model Performance:")
print("Accuracy:", accuracy_score(y, y_pred))
print("\n🔹 Classification Report:")
print(classification_report(y, y_pred))
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y, y_pred))
