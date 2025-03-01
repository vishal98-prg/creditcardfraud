import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Load preprocessed data
df = pd.read_csv("credit_card_fraud.csv")

# One-hot encode categorical columns (if they exist)
if 'TransactionType' in df.columns:
    df = pd.get_dummies(df, columns=['TransactionType'], drop_first=True)
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Handle missing values
df.fillna(0, inplace=True)

# Feature Engineering
threshold = df['Amount'].quantile(0.90)
df['HighRiskTransaction'] = (df['Amount'] > threshold).astype(int)

# Split features and labels
X = df.drop(columns=['IsFraud'])
y = df['IsFraud']

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Predictions
y_pred = model.predict(X_test)

# Print Evaluation Metrics
print("\n🔹 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, "fraud_detection_model.pkl")
print("\n✅ Model saved as 'fraud_detection_model.pkl'")
