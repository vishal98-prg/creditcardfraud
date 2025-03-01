import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load dataset
df = pd.read_csv("credit_card_fraud.csv")

# Handle missing values
df.fillna(0, inplace=True)

# Convert timestamp column (ensure correct column name)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=['TransactionType', 'Location'], drop_first=True)

# Aggregated transaction history
if 'UserID' in df.columns:
    df['AvgTransactionAmount'] = df.groupby('UserID')['Amount'].transform('mean')
    df['TransactionFrequency'] = df.groupby('UserID')['TransactionID'].transform('count')
    df['AmountDifference'] = df['Amount'] - df['AvgTransactionAmount']

# Drop unnecessary columns
df.drop(columns=['TransactionID', 'Timestamp'], errors='ignore', inplace=True)

# Anomaly Detection using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df['LOF_Anomaly'] = lof.fit_predict(df[['Amount']])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['IsFraud']))

# Ensure PCA does not exceed available features
num_features = X_scaled.shape[1]
n_components = min(10, num_features - 1)

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Define Features and Target
X = pd.DataFrame(X_pca)
y = df['IsFraud']

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# PCA Variance Explained Plot
plt.figure(figsize=(6, 4))
plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Explained')
plt.savefig("pca_variance.png")
plt.show()

# Feature Importance Plot
plt.figure(figsize=(8, 5))
xgb_importances = xgb_model.feature_importances_
sns.barplot(x=xgb_importances, y=X.columns)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.savefig("feature_importance.png")
plt.show()

# Fraud vs Non-Fraud Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='IsFraud', data=df, palette='coolwarm')
plt.title("Fraud vs Non-Fraud Transactions")
plt.savefig("fraud_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# LOF Anomaly Scores
plt.figure(figsize=(6, 4))
sns.histplot(df['LOF_Anomaly'], bins=30, kde=True, color='red')
plt.title("LOF Anomaly Score Distribution")
plt.savefig("lof_anomaly_distribution.png")
plt.show()
