import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("NIDS Machine Learning Model")
print("===========================")

# Load dataset
file_path = "dataset/cleaned_nids_data.csv"
data = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Clean column names
data.columns = data.columns.str.strip()

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

# Convert labels into numbers
y = y.map({
    "BENIGN": 0,
    "Bot": 1
})

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Replace missing values
X = X.fillna(0)

print("\nData preprocessing completed!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest model...")

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "Bot"]
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model_file = "models/nids_random_forest.pkl"
joblib.dump(model, model_file)

print("\nModel saved successfully!")
print("Saved to:", model_file)