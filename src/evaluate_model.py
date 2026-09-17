import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("NIDS Model Evaluation")
print("=====================")

# Load dataset
data = pd.read_csv("dataset/cleaned_nids_data.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

# Convert labels
y = y.map({
    "BENIGN": 0,
    "Bot": 1
})

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Clean invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("models/nids_random_forest.pkl")

print("\nModel loaded successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "Bot"]
    )
)

# -----------------------------
# 1. Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "Bot"]
)

disp.plot()
plt.title("NIDS Confusion Matrix")
plt.tight_layout()

plt.savefig("reports/confusion_matrix.png")
plt.savefig("reports/feature_importance.png")

# -----------------------------
# 2. Feature Importance
# -----------------------------

importance = model.feature_importances_

feature_names = X.columns

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot top 10 features
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig("reports/feature_importance.png")
plt.show()

print("\nEvaluation completed successfully!")
print("Graphs saved in the reports folder.")