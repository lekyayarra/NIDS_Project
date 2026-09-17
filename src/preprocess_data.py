import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("NIDS Data Preprocessing")
print("=======================")

# Load cleaned dataset
file_path = "dataset/cleaned_nids_data.csv"
data = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Clean column names
data.columns = data.columns.str.strip()

print("\nColumn names cleaned successfully!")

# Show traffic labels
print("\nTraffic Labels:")
print(data["Label"].value_counts())

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Convert labels into numbers
y = y.map({
    "BENIGN": 0,
    "Bot": 1
})

print("\nEncoded labels:")
print(y.value_counts())

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Replace missing values with 0
X = X.fillna(0)

print("\nInvalid values cleaned successfully!")

# Check remaining invalid values
remaining_infinite = np.isinf(X).sum().sum()
remaining_missing = X.isnull().sum().sum()

print("Remaining infinite values:", remaining_infinite)
print("Remaining missing values:", remaining_missing)

# Split data AFTER cleaning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining and testing data:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Show label distribution
print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\nPreprocessing completed successfully!")