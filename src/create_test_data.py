import pandas as pd
from sklearn.model_selection import train_test_split

print("Creating Independent Test Dataset")
print("=================================")

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_nids_data.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Separate features and label
X = data.drop("Label", axis=1)
y = data["Label"]

# Create the same 80/20 split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine test features and original labels
test_data = X_test.copy()
test_data["Label"] = y_test

# Save test dataset
output_file = "dataset/nids_test_data.csv"

test_data.to_csv(
    output_file,
    index=False
)

print("\nTest dataset created successfully!")
print("Test dataset shape:", test_data.shape)
print("Saved to:", output_file)

print("\nTest labels:")
print(test_data["Label"].value_counts())