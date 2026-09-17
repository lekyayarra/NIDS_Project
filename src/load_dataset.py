import pandas as pd

print("NIDS Dataset Analysis")
print("=====================")

# Load dataset
file_path = "dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv"
data = pd.read_csv(file_path)

# 1. Dataset size
print("\n1. Dataset Size:")
print(data.shape)

# 2. Number of rows
print("\n2. Number of Records:")
print(len(data))

# 3. Number of columns
print("\n3. Number of Features:")
print(len(data.columns))

# 4. Column names
print("\n4. Column Names:")
for column in data.columns:
    print(column)

# 5. Labels
print("\n5. Traffic Labels:")
print(data[" Label"].value_counts())
# 6. Check missing values
print("\n6. Missing Values:")
missing_values = data.isnull().sum()

print(missing_values[missing_values > 0])
# 7. Check duplicate rows
print("\n7. Duplicate Rows:")
print(data.duplicated().sum())
# 8. Remove duplicate rows
data = data.drop_duplicates()

print("\n8. Dataset after removing duplicates:")
print(data.shape)

# 9. Fill missing values
data["Flow Bytes/s"] = data["Flow Bytes/s"].fillna(0)

print("\n9. Missing values after cleaning:")
print(data["Flow Bytes/s"].isnull().sum())
# 10. Save cleaned dataset
output_file = "dataset/cleaned_nids_data.csv"

data.to_csv(output_file, index=False)

print("\n10. Cleaned dataset saved successfully!")
print("Saved to:", output_file)