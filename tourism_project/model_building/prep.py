
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


root_path = Path("tourism_project")

data_path = root_path / "data" / "tourism.csv"
split_path = root_path / "data" / "splits"


# Load dataset
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)


# Remove unnecessary identity column CustomerID
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("Removed CustomerID")


# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create train and test dataframes
train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test


# Create output directory
split_path.mkdir(
    parents=True,
    exist_ok=True
)


# Save CSV files
train_path = split_path / "train.csv"
test_path = split_path / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)


print("\nFiles created successfully:")
print("Train:", train_path)
print("Test :", test_path)

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
