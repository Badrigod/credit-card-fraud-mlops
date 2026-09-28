import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CREDIT CARD FRAUD - DATA PREPROCESSING")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print(f"Training:   {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")

# Scale Time and Amount
scaler = StandardScaler()

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_val[["Time", "Amount"]] = scaler.transform(
    X_val[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# Save scaler for production API
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# Save processed datasets
X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

print("\nClass distribution:")

for name, target in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    fraud = int(target.sum())
    legitimate = len(target) - fraud
    percentage = target.mean() * 100

    print(
        f"{name}: "
        f"{legitimate} legitimate, "
        f"{fraud} fraud "
        f"({percentage:.4f}% fraud)"
    )

print("\nScaling:")
print("Time and Amount standardized using StandardScaler.")
print("Scaler fitted ONLY on training data.")
print("V1-V28 left unchanged.")

print("\nFiles saved to:", PROCESSED_DIR)
print("Scaler saved to:", MODEL_DIR / "scaler.pkl")
print("\nPreprocessing completed successfully.")
