import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/creditcard.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CREDIT CARD FRAUD DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nAmount statistics:")
print(df["Amount"].describe())

print("\nTime statistics:")
print(df["Time"].describe())

# Class distribution
class_counts = df["Class"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
class_counts.plot(kind="bar")
plt.title("Credit Card Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
plt.close()

# Transaction amount distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Amount"], bins=100)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=300)
plt.close()

# Fraud transaction amounts
fraud = df[df["Class"] == 1]

plt.figure(figsize=(8, 5))
plt.hist(fraud["Amount"], bins=50)
plt.title("Fraudulent Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_amount_distribution.png", dpi=300)
plt.close()

print("\nFraud transaction statistics:")
print(fraud["Amount"].describe())

print("\nEDA completed successfully.")
print(f"Graphs saved to: {OUTPUT_DIR}")
