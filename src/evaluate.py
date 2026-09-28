import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FINAL XGBOOST MODEL - TEST SET EVALUATION")
print("=" * 60)

# Load untouched test set
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

# Load tuned model
model = joblib.load(
    MODEL_DIR / "xgboost_fraud_model_tuned.pkl"
)

print(f"\nTest samples: {len(X_test)}")
print(f"Legitimate transactions: {(y_test == 0).sum()}")
print(f"Fraud transactions: {(y_test == 1).sum()}")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)
roc_auc = roc_auc_score(
    y_test, y_prob
)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    digits=4
)
print(report)

# Save textual results
with open(
    REPORT_DIR / "final_test_results.txt",
    "w"
) as f:

    f.write(
        "Final XGBoost Credit Card Fraud Detection Results\n"
    )
    f.write("=" * 55 + "\n\n")

    f.write(f"Test samples: {len(X_test)}\n")
    f.write(
        f"Legitimate: {(y_test == 0).sum()}\n"
    )
    f.write(
        f"Fraud: {(y_test == 1).sum()}\n\n"
    )

    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nClassification Report:\n")
    f.write(report)

# Confusion matrix figure
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("XGBoost Fraud Detection - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Card Fraud Detection")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

print(
    "\nResults saved to "
    "reports/final_test_results.txt"
)

print(
    "Confusion matrix saved to "
    "reports/figures/confusion_matrix.png"
)

print(
    "ROC curve saved to "
    "reports/figures/roc_curve.png"
)

print("\nFinal test evaluation completed successfully.")
