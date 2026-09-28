import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("XGBOOST CREDIT CARD FRAUD MODEL TRAINING")
print("=" * 60)

# Load training and validation data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# Calculate class weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nTraining class counts:")
print("Legitimate:", negative)
print("Fraud:", positive)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Build model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Training completed.")

# Validation predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Metrics
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_prob)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))

# Save model
MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

# Save validation metrics
with open(REPORT_DIR / "validation_metrics.txt", "w") as f:
    f.write("XGBoost Credit Card Fraud Detection\n")
    f.write("Validation Results\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_val, y_pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_val, y_pred, digits=4))

print("Validation metrics saved to reports/validation_metrics.txt")
print("\nStep 6 completed successfully.")
