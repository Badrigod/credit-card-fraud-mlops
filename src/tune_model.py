import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 60)

# Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

X_val = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

print(f"\nTraining samples: {len(X_train)}")
print(f"Fraud samples: {positive}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Base XGBoost model
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# Focused parameter search
param_distributions = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nStarting RandomizedSearchCV...")
print("This may take several minutes.\n")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")

print(f"\nBest cross-validation ROC-AUC: {search.best_score_:.4f}")

# Evaluate best model on untouched validation set
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_prob)

cm = confusion_matrix(y_val, y_pred)

print("\n" + "=" * 60)
print("TUNED MODEL - VALIDATION RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))

# Save tuned model
MODEL_PATH = MODEL_DIR / "xgboost_fraud_model_tuned.pkl"
joblib.dump(best_model, MODEL_PATH)

# Save results
with open(REPORT_DIR / "tuning_results.txt", "w") as f:
    f.write("XGBoost Hyperparameter Tuning Results\n")
    f.write("=" * 50 + "\n\n")

    f.write("Best Parameters:\n")
    for parameter, value in search.best_params_.items():
        f.write(f"{parameter}: {value}\n")

    f.write(
        f"\nBest CV ROC-AUC: "
        f"{search.best_score_:.4f}\n"
    )

    f.write("\nValidation Results:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_val, y_pred, digits=4))

print(f"\nTuned model saved to: {MODEL_PATH}")
print("Results saved to: reports/tuning_results.txt")
print("\nHyperparameter tuning completed successfully.")
