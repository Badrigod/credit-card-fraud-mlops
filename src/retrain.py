from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

X_TRAIN = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

X_VAL = pd.read_csv(DATA_DIR / "X_val.csv")
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

print("Training candidate model...")

model.fit(X_TRAIN, y_train)

predictions = model.predict(X_VAL)
probabilities = model.predict_proba(X_VAL)[:, 1]

precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
roc_auc = roc_auc_score(y_val, probabilities)

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

candidate_path = MODEL_DIR / "xgboost_fraud_model_candidate.pkl"

joblib.dump(model, candidate_path)

report = f"""
AUTOMATED RETRAINING REPORT
===========================

Candidate model validation results:

Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Candidate model:
{candidate_path.name}

Status:
Candidate model created successfully.
Production promotion requires validation before replacement.
"""

(REPORT_DIR / "retraining_report.txt").write_text(
    report.strip() + "\n",
    encoding="utf-8"
)

print(report)
