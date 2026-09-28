from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "monitoring" / "logs" / "prediction_log.csv"
REPORT_FILE = BASE_DIR / "monitoring" / "monitoring_report.txt"


def main():
    if not LOG_FILE.exists():
        print("No prediction log found.")
        return

    data = pd.read_csv(LOG_FILE)

    if data.empty:
        print("Prediction log is empty.")
        return

    total = len(data)
    fraud_count = int((data["prediction"] == 1).sum())
    legitimate_count = int((data["prediction"] == 0).sum())

    fraud_rate = fraud_count / total
    average_probability = data["fraud_probability"].mean()

    # Training fraud rate was approximately 0.1726%.
    baseline_fraud_rate = 0.001726

    # Demonstration alert threshold.
    alert_threshold = 0.05

    status = (
        "ALERT: Prediction distribution should be reviewed."
        if abs(fraud_rate - baseline_fraud_rate) > alert_threshold
        else "OK: No major prediction distribution change detected."
    )

    report = f"""
CREDIT CARD FRAUD MODEL MONITORING REPORT
=========================================

Total predictions: {total}
Legitimate predictions: {legitimate_count}
Fraud predictions: {fraud_count}

Observed fraud prediction rate: {fraud_rate:.4%}
Training fraud baseline: {baseline_fraud_rate:.4%}
Average fraud probability: {average_probability:.6f}

Monitoring status:
{status}

Note:
Prediction-rate monitoring is an operational signal and does not
by itself prove feature drift or model performance degradation.
Confirmed labels are required to calculate production precision,
recall, F1 score, and other supervised performance metrics.
"""

    REPORT_FILE.write_text(report.strip() + "\n", encoding="utf-8")

    print(report)
    print(f"Monitoring report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
