from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "monitoring" / "logs" / "prediction_log.csv"

BASELINE_FRAUD_RATE = 0.001726
DRIFT_THRESHOLD = 0.05
MINIMUM_PREDICTIONS = 100


def main():
    if not LOG_FILE.exists():
        print("RETRAINING CHECK: No monitoring data available.")
        return

    data = pd.read_csv(LOG_FILE)

    if len(data) < MINIMUM_PREDICTIONS:
        print(
            f"RETRAINING CHECK: Not enough production observations "
            f"({len(data)}/{MINIMUM_PREDICTIONS})."
        )
        return

    fraud_rate = (data["prediction"] == 1).mean()
    difference = abs(fraud_rate - BASELINE_FRAUD_RATE)

    print(f"Current fraud prediction rate: {fraud_rate:.4%}")
    print(f"Baseline fraud rate: {BASELINE_FRAUD_RATE:.4%}")
    print(f"Absolute difference: {difference:.4%}")

    if difference > DRIFT_THRESHOLD:
        print(
            "RETRAINING RECOMMENDED: "
            "Prediction distribution exceeds monitoring threshold."
        )
    else:
        print(
            "NO RETRAINING REQUIRED: "
            "Prediction distribution is within threshold."
        )


if __name__ == "__main__":
    main()
