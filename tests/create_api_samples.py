import json
import joblib
import pandas as pd

X = pd.read_csv("data/processed/X_test.csv")
y = pd.read_csv("data/processed/y_test.csv").squeeze("columns")

scaler = joblib.load("models/scaler.pkl")

legit_index = y[y == 0].index[0]
fraud_index = y[y == 1].index[0]

legit = X.loc[legit_index].tolist()
fraud = X.loc[fraud_index].tolist()

# Convert scaled Time and Amount back to original values.
# The FastAPI application will scale them again before prediction.
for values in [legit, fraud]:
    original = scaler.inverse_transform(
        [[values[0], values[-1]]]
    )[0]

    values[0] = float(original[0])
    values[-1] = float(original[1])

with open("tests/legitimate_sample.json", "w") as f:
    json.dump({"features": legit}, f)

with open("tests/fraud_sample.json", "w") as f:
    json.dump({"features": fraud}, f)

print("Samples created successfully.")
print("Legitimate index:", legit_index)
print("Fraud index:", fraud_index)
