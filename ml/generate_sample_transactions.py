"""
ml/generate_sample_transactions.py

Generates a small set of demo transactions with real V1-V28 features, so the
backend/demo can look up a transaction by a friendly ID (TXN_001, TXN_002...)
and get back something predict_transaction() can actually score.

Why this exists: the Kaggle dataset has no transaction_id, merchant_id, or
device_id -- but docs/DATA_SCHEMA.md and the demo flow (enter an ID, see a
result) assume those exist. This script bridges that gap using REAL rows
from the held-out test set (same split as train.py, so nothing here was
seen during training), with transaction_id/merchant_id/device_id/timestamp
made up on top just for the demo.

Run from the repo root, AFTER running ml/train.py at least once:
    python ml/generate_sample_transactions.py

Output:
    data/sample_transactions.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from predict import predict_transaction  # reuses the same loaded model

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.json")

# Made-up demo values layered on top of real rows -- these don't exist in the
# dataset, they're just here so the demo has something readable to show.
DEMO_DEVICES_KNOWN = ["device_known_1", "device_known_2"]
DEMO_DEVICES_NEW = ["device_unknown_77", "device_unknown_88"]
DEMO_MERCHANTS = ["M_amazon", "M_uber", "M_starbucks", "M_bestbuy"]
DEMO_TIMESTAMPS_NORMAL = ["2026-09-05T14:30:00", "2026-09-05T11:15:00"]
DEMO_TIMESTAMPS_UNUSUAL = ["2026-09-05T02:15:00", "2026-09-05T03:40:00"]


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run `python ml/train.py` first.")
        return

    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c != "Class"]

    # Same split as train.py (same random_state) so these rows are guaranteed
    # to be from the TEST set, never seen during training.
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Class"]
    )

    fraud_rows = test_df[test_df["Class"] == 1].sample(3, random_state=1)
    legit_rows = test_df[test_df["Class"] == 0].sample(3, random_state=1)
    picked = pd.concat([fraud_rows, legit_rows]).reset_index(drop=True)

    samples = []
    for i, row in picked.iterrows():
        txn_id = f"TXN_{i+1:03d}"
        is_fraud_row = bool(row["Class"] == 1)

        features = {f"V{j}": float(row[f"V{j}"]) for j in range(1, 29)}
        amount = float(row["Amount"])
        time_seconds = float(row["Time"])

        # Vary the demo fields a bit so not every sample looks the same.
        device_id = (DEMO_DEVICES_NEW if is_fraud_row else DEMO_DEVICES_KNOWN)[i % 2]
        timestamp = (DEMO_TIMESTAMPS_UNUSUAL if is_fraud_row else DEMO_TIMESTAMPS_NORMAL)[i % 2]
        merchant_id = DEMO_MERCHANTS[i % len(DEMO_MERCHANTS)]

        transaction = {
            "transaction_id": txn_id,
            "amount": amount,
            "timestamp": timestamp,
            "device_id": device_id,
            "merchant_id": merchant_id,
            "time_seconds": time_seconds,
            "features": features,
        }

        result = predict_transaction(transaction)

        samples.append({
            **transaction,
            "ground_truth_is_fraud": is_fraud_row,  # from the real dataset label
            "model_output": result,                  # what predict_transaction() actually returned
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} sample transactions to {OUTPUT_PATH}")
    for s in samples:
        print(f"  {s['transaction_id']}: ground_truth_fraud={s['ground_truth_is_fraud']}, "
              f"model_risk_level={s['model_output']['risk_level']}")


if __name__ == "__main__":
    main()