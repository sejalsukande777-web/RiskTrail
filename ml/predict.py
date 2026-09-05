"""
ml/predict.py

Loads the trained model saved by ml/train.py and exposes predict_transaction().

predict_transaction() is a SHARED CONTRACT with the backend agent (Claude #4).
Do not rename risk_score / risk_level / risk_factors, and do not change the
return shape.
"""

import os
from typing import Optional
from datetime import datetime

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

_bundle = None  # cached model bundle, loaded once per process


def _load_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python ml/train.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# Demo-only whitelist of "known" devices. This lets the "New device" risk
# factor work in the hackathon demo without a real device-history database.
# In a production system this would be a lookup against stored account/device
# history instead of a hardcoded set.
KNOWN_DEVICES = {"device_known_1", "device_known_2", "device_known_3"}

# risk_level thresholds on the 0-100 risk_score:
#   0-29   -> LOW
#   30-69  -> MEDIUM
#   70-100 -> HIGH
LOW_MAX = 30
MEDIUM_MAX = 70

HIGH_AMOUNT_THRESHOLD = 2000.0  # amounts above this trigger "High transaction amount"
UNUSUAL_HOUR_START = 0          # 12am
UNUSUAL_HOUR_END = 5            # 5am -- hours in [0,5] count as "unusual"


def _risk_level_from_score(score: int) -> str:
    if score < LOW_MAX:
        return "LOW"
    if score < MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def _build_feature_row(transaction: dict, feature_columns: list, scaler) -> pd.DataFrame:
    """
    Builds a single-row DataFrame matching the columns the model was trained
    on (Time, V1..V28, Amount), in the same order, then scales Time/Amount
    with the scaler saved during training.
    """
    amount = transaction.get("amount")
    if amount is None:
        raise ValueError(
            "predict_transaction() requires 'amount' in the transaction dict "
            "-- got None or missing. Silently scoring a transaction with no "
            "amount as $0 would be misleading, so this is a hard error."
        )

    features = transaction.get("features", {}) or {}
    row = {}
    for col in feature_columns:
        if col == "Time":
            row["Time"] = transaction.get("time_seconds", 0.0)
        elif col == "Amount":
            row["Amount"] = amount
        else:
            # V1..V28: anonymized PCA features from the training dataset.
            # Default to 0.0 (mean) if not provided.
            row[col] = features.get(col, 0.0)

    df = pd.DataFrame([row], columns=feature_columns)
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    return df


def _hour_from_timestamp(timestamp) -> Optional[int]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(str(timestamp)).hour
    except ValueError:
        return None


def _rule_based_factors(transaction: dict, risk_score: int) -> list:
    """
    Simple, explainable rules layered on top of the model's own score.
    These do not require another model -- just readable checks on the
    transaction's fields, which is what makes them explainable to an
    investigator.
    """
    factors = []

    amount = transaction.get("amount")
    if amount is not None and amount > HIGH_AMOUNT_THRESHOLD:
        factors.append("High transaction amount")

    hour = _hour_from_timestamp(transaction.get("timestamp"))
    if hour is not None and UNUSUAL_HOUR_START <= hour <= UNUSUAL_HOUR_END:
        factors.append("Unusual transaction hour")

    device_id = transaction.get("device_id")
    if device_id and device_id not in KNOWN_DEVICES:
        factors.append("New device")

    if risk_score >= MEDIUM_MAX:
        factors.append("Matches learned fraud pattern (model score)")

    if not factors:
        factors.append("No specific risk factors identified")

    return factors


def predict_transaction(transaction: dict) -> dict:
    """
    Args:
        transaction: dict, expected keys:
            - amount (float, REQUIRED) -- transaction amount. Raises
              ValueError if missing or None; there is no silent default.
            - timestamp (str, optional) -- ISO 8601 datetime string, e.g.
              "2026-09-05T02:14:00". Used only for the "Unusual transaction
              hour" risk factor. If missing, that factor is skipped.
            - device_id (str, optional) -- used only for the "New device"
              risk factor. If missing, that factor is skipped.
            - time_seconds (float, optional) -- seconds elapsed since the
              first transaction in the training dataset (matches the
              dataset's "Time" column). Defaults to 0.0 if not provided.
            - features (dict, optional) -- {"V1": ..., ..., "V28": ...},
              the anonymized PCA features from the training dataset.
              Any missing Vx defaults to 0.0 (the dataset mean).

            transaction_id and merchant_id are accepted by the caller's
            transaction dict but are not used inside this function -- see
            ml/README.md for why (the training dataset has no merchant
            field).

    Returns:
        {
            "risk_score": int,        # 0-100
            "risk_level": str,        # "LOW" | "MEDIUM" | "HIGH"
            "risk_factors": list[str]
        }
    """
    bundle = _load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    X = _build_feature_row(transaction, feature_columns, scaler)
    fraud_proba = model.predict_proba(X)[0][1]
    risk_score = int(round(fraud_proba * 100))
    risk_level = _risk_level_from_score(risk_score)
    risk_factors = _rule_based_factors(transaction, risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
    }