"""
rag/tools.py

Small lookup tool functions used by the investigation agent (and useful
for manual testing / the evaluation script). Plain functions, no classes.
"""

import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_THIS_DIR), "data")

SAMPLE_TRANSACTIONS_PATH = os.path.join(_DATA_DIR, "sample_transactions.json")
MERCHANT_HISTORY_PATH = os.path.join(_DATA_DIR, "merchant_history.json")


def get_transaction(transaction_id: str) -> dict | None:
    """
    Look up a demo transaction's INPUT fields (amount, timestamp, device_id,
    merchant_id, time_seconds, features) from data/sample_transactions.json,
    by transaction_id.

    IMPORTANT: this returns only the input fields needed to call
    ml.predict.predict_transaction() -- it deliberately does NOT return the
    file's `model_output` field. That field is a reference/sanity-check
    value only. risk_score / risk_level / risk_factors must always come
    from calling predict_transaction() live, never from this cached value.
    Callers (e.g. the backend, or rag/evaluate_retrieval.py) should call
    predict_transaction() themselves with the fields returned here.

    Returns None if transaction_id is not found in sample_transactions.json
    (this matches get_merchant_history()'s "not found -> None" convention
    below -- callers should check for None rather than catch an exception).

    Raises FileNotFoundError if sample_transactions.json itself is missing
    (a genuine setup/environment problem, not a "not found" lookup result).
    """
    if not os.path.exists(SAMPLE_TRANSACTIONS_PATH):
        raise FileNotFoundError(
            f"{SAMPLE_TRANSACTIONS_PATH} not found. This file is owned by "
            "the ML component (Claude #2) and should already exist in data/."
        )

    with open(SAMPLE_TRANSACTIONS_PATH, "r", encoding="utf-8") as f:
        transactions = json.load(f)

    for txn in transactions:
        if txn.get("transaction_id") == transaction_id:
            # strip model_output -- never serve the cached reference value
            return {k: v for k, v in txn.items() if k != "model_output"}

    return None


def get_merchant_history(merchant_id: str) -> dict | None:
    """
    Look up synthetic merchant history from data/merchant_history.json.
    Returns None if the merchant_id has no record (this is expected and
    fine -- merchant history is optional context, not required for every
    transaction).
    """
    if merchant_id is None:
        return None

    with open(MERCHANT_HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    return history.get(merchant_id)


if __name__ == "__main__":
    # quick manual smoke test
    txn = get_transaction("TXN_001")
    print("TXN_001 input fields:", txn)

    missing = get_transaction("TXN_DOES_NOT_EXIST")
    print("Unknown transaction_id returns:", missing)  # should print: None

    print("Merchant M002 history:", get_merchant_history("M002"))
    print("Merchant M999 (unknown, but present) history:", get_merchant_history("M999"))
    print("Merchant with no record at all:", get_merchant_history("M_DOES_NOT_EXIST"))