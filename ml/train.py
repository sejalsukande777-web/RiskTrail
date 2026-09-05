"""
ml/train.py

Trains the RiskTrail fraud detection model.

Dataset: Kaggle "Credit Card Fraud Detection" dataset (mlg-ulb)
         https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
         Download creditcard.csv and place it at: data/creditcard.csv
         (i.e. RiskTrail/data/creditcard.csv, relative to the repo root)

Run from the repo root:
    python ml/train.py

Outputs:
    - Prints real evaluation metrics (precision, recall, F1, ROC-AUC,
      confusion matrix) measured on a held-out test set.
    - Saves the trained model + scaler to ml/model.joblib
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def load_data(path):
    if not os.path.exists(path):
        print(f"ERROR: dataset not found at {path}")
        print("Download the Kaggle 'Credit Card Fraud Detection' dataset:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("and place the file as creditcard.csv at that path, then re-run this script.")
        sys.exit(1)
    return pd.read_csv(path)


def preprocess(df):
    """
    V1-V28 are already PCA-transformed by the dataset provider and are on a
    similar, roughly-standardized scale. Time and Amount are raw and on very
    different scales, so we standardize just those two before training.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    feature_columns = [c for c in df.columns if c != "Class"]
    X = df[feature_columns]
    y = df["Class"]
    return X, y, scaler, feature_columns


def main():
    df = load_data(DATA_PATH)
    X, y, scaler, feature_columns = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression: chosen because it's simple, fast to train, and its
    # output is a directly interpretable probability. For RiskTrail we need
    # to explain WHY a transaction looks risky, not just squeeze out maximum
    # accuracy, so an interpretable model beats a black-box one here.
    # class_weight="balanced" compensates for the heavy class imbalance
    # (fraud is a tiny fraction of all transactions).
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Evaluation on held-out test set ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print()
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    fraud_count_test = int(y_test.sum())
    false_positives = int(cm[0][1])
    print(f"Fraud cases in test set: {fraud_count_test}")
    print(f"False positives (legit txns flagged as fraud): {false_positives}")

    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": feature_columns},
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()