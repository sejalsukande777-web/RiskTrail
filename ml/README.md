# ML Part - RiskTrail (Fraud Detection)

This part just answers one thing: is this transaction fraud or not.
Why it's fraud and what to do about it is handled by the other parts of the project (RAG, backend, frontend).

## Dataset

We used the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Reasons for picking this one:
- It's a real dataset, already labeled (fraud / not fraud)
- Small enough to train on a normal laptop, no GPU needed
- Easy to find, lots of people use it for practice/projects

Put the CSV file here after downloading:
```
data/creditcard.csv
```

The columns are: Time, V1 to V28 (these are encoded/hidden columns, Kaggle didn't give the real names for privacy reasons), Amount, and Class (0 = normal, 1 = fraud).

One thing to note - this dataset doesn't have merchant_id or device_id columns. So the model itself doesn't use those. device_id is used separately just for a simple rule (explained below), not by the model.

Fraud is very rare in this dataset, like 0.17% of all transactions. Keep that in mind when looking at results below.

## Model used

We went with Logistic Regression from scikit-learn.

Why this one and not something fancier - because it's simple, trains fast, and we can actually explain its output (it gives a probability, not a black box score). Since this whole project is about explaining WHY a transaction is risky, a simple model made more sense than a complicated one.

We also used `class_weight="balanced"` because fraud cases are so rare - without this the model would just predict "not fraud" for everything and still look "accurate."

Amount and Time columns are scaled before training (StandardScaler) since V1-V28 are already scaled by Kaggle.

Model is saved using joblib to `ml/model.joblib` after training.

## How to run it

```
pip install -r requirements.txt
python ml/train.py
```

This trains the model, prints the evaluation results, and saves the model file.

## Real results (from our actual run, not made up numbers)

Ran on the real dataset, 20% kept aside for testing (56,962 transactions, 98 of them fraud):

- Precision: 0.0609
- Recall: 0.9184
- F1-score: 0.1141
- ROC-AUC: 0.9722

Confusion matrix:
```
[[55475  1389]
 [    8    90]]
```
(format: [[TN, FP], [FN, TP]])

### What this actually means

Recall is high (92%) - so out of 98 real fraud cases, the model catches 90 of them. That's good.

Precision is low (6%) - meaning most of what it flags as "fraud" is actually not fraud. It flagged 1,389 normal transactions as fraud too.

This is a known tradeoff. We used class_weight="balanced" specifically to catch more fraud, and the cost of that is more false alarms. Since fraud is so rare, even a small percentage of wrong flags on normal transactions ends up being a big number.

ROC-AUC being 0.97 tells us the model is actually pretty good at ranking risky vs safe transactions overall, the low precision is more about where we set the cutoff, not that the model is bad.

### Why we're not just using accuracy

If we just check accuracy, this model would look ~98% accurate. But that number is misleading because fraud is only 0.17% of the data - even a model that never predicts fraud would score high on accuracy. That's why we're focusing on precision/recall/F1/ROC-AUC instead.

### About the false positives

1,389 normal transactions got wrongly flagged as fraud in our test set. In a real company, that means an actual investigator has to manually check 1,389 transactions that turned out to be fine, just to catch 90 real fraud cases. That's a lot of extra work, and if it happens too much, people start ignoring the flags altogether. It's a genuine tradeoff of this approach, not something we're hiding.

## predict_transaction() function

This is what the backend team will actually call. Function signature and output format should NOT be changed since it's already used as the shared format across the project.

```python
from ml.predict import predict_transaction

result = predict_transaction({
    "transaction_id": "TXN_001",
    "amount": 5000.0,
    "timestamp": "2026-09-05T02:15:00",
    "device_id": "device_unknown_99",
    "merchant_id": "M999",
    "time_seconds": 100000,
    "features": {"V1": 5.2}
})
```

Output looks like:
```python
{
    "risk_score": 100,
    "risk_level": "HIGH",
    "risk_factors": [
        "High transaction amount",
        "Unusual transaction hour",
        "New device",
        "Matches learned fraud pattern (model score)"
    ]
}
```

Notes for whoever is calling this function:
- Only `amount` is actually required. Everything else is optional, if you don't pass timestamp/device_id/features, those specific risk factors just won't show up, it won't throw an error.
- `features` should match V1-V28 if you have them. If not, it's fine to skip, the model just treats missing ones as average (0.0).
- Run `python ml/train.py` once first so `model.joblib` exists. If it doesn't exist yet, predict_transaction() will give a clear error telling you to run train.py.

## risk_level cutoffs

- 0-29 → LOW
- 30-69 → MEDIUM
- 70-100 → HIGH

These are just reasonable starting values for the demo, not something we calculated scientifically. Can be adjusted later if needed.

## How risk_factors are decided

These are simple if-else rules on top of the model's score, not another ML model:

- "High transaction amount" - if amount > 2000
- "Unusual transaction hour" - if timestamp hour is between 12am-5am
- "New device" - if device_id is not in a small hardcoded list of known devices (this is just a placeholder for the demo, a real system would check actual device history)
- "Matches learned fraud pattern (model score)" - added automatically if the model's own risk_score is 70+
- If none of these trigger, it just says "No specific risk factors identified" so the frontend always has something to show