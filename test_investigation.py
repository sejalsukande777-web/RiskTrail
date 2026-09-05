from rag.investigation_agent import investigate_transaction
from rag.tools import get_transaction
import json

# use a real transaction from your sample data
txn_id = "TXN_001"
transaction = get_transaction(txn_id)

# fabricated risk output for testing (replace with real predict_transaction()
# output once ml/predict.py is wired up, if you want to test that path too)
result = investigate_transaction(
    transaction=transaction,
    risk_score=87,
    risk_level="HIGH",
    risk_factors=["High transaction amount", "New device", "Odd hours"],
)

print(json.dumps(result, indent=2))
