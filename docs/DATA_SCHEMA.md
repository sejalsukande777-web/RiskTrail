# DATA_SCHEMA.md — RiskTrail

All data under `data/` must clearly label anything synthetic. Never present
synthetic cases as real customer transactions. Never fabricate metrics.

## Transaction
Source: public fraud dataset (ML training/lookup) where possible.

| Field           | Type   | Notes                          |
|-----------------|--------|---------------------------------|
| transaction_id  | string | Unique ID, e.g. `TXN_001`       |
| amount          | number |                                 |
| merchant_id     | string |                                 |
| device_id       | string |                                 |
| timestamp       | string | ISO 8601                        |
| ...             |        | additional dataset-specific features |

## ML Output (produced at request time, not stored)
| Field         | Type          |
|---------------|---------------|
| risk_score    | number (0–100) |
| risk_level    | string        |
| risk_factors  | array[string] |

## Investigation Case (synthetic, clearly labeled)
```json
{
  "id": "CASE_014",
  "synthetic": true,
  "summary": "...",
  "text": "Full case narrative used for retrieval...",
  "tags": ["high-amount", "new-device"]
}
```

## Risk Policy (can be synthetic or adapted from public guidance, labeled)
```json
{
  "id": "POLICY_001",
  "synthetic": true,
  "title": "High-Value Transaction Review",
  "text": "Full policy text used for retrieval..."
}
```

## Merchant History (synthetic, clearly labeled)
```json
{
  "merchant_id": "MERCH_042",
  "synthetic": true,
  "transaction_count": 120,
  "flagged_count": 4,
  "notes": "..."
}
```

## Evidence Item (as returned by `/api/investigate`)
```json
{
  "type": "case | policy",
  "id": "string",
  "text": "string"
}
```

## Labeling Rule
Every synthetic record in `data/` must include `"synthetic": true` (or
equivalent clear labeling in the loading code / README) so nobody mistakes
it for real customer or case data.
