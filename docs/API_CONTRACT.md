# API_CONTRACT.md — RiskTrail

This is the binding contract between frontend, backend, ML, and RAG. Field
names listed here are **shared contracts** — see the change-control rule at
the bottom before renaming or restructuring anything.

## `POST /api/investigate`

### Request
```json
{
  "transaction_id": "TXN_001"
}
```

| Field            | Type   | Required | Notes                        |
|-------------------|--------|----------|-------------------------------|
| transaction_id    | string | yes      | Must match a known transaction |

### Response — success (200)
```json
{
  "transaction_id": "TXN_001",
  "risk_score": 87,
  "risk_level": "HIGH",
  "risk_factors": [
    "High transaction amount",
    "New device"
  ],
  "evidence": [
    {
      "type": "case",
      "id": "CASE_014",
      "text": "Similar transaction..."
    },
    {
      "type": "policy",
      "id": "POLICY_001",
      "text": "High-value transactions..."
    }
  ],
  "investigation": "The transaction shows...",
  "recommendation": "Review and verify the transaction."
}
```

| Field           | Type            | Notes                                            |
|-----------------|-----------------|---------------------------------------------------|
| transaction_id  | string          | Echoes the request                                |
| risk_score      | number (0–100)  | From the ML model only                            |
| risk_level      | string          | e.g. `"LOW"`, `"MEDIUM"`, `"HIGH"`                |
| risk_factors    | array[string]   | From the ML model / feature analysis              |
| evidence        | array[object]   | Each item: `{ type, id, text }`. `type` is `"case"` or `"policy"` (extend only with agreement) |
| investigation   | string          | LLM-written narrative, grounded in risk_factors + evidence |
| recommendation  | string          | LLM-written next action for the investigator      |

### Response — error
```json
{
  "error": "Transaction not found",
  "transaction_id": "TXN_999"
}
```
Use standard HTTP status codes (404 for unknown transaction, 422 for a bad
request body, 500 for unexpected backend/LLM failure). Always include
`transaction_id` in error responses when it was provided, so the frontend
can show which lookup failed.

## Change Control
Do not casually rename or restructure: `transaction_id`, `risk_score`,
`risk_level`, `risk_factors`, `evidence`, `investigation`, `recommendation`.

If a change is genuinely needed:
1. State why it's necessary.
2. List every component that depends on the current shape (ML, RAG,
   backend, frontend).
3. Update this file, DATA_SCHEMA.md, and all dependent code together —
   never partially.
