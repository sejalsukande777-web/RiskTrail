# ARCHITECTURE.md — RiskTrail

## System Diagram
```
┌────────────┐     POST /api/investigate      ┌──────────────┐
│  React     │ ──────────────────────────────▶│   FastAPI     │
│  Dashboard │◀──────────────────────────────  │   Backend     │
└────────────┘         JSON response           └──────┬───────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────┐
                        ▼                               ▼                           ▼
                ┌───────────────┐             ┌──────────────────┐         ┌───────────────┐
                │  ML Detector   │             │ Investigation     │         │  Data Store    │
                │  (ml/)         │             │ Agent (rag/)       │         │  (data/)       │
                │  risk_score,   │──factors───▶│  → search cases    │◀───────▶│  transactions, │
                │  risk_level,   │             │  → search policies │         │  cases,        │
                │  risk_factors  │             │  → hybrid RAG      │         │  policies,     │
                └───────────────┘             │  → LLM call         │         │  merchants      │
                                                │  → investigation,   │         └───────────────┘
                                                │    recommendation   │
                                                └──────────────────┘
```

## Component Roles (fixed distinction — do not blur these)
- **ML = Detector.** Produces `risk_score`, `risk_level`, `risk_factors`.
  Nothing else touches these numbers.
- **RAG = Evidence.** Retrieves real supporting material (cases, policies)
  via keyword + vector + hybrid search. Never fabricates results.
- **Investigation Agent = Connector.** Orchestrates: pulls the transaction,
  takes the ML risk factors, calls RAG, assembles everything for the LLM.
- **LLM = Explanation + Recommendation.** Writes `investigation` and
  `recommendation` text grounded only in the risk factors and evidence it
  was given. Cannot invent a score, cannot invent evidence, cannot override
  the ML detector.
- **React = Investigator interface.** Renders the shared response shape;
  does not compute or transform risk data itself.

## Data Flow (per request)
1. Frontend sends `{ transaction_id }` to `POST /api/investigate`.
2. Backend looks up the transaction.
3. Backend calls the ML detector → `risk_score`, `risk_level`,
   `risk_factors`.
4. Backend calls the investigation agent, which:
   a. Searches similar cases and risk policies (hybrid RAG).
   b. Optionally checks merchant history.
   c. Assembles risk factors + evidence into a prompt.
   d. Calls the LLM for `investigation` + `recommendation` text.
5. Backend returns the shared JSON response.
6. Frontend renders risk score/level/factors, evidence cards, investigation
   text, and recommendation.

## Suggested Tool Functions (rag/ + backend/)
- `get_transaction(transaction_id)`
- `get_merchant_history(merchant_id)`
- `search_similar_cases(query)`
- `search_risk_policy(query)`
- `calculate_risk_factors(transaction, model_output)`

## Failure Handling
- Invalid `transaction_id` → clear error response, not a crash.
- Insufficient evidence → investigation agent says so explicitly rather than
  inventing supporting material.
- LLM call failure → backend returns a graceful error; frontend shows an
  error state (see UI_SPEC.md).

## Non-Goals
No microservices, no message queues, no complex agent framework, no
authentication system, no multi-tenant setup. This is a single FastAPI
service calling into `ml/` and `rag/` as plain Python modules.
