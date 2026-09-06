# RiskTrail — AI Fraud Investigation Copilot

RiskTrail doesn't just tell an investigator that a transaction is risky. It
traces the signals and evidence behind that risk and explains what to do
next.

## What it does
1. An ML model scores a transaction for fraud risk.
2. An investigation agent searches historical cases and risk policies
   using hybrid (keyword + vector) retrieval.
3. An LLM combines the risk factors and retrieved evidence into a plain-
   language investigation and a concrete recommendation.
4. A React dashboard presents it all to a human investigator.

## Project Status
**Complete and tested end-to-end.** ML, RAG + investigation agent, backend,
and frontend are all built and wired together. Real, independently
verified results:

- **ML**: Logistic Regression on the Kaggle Credit Card Fraud dataset —
  Precision 0.0609, Recall 0.9184, F1 0.1141, ROC-AUC 0.9722 on a held-out
  test set (56,962 transactions, 98 fraud). See `ml/README.md` for the full
  discussion of the precision/recall trade-off and false positives.
- **RAG retrieval**: hybrid (BM25 + embeddings) hit-rate@3 for cases:
  100%; hit-rate@2 for policies: 90% (vs. 80% for BM25 alone) across 10
  test queries. See `rag/README.md`.
- **End-to-end**: verified working for high-risk, low-risk, and
  borderline/MEDIUM transactions; unknown transaction IDs (404); and the
  RAG agent's "evidence not closely relevant" honest-hedging behavior —
  all tested through the actual running frontend, not just the API
  directly.

Run it yourself: see **Getting Started** below.

## Repository Structure
```
RiskTrail/
├── frontend/     # React + Vite dashboard
├── backend/      # FastAPI app, POST /api/investigate
├── ml/           # Fraud model: train, predict, evaluate
├── rag/          # Hybrid RAG + investigation agent
├── data/         # Dataset + synthetic supporting data (clearly labeled)
├── docs/         # Architecture, contracts, schema, UI spec, agent rules
├── README.md
├── requirements.txt
└── .gitignore
```

## Docs — read these first
- [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) — single source of truth
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md)
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md)
- [`docs/UI_SPEC.md`](docs/UI_SPEC.md)
- [`docs/AGENT_RULES.md`](docs/AGENT_RULES.md)

## Tech Stack
- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, FastAPI
- **ML:** pandas, NumPy, scikit-learn, joblib
- **RAG:** BM25 + vector embeddings, hybrid ranking
- **LLM:** simple API integration (no heavy agent framework)

## Getting Started

```bash
# 1. Install Python dependencies (repo root)
pip install -r requirements.txt

# 2. Download the ML dataset (not included in the repo — see ml/README.md)
#    Place it at: data/creditcard.csv

# 3. Train the fraud model (one-time; ml/model.joblib is already committed,
#    so this step is optional unless you want to retrain)
python ml/train.py

# 4. Get a free Groq API key (no credit card required):
#    https://console.groq.com/keys
#    Create a .env file in the repo root with:
#    GROQ_API_KEY=your_key_here

# 5. Run the backend (repo root, separate terminal)
uvicorn backend.main:app --reload --port 8000

# 6. Run the frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** and try transaction IDs `TXN_001` through
`TXN_006` (or any unknown ID, e.g. `TXN_999`, to see the error handling).

## Safety
RiskTrail is strictly defensive: detect, investigate, retrieve evidence,
explain, and recommend action to a human investigator. See
`docs/PROJECT_CONTEXT.md` for full safety rules.