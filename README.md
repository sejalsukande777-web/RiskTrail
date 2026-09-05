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
Foundation stage: architecture, documentation, folder structure, and API
contract are established. ML, RAG, backend, and frontend implementation are
tracked in `docs/AGENT_RULES.md`'s sequential workflow.

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
# backend + ML + RAG (Python)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# frontend
cd frontend
npm install
npm run dev
```

## Safety
RiskTrail is strictly defensive: detect, investigate, retrieve evidence,
explain, and recommend action to a human investigator. See
`docs/PROJECT_CONTEXT.md` for full safety rules.
