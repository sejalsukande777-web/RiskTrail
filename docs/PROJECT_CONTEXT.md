# PROJECT_CONTEXT.md — RiskTrail (Single Source of Truth)

## Project Name
RiskTrail — AI Fraud Investigation Copilot

## Purpose
RiskTrail is a student hackathon project that goes beyond flagging suspicious
transactions. It investigates **why** a transaction is risky by combining an
ML fraud detector with a Hybrid RAG evidence system and an LLM investigation
agent, then presents the findings to a human investigator in a React
dashboard.

## Problem Statement
Fraud-detection models can say a transaction is risky, but they don't explain
*why*, don't point to similar past cases or policies, and don't tell an
investigator what to do next. RiskTrail closes that gap.

## Core Architecture
```
Transaction
    ↓
ML Fraud Detection            → risk_score, risk_level, risk_factors
    ↓
Investigation Agent
    ↓
Hybrid RAG (keyword + vector) → supporting evidence (cases, policies)
    ↓
LLM Investigation             → investigation narrative + recommendation
    ↓
React Dashboard
```

Key distinction:
- **ML answers:** "Is this transaction suspicious?"
- **RiskTrail answers:** "Why is it suspicious, what evidence supports that,
  and what should the investigator do next?"

This is not a generic chatbot and not a generic fraud-prediction demo.

## Technology Stack
- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, FastAPI
- **ML:** Python, pandas, NumPy, scikit-learn, joblib
- **RAG:** BM25 (or similar keyword retrieval) + vector embeddings, combined
  via a simple hybrid ranking step
- **LLM:** any suitable LLM API via a simple integration — no heavyweight
  agent framework

## Repository Structure
```
RiskTrail/
├── frontend/     # React + Vite dashboard
├── backend/      # FastAPI app, POST /api/investigate
├── ml/           # Fraud model: train, predict, evaluate
├── rag/          # Hybrid RAG + investigation agent + LLM calls
├── data/         # Dataset + synthetic supporting data (clearly labeled)
├── docs/         # This file + ARCHITECTURE, API_CONTRACT, DATA_SCHEMA,
│                 # UI_SPEC, AGENT_RULES
├── README.md
├── requirements.txt
└── .gitignore
```

## Component Responsibilities

**ML (`ml/`)** — Claude #2
Dataset, preprocessing, train/test split, model training, prediction,
risk score, risk level, risk factors, model save/load, evaluation
(precision, recall, F1, ROC-AUC, confusion matrix), honest discussion of
false positives. Exposes `predict_transaction(transaction)`.

**RAG + Investigation Agent (`rag/`)** — Claude #3
Historical cases, risk policies, keyword + vector + hybrid retrieval,
retrieval evaluation, investigation agent that combines ML risk factors +
retrieved evidence into an LLM-written investigation and recommendation.
Must never invent evidence; must say so if evidence is insufficient.

**Backend (`backend/`)** — Claude #4
FastAPI app exposing `POST /api/investigate`, wiring together ML → RAG →
investigation agent → LLM, returning the shared response shape, with error
handling. No unnecessary microservices.

**Frontend (`frontend/`)** — Gemini + Antigravity
Transaction ID input, Investigate button, loading state, risk score/level/
factors, evidence cards, AI investigation text, recommendation, error
handling. Must follow the existing API contract exactly — no inventing a
different response shape.

## API Contract (summary — full detail in API_CONTRACT.md)
`POST /api/investigate` — request `{ "transaction_id": "TXN_001" }` — returns
a JSON object with the fixed field names: `transaction_id`, `risk_score`,
`risk_level`, `risk_factors`, `evidence`, `investigation`, `recommendation`.
These names are shared contracts — see AGENT_RULES.md before changing any of
them.

## Data Schema
See DATA_SCHEMA.md for transaction, evidence, case, and policy shapes.

## Shared Field Names (do not rename casually)
`transaction_id`, `risk_score`, `risk_level`, `risk_factors`, `evidence`,
`investigation`, `recommendation`

## Coding Rules
This is a fresher/student hackathon project. Code must be simple,
understandable, and demo-able:
- Simple functions, clear variable names, small files, minimal useful comments
- No over-engineering: no unnecessary classes, design patterns, utility
  layers, microservices, heavy configuration, or agent frameworks
- No unnecessary dependencies

## Agent Collaboration Rules
See AGENT_RULES.md.

## Safety Rules
RiskTrail is strictly defensive: detect, investigate, retrieve evidence,
explain, and recommend action to a human investigator. It must never add
functionality for committing fraud, evading detection, payment abuse,
credential theft, offensive cybersecurity, or attack automation. The LLM
never replaces the ML detector, never invents a risk score, and never
invents evidence.

## Definition of Done
- ✅ ML model trained and evaluated with real (not fabricated) metrics —
  see `ml/README.md`
- ✅ RAG retrieves real evidence from real (synthetic, clearly labeled)
  data — no invented evidence; retrieval evaluation confirmed 100%/90%
  hit-rates — see `rag/README.md`
- ✅ Backend correctly wires ML → RAG → investigation agent → LLM and
  returns the shared response shape — see `backend/README.md`
- ✅ Frontend renders the shared response shape correctly, with loading
  and error states — see `frontend/README.md`
- ✅ End-to-end flow tested: high-risk, low-risk, borderline/MEDIUM,
  invalid ID, and insufficient/weakly-relevant evidence — all verified
  through the actual running frontend. (LLM-provider-failure was
  intentionally left as a manual pre-demo check rather than a committed
  test, since it requires temporarily breaking a working API key.)
- ✅ README + docs allow a student to explain the whole system in a demo