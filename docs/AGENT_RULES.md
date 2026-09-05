# AGENT_RULES.md — RiskTrail Multi-Agent Collaboration Rules

RiskTrail is built by multiple agents working sequentially. These rules keep
their work compatible.

## Roles
- **Claude #1 (this agent):** Lead / Architect / Integrator. Owns
  architecture, docs, contracts, integration, review, and final testing.
- **Claude #2:** ML — dataset, model, evaluation, `predict_transaction()`.
- **Claude #3:** RAG + Investigation Agent — retrieval, evidence, LLM
  investigation/recommendation.
- **Claude #4:** Backend — FastAPI, wiring ML + RAG + LLM together.
- **Gemini + Antigravity:** Frontend — React dashboard.

## Ground Rules for Every Agent
1. Read `docs/PROJECT_CONTEXT.md` before writing code.
2. Follow `docs/API_CONTRACT.md` exactly — do not invent a different
   response shape.
3. Do not casually rename shared fields (see API_CONTRACT.md's Change
   Control section). If a change is genuinely needed, flag it to the Lead
   rather than shipping it silently.
4. Use existing files/modules instead of duplicating functionality.
5. Keep dependencies minimal and justified.
6. Keep code simple and explainable by a fresher during a demo — no
   over-engineering (see PROJECT_CONTEXT.md Coding Rules).
7. Never fabricate: model metrics, retrieval metrics, evidence, or Git
   history. Report real, measured results only.
8. Label all synthetic data clearly (see DATA_SCHEMA.md).
9. Stay within the safety boundaries in PROJECT_CONTEXT.md — defensive
   fraud investigation only.

## Review Checklist (used by Claude #1 after each agent's work)
1. Does it follow PROJECT_CONTEXT.md?
2. Does it follow the API contract?
3. Does it use existing files instead of duplicating functionality?
4. Are dependencies reasonable?
5. Is the code understandable?
6. Are errors handled?
7. Are metrics real?
8. Is synthetic data clearly labeled?
9. Does it integrate with existing components?
10. Can a student explain it during a hackathon demo?

## Sequential Workflow
1. Claude #1 — architecture, docs, repo foundation (this step)
2. Claude #2 — ML
3. Claude #1 — review + integrate ML
4. Claude #3 — RAG + investigation agent
5. Claude #1 — review + integrate RAG
6. Claude #4 — FastAPI backend
7. Claude #1 — review + integrate backend
8. Gemini + Antigravity — React frontend
9. Claude #1 — integrate frontend with backend
10. Claude #1 — end-to-end tests (high-risk, low-risk, borderline, invalid
    ID, insufficient evidence, backend/API failure, LLM failure)
11. Claude #1 — fix integration issues
12. Claude #1 — final README, docs, demo flow, metrics, cleanup

## Git
One repository. Meaningful, real commits only — no fabricated history.
Suggested commit sequence is in PROJECT_CONTEXT.md's spirit: setup → dataset
→ model → evaluation → investigation data → hybrid search → investigation
agent → backend → frontend → integration → fixes → final cleanup.
