"""
RiskTrail Backend API

Wires together:
- ml.predict.predict_transaction()          -> live risk scoring
- rag.tools.get_transaction()               -> transaction lookup
- rag.investigation_agent.investigate_transaction() -> live RAG + LLM investigation

Exposes:
- GET  /            -> health check
- GET  /health      -> health check
- POST /api/investigate -> main investigation endpoint
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ml.predict import predict_transaction
from rag.tools import get_transaction
from rag.investigation_agent import investigate_transaction

app = FastAPI(title="RiskTrail Backend")

# Allow the React (Vite) frontend running on a different port to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvestigateRequest(BaseModel):
    transaction_id: str


def error_body(message: str, transaction_id: str) -> dict:
    """Standard error response shape used across this API."""
    return {"error": message, "transaction_id": transaction_id}


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/investigate")
def investigate(request: InvestigateRequest):
    transaction_id = request.transaction_id

    # Step 1: look up the transaction's INPUT fields (never model_output)
    # get_transaction() now returns None for unknown IDs (fixed in rag/tools.py).
    try:
        transaction = get_transaction(transaction_id)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(f"Transaction lookup failed: {exc}", transaction_id),
        )

    if transaction is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_body("Transaction not found", transaction_id),
        )

    # Step 2: run the ML model live to get real, current risk numbers
    try:
        ml_result = predict_transaction(transaction)
        risk_score = ml_result["risk_score"]
        risk_level = ml_result["risk_level"]
        risk_factors = ml_result["risk_factors"]
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(f"ML prediction failed: {exc}", transaction_id),
        )

    # Step 3: run the RAG + LLM investigation live, using the fresh ML output
    try:
        rag_result = investigate_transaction(
            transaction=transaction,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(f"Investigation agent failed: {exc}", transaction_id),
        )

    # Step 4: merge into the exact response shape from docs/API_CONTRACT.md
    response = {
        "transaction_id": transaction_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "evidence": rag_result["evidence"],
        "investigation": rag_result["investigation"],
        "recommendation": rag_result["recommendation"],
    }
    return response