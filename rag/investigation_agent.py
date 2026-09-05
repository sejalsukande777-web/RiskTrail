"""
rag/investigation_agent.py

The main exposed interface for the RAG + Investigation Agent component:

    investigate_transaction(transaction, risk_score, risk_level, risk_factors)

This is the shared contract with the backend (Claude #4) -- see
docs/API_CONTRACT.md. Do not rename the returned keys or change the
evidence item shape.

Flow:
  1. Build a search query from the ML risk_factors + transaction fields.
  2. Retrieve similar historical cases + relevant policies (hybrid search).
  3. Optionally look up merchant history for extra context.
  4. Assemble ONLY the retrieved evidence into a prompt for the LLM.
  5. Ask the LLM to write an investigation narrative + recommendation,
     grounded strictly in that evidence -- it is explicitly instructed not
     to invent case numbers, policy names, or details that were not
     retrieved.
  6. If nothing relevant was retrieved, skip the LLM call for the "no
     evidence" framing and say so plainly instead of forcing a confident
     narrative from weak matches.
"""

import os
import re

from dotenv import load_dotenv

from rag.retrieval import search_similar_cases, search_risk_policy
from rag.tools import get_merchant_history

load_dotenv()

# LLM: Groq's free-tier API (OpenAI-compatible endpoint), not Anthropic --
# chosen so the project has no per-call cost during development/demo.
# Expected env var: GROQ_API_KEY (get one free, no card required, at
# console.groq.com/keys)
LLM_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Below this hybrid score, a retrieved case/policy is considered too weak
# to count as real evidence (tune this using rag/evaluate_retrieval.py
# results once embeddings are available; BM25-only scores are already
# min-max normalized to 0-1 per query, so 0.15 is a conservative floor).
MIN_EVIDENCE_SCORE = 0.15


def _build_search_query(transaction, risk_factors):
    """Turn the ML risk_factors + a few transaction fields into one query
    string for both the case search and the policy search."""
    parts = list(risk_factors) if risk_factors else []
    if transaction.get("amount") is not None:
        parts.append(f"amount {transaction['amount']}")
    if transaction.get("device_id"):
        parts.append(f"device {transaction['device_id']}")
    return " ".join(str(p) for p in parts) if parts else "unusual transaction risk review"


def _filter_by_score(results, min_score=MIN_EVIDENCE_SCORE):
    return [r for r in results if r.get("score", 0) >= min_score]


def _format_evidence_for_prompt(evidence):
    lines = []
    for e in evidence:
        lines.append(f"[{e['type'].upper()} {e['id']}] {e['text']}")
    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are a fraud investigation assistant. You write a short \
investigation narrative and a recommended next action for a flagged transaction.

You are given:
- The transaction's risk score, risk level, and risk factors (already computed \
by a separate machine learning model -- you must NOT change, second-guess, or \
recompute these; treat them as ground truth input).
- A list of retrieved evidence: historical case narratives and/or risk policy \
excerpts, each with an ID.
- Optionally, merchant history context.

STRICT RULES -- follow these exactly:
1. Only reference case IDs, policy IDs, and facts that appear in the evidence \
you were given below. Never invent a case number, policy name, or detail that \
is not present in the retrieved text.
2. If the evidence list is empty, or none of it is clearly relevant to this \
transaction's risk factors, say so explicitly in your investigation text \
(e.g. "No closely matching historical cases or policies were retrieved for \
this transaction's risk factors.") instead of writing a confident-sounding \
narrative anyway.
3. Do not state or imply a risk score, risk level, or risk factor different \
from the ones you were given.
4. Keep the investigation narrative to 3-5 sentences and the recommendation \
to 1-2 sentences. Write for a human fraud investigator reading this in a \
dashboard, not for the customer.

Respond in exactly this format, with no other text before or after:
INVESTIGATION: <your narrative>
RECOMMENDATION: <your recommendation>
"""


def _call_llm(prompt_text):
    """Single direct call to Groq's free-tier API (OpenAI-compatible client
    pointed at Groq's endpoint -- not calling OpenAI itself). Raises
    RuntimeError with a clear message if GROQ_API_KEY is not set, rather
    than failing with an unclear low-level error."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. Get a free key "
            "(no credit card required) at console.groq.com/keys and add it "
            "to your .env file (see rag/README.md)."
        )

    from openai import OpenAI  # imported here so the rest of this module still
                                # works (e.g. for tests) even if the openai
                                # package isn't installed yet

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        # NOTE: openai/gpt-oss-20b is a reasoning model. By default its
        # internal "thinking" tokens get mixed into the same output as the
        # final answer and can consume most of max_tokens before the actual
        # INVESTIGATION/RECOMMENDATION text is written, cutting it off.
        # reasoning_effort="low" minimizes that internal reasoning, and
        # reasoning_format="hidden" excludes it from the returned content
        # entirely so we only get the final answer. Both are Groq-specific
        # params not in the standard OpenAI client, so they go in extra_body.
        max_tokens=1024,
        extra_body={"reasoning_effort": "low", "reasoning_format": "hidden"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    )
    return response.choices[0].message.content


def _parse_llm_response(raw_text):
    """Pull INVESTIGATION and RECOMMENDATION out of the LLM's formatted reply.
    Falls back gracefully if the model doesn't follow the format exactly."""
    inv_match = re.search(r"INVESTIGATION:\s*(.*?)(?:\nRECOMMENDATION:|\Z)", raw_text, re.DOTALL)
    rec_match = re.search(r"RECOMMENDATION:\s*(.*)", raw_text, re.DOTALL)

    investigation = inv_match.group(1).strip() if inv_match else raw_text.strip()
    recommendation = rec_match.group(1).strip() if rec_match else (
        "Recommendation unavailable -- the model's response appears to have "
        "been cut off before reaching the recommendation. Review the "
        "investigation notes above manually, and consider raising max_tokens "
        "in rag/investigation_agent.py if this happens often."
    )
    return investigation, recommendation


def investigate_transaction(
    transaction: dict,
    risk_score: int,
    risk_level: str,
    risk_factors: list,
) -> dict:
    """
    Shared contract with the backend. See docs/API_CONTRACT.md.

    Returns:
    {
        "evidence": [{"type": "case"|"policy", "id": str, "text": str}, ...],
        "investigation": str,
        "recommendation": str,
    }
    """
    query = _build_search_query(transaction, risk_factors)

    case_results = search_similar_cases(query, top_k=3)
    policy_results = search_risk_policy(query, top_k=2)

    case_evidence = _filter_by_score(case_results)
    policy_evidence = _filter_by_score(policy_results)

    # evidence item shape is fixed by the contract: type, id, text only
    # (drop the internal "score" field before returning to the backend)
    evidence = [
        {"type": e["type"], "id": e["id"], "text": e["text"]}
        for e in (case_evidence + policy_evidence)
    ]

    merchant_id = transaction.get("merchant_id")
    merchant_context = get_merchant_history(merchant_id) if merchant_id else None

    if not evidence:
        # No relevant evidence retrieved -- per the hard rule, do not force
        # a confident narrative. Skip the LLM call entirely for this case
        # (nothing to ground it in) and return a plain, honest response.
        return {
            "evidence": [],
            "investigation": (
                f"No closely matching historical cases or policies were retrieved "
                f"for this transaction's risk factors ({', '.join(risk_factors) if risk_factors else 'none provided'}). "
                f"This transaction is rated {risk_level} ({risk_score}/100) by the fraud model, "
                f"but there is insufficient supporting evidence in the case/policy knowledge base "
                f"to produce a grounded investigation narrative."
            ),
            "recommendation": "Escalate for manual investigator review; no automated evidence match found.",
        }

    prompt_parts = [
        f"Risk score: {risk_score}/100",
        f"Risk level: {risk_level}",
        f"Risk factors: {', '.join(risk_factors) if risk_factors else 'none provided'}",
        "",
        "Retrieved evidence:",
        _format_evidence_for_prompt(evidence),
    ]
    if merchant_context:
        prompt_parts += ["", f"Merchant history context: {merchant_context}"]

    prompt_text = "\n".join(prompt_parts)

    raw_response = _call_llm(prompt_text)
    investigation, recommendation = _parse_llm_response(raw_response)

    return {
        "evidence": evidence,
        "investigation": investigation,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    # manual smoke test using a fabricated transaction (no LLM call --
    # just verifies evidence assembly works end-to-end without a key)
    demo_transaction = {
        "amount": 4200.0,
        "timestamp": "2026-09-05T02:47:00",
        "device_id": "device_unknown_77",
        "merchant_id": "M002",
    }
    demo_risk_factors = ["High transaction amount", "New device", "Odd hours"]

    query = _build_search_query(demo_transaction, demo_risk_factors)
    print("Search query:", query)
    print("\nCase evidence (BM25 hybrid, embeddings may be unavailable offline):")
    for e in _filter_by_score(search_similar_cases(query, top_k=3)):
        print(" ", e["id"], e["score"])
    print("\nPolicy evidence:")
    for e in _filter_by_score(search_risk_policy(query, top_k=2)):
        print(" ", e["id"], e["score"])
    print("\n(Skipping actual LLM call in this smoke test -- requires GROQ_API_KEY)")