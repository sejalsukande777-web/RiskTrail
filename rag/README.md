# RiskTrail — RAG + Investigation Agent (`rag/`)

This is Claude #3's component: given a transaction and its already-computed
ML output (`risk_score`, `risk_level`, `risk_factors`), retrieve supporting
evidence (historical cases + risk policies) and produce a grounded
investigation narrative + recommendation.

## Files in this component

- `data/cases.json` — 22 **synthetic** historical investigation case
  narratives (`synthetic: true` on every entry).
- `data/policies.json` — 12 **synthetic** fraud risk policies (`synthetic: true`
  on every entry).
- `data/merchant_history.json` — 10 **synthetic** merchant history records,
  keyed by `merchant_id` (`synthetic: true` on every entry).
- `rag/retrieval.py` — BM25 + embedding + hybrid retrieval,
  `search_similar_cases()`, `search_risk_policy()`.
- `rag/tools.py` — `get_transaction()`, `get_merchant_history()`.
- `rag/investigation_agent.py` — `investigate_transaction()`, the LLM call,
  evidence assembly, "don't invent evidence" system prompt.
- `rag/evaluate_retrieval.py` — 10 test queries with hand-labeled expected
  case/policy IDs, and a script that reports real hit-rate numbers.

**None of the data in `data/cases.json`, `data/policies.json`, or
`data/merchant_history.json` is real. It does not describe any real
customer, real transaction, or real company. Every entry is marked
`"synthetic": true` and was written for this hackathon demo.**

## Retrieval method: BM25 + embeddings, combined by weighted score

- **Keyword retrieval**: BM25 (`rank-bm25`), tokenized on lowercase
  words, indexed over each case/policy's title + summary + tags + full text.
- **Vector retrieval**: `sentence-transformers`, model
  `all-MiniLM-L6-v2` (small, fast, CPU-friendly, ~80MB) — embeds the same
  indexed text and the query, cosine similarity via normalized dot product.
- **Hybrid combination**: min-max normalize both score lists to 0–1 *per
  query*, then combine as `0.5 * bm25_norm + 0.5 * cosine_norm`
  (`BM25_WEIGHT = 0.5` in `retrieval.py`).

  **Why weighted combination instead of Reciprocal Rank Fusion (RRF):**
  both are simple, but this domain has a lot of exact-phrase signal
  ("new device", "odd hours", "chargeback") that keyword search is
  already very good at, and a directly-tunable weighted score is easier
  to explain live in a demo ("half keyword match, half semantic match")
  than RRF's rank-based formula — that's the whole reason, kept
  deliberately simple per the project's no-over-engineering rule.

If the embedding model can't be loaded (e.g. no network access to
download it), `retrieval.py` automatically falls back to BM25-only
rather than crashing, and logs why.

## Retrieval evaluation — real numbers (BM25-only and hybrid)

Evaluated on 10 hand-labeled test queries (`rag/evaluate_retrieval.py`),
each with a manually chosen "correct" expected case ID and policy ID.

### BM25-only

Case hit-rate@3: 10/10 = 100.0%
Policy hit-rate@2: 8/10 = 80.0%


### Hybrid (BM25 + all-MiniLM-L6-v2 embeddings)

Case hit-rate@3: 10/10 = 100.0%
Policy hit-rate@2: 9/10 = 90.0%


Hybrid retrieval improved policy retrieval by one query over BM25-only:
"failed login attempts followed by a large wire request" was missed
entirely by BM25 (it retrieved POLICY_006 and POLICY_003 instead of the
expected POLICY_007) but is correctly retrieved at rank 2 under hybrid.

The one remaining miss under both methods: the query "large transfer
from a new device in the middle of the night" expects `POLICY_002` (New
Device Verification) but both BM25 and hybrid retrieve `POLICY_003`
(Odd-Hours) and `POLICY_007` (Account Takeover) instead. This is a
reasonable near-miss rather than a failure — all three policies are
genuinely relevant to that transaction, retrieval just weighted the
night-time language more heavily than the "new device" phrasing. Case
retrieval was a clean 100% under both methods, with no misses at all.

Reproduce with:
```bash
python -m rag.evaluate_retrieval
```

(Needs one-time internet access to download `all-MiniLM-L6-v2` from
HuggingFace on first run; cached under `~/.cache/huggingface` after
that. If that download isn't reachable, the script prints BM25-only
results and an explicit note that hybrid could not be evaluated.)

## How "don't invent evidence" and "insufficient evidence" are enforced

Not just claimed in prose — actually enforced in `investigation_agent.py`:

1. **Evidence score floor**: retrieved cases/policies below
   `MIN_EVIDENCE_SCORE` (0.15, tunable) are filtered out entirely before
   ever reaching the LLM prompt (`_filter_by_score`). Weak/irrelevant
   matches never become "evidence."
2. **No evidence → no LLM call**: if filtering leaves zero evidence items,
   `investigate_transaction()` returns immediately with a plain, honest
   `investigation` string stating that no matching cases/policies were
   found, and a generic "escalate for manual review" recommendation. The
   LLM is never even called in this case — there is nothing to ground a
   generation in, so we don't ask it to generate one.
3. **System prompt constraints**: when there IS evidence, the system
   prompt given to the LLM (`SYSTEM_PROMPT` in `investigation_agent.py`)
   explicitly instructs it to (a) only reference case/policy IDs and facts
   present in the evidence block it was given, (b) never invent an ID or
   detail, (c) say so explicitly if the evidence is weak/irrelevant rather
   than write a confident narrative anyway, and (d) never restate or imply
   a different risk score/level than what was passed in.
4. **Evidence is passed as literal text, not summarized by the LLM
   first**: the exact retrieved `text` fields are inserted into the
   prompt verbatim (`_format_evidence_for_prompt`), so the LLM is reading
   the same evidence the backend/frontend will also display — no
   paraphrasing step where facts could drift.

## LLM integration notes

- Uses the Anthropic API directly via the `anthropic` Python package
  (already added to `requirements.txt`).
- Reads the API key from the `ANTHROPIC_API_KEY` environment variable
  (via `python-dotenv` / a `.env` file) — never hardcoded.
- Model used: `claude-sonnet-4-6`, one single system+user message call per
  investigation, `max_tokens=500`. No agent framework, no multi-step tool
  loop — this is intentionally a single prompt-and-response call per the
  project's scope rules.
- If `ANTHROPIC_API_KEY` isn't set, `investigate_transaction()` raises a
  clear `RuntimeError` telling you to set it — it does not fail silently
  or fall back to fabricated text.

## What backend (Claude #4) needs to know

```python
from rag.investigation_agent import investigate_transaction

result = investigate_transaction(
    transaction=transaction_dict,   # the same dict shape used for predict_transaction()
    risk_score=risk_score,          # int, from ML — do not compute this yourself
    risk_level=risk_level,          # str, from ML
    risk_factors=risk_factors,      # list[str], from ML
)
# result = {
#     "evidence": [{"type": "case"|"policy", "id": str, "text": str}, ...],
#     "investigation": str,
#     "recommendation": str,
# }
```

- Call `predict_transaction()` (ML) **first**, then pass its three outputs
  straight into `investigate_transaction()` — this function does not call
  ML itself and does not re-derive or sanity-check the risk score.
- `evidence` can be an **empty list** — this is a valid, expected response
  when nothing relevant was retrieved, not a bug. Render it as "no
  supporting evidence found" in the UI rather than assuming it's always
  non-empty.
- `evidence` item shape is fixed: exactly `type`, `id`, `text` — no
  `score` field is included (it's used internally for filtering, then
  stripped before returning) — matches `docs/DATA_SCHEMA.md` /
  `docs/API_CONTRACT.md` exactly.
- `get_merchant_history(merchant_id)` returns `None` for unknown
  merchants — this is normal and does not indicate an error; merchant
  context is optional, not required for every transaction.
  `data/merchant_history.json` includes entries for all 6 merchant IDs
  used across `data/sample_transactions.json` (`M_amazon`, `M_uber`,
  `M_starbucks`, `M_bestbuy`), plus a broader set of synthetic merchants
  (`M001`-`M009`, `M999`) for general testing.
- Requires `ANTHROPIC_API_KEY` set in the environment/`.env` — make sure
  this is set wherever the backend runs, not just locally.
- `rag/retrieval.py` builds its BM25 + embedding indexes once at import
  time (a few hundred milliseconds for BM25; embeddings load lazily on
  first search call). Importing `rag.investigation_agent` at backend
  startup (rather than per-request) avoids paying that cost on every API
  call.