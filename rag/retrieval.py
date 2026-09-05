"""
rag/retrieval.py

Hybrid retrieval over cases.json + policies.json using:
  - BM25 (keyword search, via rank-bm25)
  - Embeddings (semantic search, via sentence-transformers)
  - A simple weighted-score hybrid combination of the two

Why weighted score combination instead of Reciprocal Rank Fusion (RRF):
both are simple, but weighted combination lets us directly tune how much
we trust keyword vs. semantic matching for this domain (fraud/policy text
has a lot of exact-match phrases like "new device" or "odd hours" that
keyword search is very good at), and the resulting combined score is more
interpretable in a demo ("60% keyword score + 40% semantic score") than
RRF's rank-based score.

This file loads data/cases.json and data/policies.json once at import
time and builds both indexes. No classes -- just plain functions, per
project conventions.
"""

import json
import os
import re

from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_THIS_DIR), "data")
CASES_PATH = os.path.join(_DATA_DIR, "cases.json")
POLICIES_PATH = os.path.join(_DATA_DIR, "policies.json")

# small + fast model, good default for a hackathon demo (~80MB, CPU is fine)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# hybrid weighting: score = BM25_WEIGHT * norm(bm25) + (1 - BM25_WEIGHT) * norm(cosine)
BM25_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Loading + indexing (runs once at import)
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text):
    # simple lowercase word tokenizer, good enough for BM25 on this dataset
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_corpus(records, kind):
    """Turn case/policy records into (id, kind, display_text, index_text)."""
    corpus = []
    for r in records:
        if kind == "case":
            display_text = r["text"]
        else:  # policy
            display_text = r["text"]
        # index over title/summary + full text so short keyword queries
        # (e.g. "new device") still match well
        extra = r.get("title", "") + " " + r.get("summary", "") + " " + " ".join(r.get("tags", []))
        index_text = (extra + " " + display_text).strip()
        corpus.append({
            "id": r["id"],
            "type": kind,
            "text": display_text,
            "index_text": index_text,
        })
    return corpus


_cases_raw = _load_json(CASES_PATH)
_policies_raw = _load_json(POLICIES_PATH)

_case_corpus = _build_corpus(_cases_raw, "case")
_policy_corpus = _build_corpus(_policies_raw, "policy")

_case_tokenized = [_tokenize(item["index_text"]) for item in _case_corpus]
_policy_tokenized = [_tokenize(item["index_text"]) for item in _policy_corpus]

_case_bm25 = BM25Okapi(_case_tokenized)
_policy_bm25 = BM25Okapi(_policy_tokenized)

# Embeddings are loaded lazily (only when first search happens) so that
# importing this module doesn't require a network call / model download
# just to e.g. run get_transaction() elsewhere.
_embedding_model = None
_case_embeddings = None
_policy_embeddings = None
_embedding_load_error = None  # set the first time model loading fails, so
                               # callers (e.g. evaluate_retrieval.py) can
                               # detect a silent BM25 fallback and report it
                               # honestly instead of mislabeling it "hybrid"


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if not _EMBEDDINGS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers"
            )
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embeddings_actually_available():
    """Try (once) to load the embedding model. Returns True if it works,
    False if it fails for any reason (e.g. no network access to download
    it). Use this before claiming a "hybrid" result set is real."""
    global _embedding_load_error
    if _embedding_model is not None:
        return True
    try:
        _get_embedding_model()
        return True
    except Exception as e:
        _embedding_load_error = str(e)
        return False


def _get_case_embeddings():
    global _case_embeddings
    if _case_embeddings is None:
        model = _get_embedding_model()
        texts = [item["index_text"] for item in _case_corpus]
        _case_embeddings = model.encode(texts, normalize_embeddings=True)
    return _case_embeddings


def _get_policy_embeddings():
    global _policy_embeddings
    if _policy_embeddings is None:
        model = _get_embedding_model()
        texts = [item["index_text"] for item in _policy_corpus]
        _policy_embeddings = model.encode(texts, normalize_embeddings=True)
    return _policy_embeddings


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(scores):
    scores = list(scores)
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _bm25_scores(query, bm25_index):
    tokens = _tokenize(query)
    return list(bm25_index.get_scores(tokens))


def _cosine_scores(query, doc_embeddings):
    model = _get_embedding_model()
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    # embeddings are normalized, so dot product == cosine similarity
    return list(np.dot(doc_embeddings, query_emb))


def _hybrid_search(query, corpus, bm25_index, get_doc_embeddings, top_k, use_embeddings=True):
    bm25_raw = _bm25_scores(query, bm25_index)
    bm25_norm = _minmax_normalize(bm25_raw)

    if use_embeddings and _EMBEDDINGS_AVAILABLE:
        try:
            doc_embeddings = get_doc_embeddings()
            cosine_raw = _cosine_scores(query, doc_embeddings)
            cosine_norm = _minmax_normalize(cosine_raw)
        except Exception as e:
            # embedding model unavailable (e.g. no network access to download
            # it) -- fall back to BM25-only rather than crashing the demo
            global _embedding_load_error
            _embedding_load_error = str(e)
            cosine_norm = [0.0] * len(corpus)
            use_embeddings = False
    else:
        cosine_norm = [0.0] * len(corpus)
        use_embeddings = False

    weight = BM25_WEIGHT if use_embeddings else 1.0
    combined = [
        weight * b + (1 - weight) * c
        for b, c in zip(bm25_norm, cosine_norm)
    ]

    ranked_idx = sorted(range(len(corpus)), key=lambda i: combined[i], reverse=True)[:top_k]

    results = []
    for i in ranked_idx:
        item = corpus[i]
        results.append({
            "type": item["type"],
            "id": item["id"],
            "text": item["text"],
            "score": round(combined[i], 4),
        })
    return results


# ---------------------------------------------------------------------------
# Public API (used by rag/tools.py and rag/investigation_agent.py)
# ---------------------------------------------------------------------------

def search_similar_cases(query, top_k=3, use_embeddings=True):
    """Search historical cases with hybrid (BM25 + embeddings) retrieval."""
    return _hybrid_search(
        query, _case_corpus, _case_bm25, _get_case_embeddings, top_k, use_embeddings
    )


def search_risk_policy(query, top_k=2, use_embeddings=True):
    """Search risk policies with hybrid (BM25 + embeddings) retrieval."""
    return _hybrid_search(
        query, _policy_corpus, _policy_bm25, _get_policy_embeddings, top_k, use_embeddings
    )


if __name__ == "__main__":
    # quick manual smoke test
    print("Cases indexed:", len(_case_corpus))
    print("Policies indexed:", len(_policy_corpus))
    demo_query = "large transaction from a new device late at night"
    print("\nQuery:", demo_query)
    print("\nTop cases (BM25-only, since embeddings need network access):")
    for r in search_similar_cases(demo_query, top_k=3, use_embeddings=False):
        print(" ", r["id"], r["score"], "-", r["text"][:80], "...")
    print("\nTop policies (BM25-only):")
    for r in search_risk_policy(demo_query, top_k=2, use_embeddings=False):
        print(" ", r["id"], r["score"], "-", r["text"][:80], "...")