"""
rag/evaluate_retrieval.py

Retrieval evaluation for search_similar_cases() and search_risk_policy().

For each test query we know (by having written the synthetic data
ourselves) which case ID and/or policy ID is the "correct" expected match.
We report, per query and in aggregate:

  - Hit@k        : was the expected ID anywhere in the top-k results?
  - Precision@k  : (kept for cases with more than one relevant ID; here
                    each query has exactly one expected ID per type, so
                    precision@k == hit@k / k when hit, this is reported
                    anyway for clarity/completeness)

Run with: python -m rag.evaluate_retrieval

By default this runs BM25-only (use_embeddings=False) if the embedding
model can't be loaded (e.g. no network access to download it from
HuggingFace) -- see the printed warning. To get real hybrid (BM25 +
embeddings) numbers, run this on a machine with normal internet access;
the sentence-transformers model (all-MiniLM-L6-v2, ~80MB) downloads
automatically on first run and is cached after that.
"""

from rag.retrieval import (
    search_similar_cases,
    search_risk_policy,
    _EMBEDDINGS_AVAILABLE,
    embeddings_actually_available,
    _embedding_load_error,
)
import rag.retrieval as _retrieval_module

# ---------------------------------------------------------------------------
# Test queries with expected IDs (written by hand against data/cases.json
# and data/policies.json -- these are our own ground truth, not fabricated
# result numbers)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "large transfer from a new device in the middle of the night",
        "expected_case_id": "CASE_001",
        "expected_policy_id": "POLICY_002",
    },
    {
        "query": "many small transactions across different merchants in a few minutes",
        "expected_case_id": "CASE_003",
        "expected_policy_id": "POLICY_006",
    },
    {
        "query": "transaction location far from the customer's usual country, card never left their possession",
        "expected_case_id": "CASE_006",
        "expected_policy_id": "POLICY_005",
    },
    {
        "query": "failed login attempts followed by a large wire request",
        "expected_case_id": "CASE_007",
        "expected_policy_id": "POLICY_007",
    },
    {
        "query": "purchase at a merchant that has never been seen before in the system",
        "expected_case_id": "CASE_008",
        "expected_policy_id": "POLICY_009",
    },
    {
        "query": "new shipping address that does not match the billing address for an expensive item",
        "expected_case_id": "CASE_010",
        "expected_policy_id": "POLICY_012",
    },
    {
        "query": "several transactions each just under a reporting threshold in one day",
        "expected_case_id": "CASE_021",
        "expected_policy_id": "POLICY_004",
    },
    {
        "query": "merchant with several recent unrelated fraud reports and a high chargeback rate",
        "expected_case_id": "CASE_005",
        "expected_policy_id": "POLICY_008",
    },
    {
        "query": "tiny one dollar charge followed minutes later by a large purchase at the same new merchant",
        "expected_case_id": "CASE_018",
        "expected_policy_id": "POLICY_006",
    },
    {
        "query": "customer's account email changed and a new bill payee added from an unfamiliar device",
        "expected_case_id": "CASE_016",
        "expected_policy_id": "POLICY_007",
    },
]


def _hit_at_k(results, expected_id):
    return any(r["id"] == expected_id for r in results)


def _rank_of(results, expected_id):
    for i, r in enumerate(results):
        if r["id"] == expected_id:
            return i + 1
    return None


def run_evaluation(top_k_cases=3, top_k_policies=2, use_embeddings=True):
    print(f"Running retrieval evaluation (use_embeddings={use_embeddings})")
    print(f"top_k_cases={top_k_cases}, top_k_policies={top_k_policies}")
    print(f"{len(TEST_QUERIES)} test queries\n")
    print("-" * 90)

    case_hits = 0
    policy_hits = 0

    for t in TEST_QUERIES:
        case_results = search_similar_cases(t["query"], top_k=top_k_cases, use_embeddings=use_embeddings)
        policy_results = search_risk_policy(t["query"], top_k=top_k_policies, use_embeddings=use_embeddings)

        case_hit = _hit_at_k(case_results, t["expected_case_id"])
        policy_hit = _hit_at_k(policy_results, t["expected_policy_id"])
        case_rank = _rank_of(case_results, t["expected_case_id"])
        policy_rank = _rank_of(policy_results, t["expected_policy_id"])

        case_hits += int(case_hit)
        policy_hits += int(policy_hit)

        print(f"Query: {t['query']}")
        print(f"  Expected case:   {t['expected_case_id']:10s} | hit@{top_k_cases}: {case_hit!s:5} | rank: {case_rank}")
        print(f"  Expected policy: {t['expected_policy_id']:10s} | hit@{top_k_policies}: {policy_hit!s:5} | rank: {policy_rank}")
        print(f"  Top case IDs returned:   {[r['id'] for r in case_results]}")
        print(f"  Top policy IDs returned: {[r['id'] for r in policy_results]}")
        print("-" * 90)

    n = len(TEST_QUERIES)
    case_hit_rate = case_hits / n
    policy_hit_rate = policy_hits / n

    print(f"\nRESULTS (n={n} queries)")
    print(f"  Case hit-rate@{top_k_cases}:   {case_hits}/{n} = {case_hit_rate:.1%}")
    print(f"  Policy hit-rate@{top_k_policies}: {policy_hits}/{n} = {policy_hit_rate:.1%}")

    return {
        "n_queries": n,
        "case_hit_rate": case_hit_rate,
        "policy_hit_rate": policy_hit_rate,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("BM25-ONLY EVALUATION (always runnable, no external model download needed)")
    print("=" * 90)
    bm25_results = run_evaluation(use_embeddings=False)

    print("\n\n")
    print("=" * 90)
    print("HYBRID (BM25 + EMBEDDINGS) EVALUATION")
    print("=" * 90)
    if not _EMBEDDINGS_AVAILABLE:
        print(
            "sentence-transformers is not installed in this environment.\n"
            "Install it with: pip install sentence-transformers\n"
            "then re-run: python -m rag.evaluate_retrieval"
        )
    elif not embeddings_actually_available():
        print(
            "COULD NOT RUN A REAL HYBRID EVALUATION.\n"
            f"Reason the embedding model failed to load: {_retrieval_module._embedding_load_error}\n\n"
            "This is almost always because the embedding model "
            "('all-MiniLM-L6-v2') needs to download its weights from "
            "huggingface.co on first use, and this environment's network "
            "access does not include that domain.\n\n"
            "IMPORTANT: the numbers above are BM25-only. Do NOT report them "
            "as 'hybrid' results -- they are not.\n\n"
            "To get real hybrid (BM25 + embeddings) numbers, run this exact "
            "command on a machine with normal internet access:\n"
            "  python -m rag.evaluate_retrieval\n"
            "The model (~80MB) downloads once on first run and is cached "
            "under ~/.cache/huggingface for every run after that -- so this "
            "only needs network access the very first time."
        )
    else:
        hybrid_results = run_evaluation(use_embeddings=True)