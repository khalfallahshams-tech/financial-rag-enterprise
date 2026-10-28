"""
Evaluation Regression Guard Test for FiQA Financial RAG.
Evaluates a fixed 5-query subset from the academic evaluation benchmark
and asserts that retrieval nDCG@10 does not drop below the established baseline.

Rationale for Tolerance (0.05):
-------------------------------
The empirical baseline nDCG@10 on this fixed 5-query subset is 0.6708.
A tolerance delta of 0.05 is established based on the observed run-to-run variance
in vector similarity searches and floating-point operations across CPU architectures.
Any drop larger than 0.05 indicates an index corruption, embedding model mismatch,
or retrieval logic regression.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.metrics import ndcg_at_k
from src.evaluation.retrieval_eval import is_chunk_relevant, parse_ground_truth_contexts
from src.retrieval.dense import DenseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_CSV = BASE_DIR / "data" / "raw" / "fiqa_eval.csv"

STORED_BASELINE_NDCG_10 = 0.6708
TOLERANCE = 0.05

def test_retrieval_ndcg_regression_guard():
    if not EVAL_CSV.exists():
        pytest.skip(f"Evaluation benchmark file not found at {EVAL_CSV}")

    df_subset = pd.read_csv(EVAL_CSV).head(5)
    retriever = DenseRetriever()

    ndcg_scores = []
    for _, row in df_subset.iterrows():
        question = row["question"]
        gt_contexts = parse_ground_truth_contexts(row["contexts"])
        results = retriever.retrieve(question, top_k=10)

        binary_rels = [
            1.0 if is_chunk_relevant(r["text"], gt_contexts) else 0.0
            for r in results
        ]
        score = ndcg_at_k(binary_rels, k=10)
        ndcg_scores.append(score)

    mean_ndcg_10 = float(sum(ndcg_scores) / len(ndcg_scores))
    min_allowable_score = STORED_BASELINE_NDCG_10 - TOLERANCE

    print(f"\n[REGRESSION GUARD] Current nDCG@10: {mean_ndcg_10:.4f} | Baseline: {STORED_BASELINE_NDCG_10:.4f} (Min Allowed: {min_allowable_score:.4f})")

    assert mean_ndcg_10 >= min_allowable_score, (
        f"RETRIEVAL REGRESSION DETECTED! Current nDCG@10 ({mean_ndcg_10:.4f}) "
        f"dropped more than tolerance ({TOLERANCE}) below baseline ({STORED_BASELINE_NDCG_10:.4f})."
    )

