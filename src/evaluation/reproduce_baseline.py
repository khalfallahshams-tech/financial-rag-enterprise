"""
Benchmark Reproduction Script for FiQA Retrieval Baseline.
Evaluates:
1. Dense Retrieval (K=8)
2. BM25 Sparse Retrieval (K=8)
3. Hybrid Pool (Dense K=8 + BM25 K=8)
4. Two-Stage Champion: Hybrid (K=8) + Cross-Encoder Re-Ranking (Top-3)
Across all 30 human-annotated queries in data/raw/fiqa_eval.csv.
"""

import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for robust matching across raw scrapes."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def parse_ground_truth_contexts(raw_val: Any) -> list[str]:
    """Safely extract normalized ground-truth context strings."""
    try:
        parsed = ast.literal_eval(str(raw_val))
        if isinstance(parsed, list):
            return [normalize_text(c) for c in parsed if normalize_text(c)]
        return [normalize_text(parsed)]
    except Exception:
        return [normalize_text(raw_val)]


def check_hit_rank(retrieved_docs: list[str], ground_truth_contexts: list[str]) -> int | None:
    """
    Check if any ground-truth context matches retrieved chunks under whitespace normalization.
    Uses robust 50-char sliding prefix/containment check.
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_clean = normalize_text(doc)
        for gt in ground_truth_contexts:
            gt_snippet = gt[:50] if len(gt) >= 50 else gt
            if gt_snippet in doc_clean or doc_clean[:50] in gt:
                return rank
    return None


def evaluate_retriever_pipeline(
    df_eval: pd.DataFrame,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> dict[str, Any]:
    """Execute evaluation across the 30 queries for all 4 retrieval strategies."""

    strategies = ["Dense (K=8)", "BM25 (K=8)", "Hybrid Union (K=16)", "Hybrid + Re-Ranker (Top-3)"]
    records = {s: [] for s in strategies}
    latencies = {s: [] for s in strategies}

    print(f"Starting benchmark reproduction across {len(df_eval)} queries...")

    for idx, row in df_eval.iterrows():
        q = row["question"]
        gt_contexts = parse_ground_truth_contexts(row["contexts"])

        # 1. Dense Only (K=8)
        t0 = time.perf_counter()
        dense_cands = dense_retriever.retrieve(q, top_k=8)
        latencies["Dense (K=8)"].append((time.perf_counter() - t0) * 1000)
        dense_docs = [c["text"] for c in dense_cands]
        dense_rank = check_hit_rank(dense_docs, gt_contexts)
        records["Dense (K=8)"].append(dense_rank)

        # 2. BM25 Only (K=8)
        t0 = time.perf_counter()
        bm25_cands = sparse_retriever.retrieve(q, top_k=8)
        latencies["BM25 (K=8)"].append((time.perf_counter() - t0) * 1000)
        bm25_docs = [c["text"] for c in bm25_cands]
        bm25_rank = check_hit_rank(bm25_docs, gt_contexts)
        records["BM25 (K=8)"].append(bm25_rank)

        # 3. Hybrid Pool (Dense K=8 + BM25 K=8)
        t0 = time.perf_counter()
        hybrid_cands = hybrid_retriever.retrieve(q, dense_k=8, bm25_k=8)
        latencies["Hybrid Union (K=16)"].append((time.perf_counter() - t0) * 1000)
        hybrid_docs = [c["text"] for c in hybrid_cands]
        hybrid_rank = check_hit_rank(hybrid_docs, gt_contexts)
        records["Hybrid Union (K=16)"].append(hybrid_rank)

        # 4. Hybrid + Cross-Encoder Re-Ranker (Top-3)
        t0 = time.perf_counter()
        reranked_cands = reranker.rerank(q, hybrid_cands, top_n=3)
        latencies["Hybrid + Re-Ranker (Top-3)"].append((time.perf_counter() - t0) * 1000)
        reranked_docs = [c["text"] for c in reranked_cands]
        reranked_rank = check_hit_rank(reranked_docs, gt_contexts)
        records["Hybrid + Re-Ranker (Top-3)"].append(reranked_rank)

    # Compute aggregate metrics
    results_summary = {}
    for s in strategies:
        ranks = records[s]
        n = len(ranks)
        hit_1 = sum(1 for r in ranks if r is not None and r == 1) / n
        hit_3 = sum(1 for r in ranks if r is not None and r <= 3) / n
        hit_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
        mrr = sum(1.0 / r for r in ranks if r is not None) / n
        avg_lat = float(np.mean(latencies[s]))
        p95_lat = float(np.percentile(latencies[s], 95))

        results_summary[s] = {
            "Hit@1": round(hit_1 * 100, 1),
            "Hit@3": round(hit_3 * 100, 1),
            "Hit@5": round(hit_5 * 100, 1),
            "MRR": round(mrr, 3),
            "Avg_Latency_ms": round(avg_lat, 1),
            "P95_Latency_ms": round(p95_lat, 1),
            "Hit@1_Count": f"{sum(1 for r in ranks if r is not None and r == 1)}/{n}",
            "Hit@3_Count": f"{sum(1 for r in ranks if r is not None and r <= 3)}/{n}",
        }

    return results_summary


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_csv = base_dir / "data" / "raw" / "fiqa_eval.csv"
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_eval = pd.read_csv(eval_csv)
    print(f"Loaded {len(df_eval)} queries from {eval_csv}")

    print("Initializing retrievers and cross-encoder...")
    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    reranker = CrossEncoderReranker()

    results = evaluate_retriever_pipeline(
        df_eval=df_eval,
        dense_retriever=dense,
        sparse_retriever=sparse,
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    df_summary = pd.DataFrame.from_dict(results, orient="index")
    print("\n" + "=" * 75)
    print("REPRODUCTION BENCHMARK RESULTS (Suite 1: 30 Academic Queries)")
    print("=" * 75)
    print(df_summary.to_string())
    print("=" * 75)

    # Save to JSON
    out_file = output_dir / "baseline_reproduction_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
