"""
Candidate Pool Depth Sweep: K=8 vs K=12 vs K=16.
Measures Pareto Trade-off between Recall Ceiling and CPU Latency.
"""

import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def parse_gt(raw_val: Any) -> list[str]:
    try:
        parsed = ast.literal_eval(str(raw_val))
        if isinstance(parsed, list):
            return [norm(c) for c in parsed if norm(c)]
        return [norm(parsed)]
    except Exception:
        return [norm(raw_val)]


def check_hit_rank(retrieved_docs: list[str], gt_contexts: list[str]) -> int | None:
    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_clean = norm(doc)
        for gt in gt_contexts:
            gt_prefix = gt[:50] if len(gt) >= 50 else gt
            if gt_prefix in doc_clean or doc_clean[:50] in gt:
                return rank
    return None


def run_sweep():
    eval_csv = BASE_DIR / "data" / "raw" / "fiqa_eval.csv"
    df_eval = pd.read_csv(eval_csv)
    n = len(df_eval)

    print(f"Sweeping candidate pool depths K in [8, 12, 16] across {n} queries...")

    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    k_values = [8, 12, 16]
    sweep_results = {}

    for k in k_values:
        print(f"\n--- Running Sweep for K={k} ---")
        ranks = []
        latencies = []

        for idx, row in df_eval.iterrows():
            q = row["question"]
            gt = parse_gt(row["contexts"])

            t0 = time.perf_counter()
            cands = hybrid.retrieve(q, dense_k=k, bm25_k=k)
            pairs = [[q, c["text"]] for c in cands]
            scores = reranker.predict(pairs)
            sorted_idx = sorted(range(len(pairs)), key=lambda i: scores[i], reverse=True)[:3]
            top3_docs = [cands[i]["text"] for i in sorted_idx]
            dt = (time.perf_counter() - t0) * 1000

            latencies.append(dt)
            rank = check_hit_rank(top3_docs, gt)
            ranks.append(rank)

        hit_1 = sum(1 for r in ranks if r == 1) / n
        hit_3 = sum(1 for r in ranks if r is not None and r <= 3) / n
        mrr = sum(1.0 / r for r in ranks if r is not None) / n
        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        sweep_results[f"K={k}"] = {
            "Candidate_K_Per_Channel": k,
            "Total_Union_Pool_Size": f"{k} to {k*2}",
            "Hit@1": round(hit_1 * 100, 1),
            "Hit@3": round(hit_3 * 100, 1),
            "MRR": round(mrr, 3),
            "Avg_Latency_ms": round(avg_lat, 1),
            "P95_Latency_ms": round(p95_lat, 1),
        }
        print(f"K={k} -> Hit@1: {hit_1*100:.1f}%, Hit@3: {hit_3*100:.1f}%, MRR: {mrr:.3f}, Latency: {avg_lat:.1f} ms")

    out_json = BASE_DIR / "results" / "candidate_depth_sweep.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PARETO CANDIDATE DEPTH SWEEP RESULTS")
    print("=" * 80)
    df_res = pd.DataFrame.from_dict(sweep_results, orient="index")
    print(df_res.to_string())
    print("=" * 80)


if __name__ == "__main__":
    run_sweep()

