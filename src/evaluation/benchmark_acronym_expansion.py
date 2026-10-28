"""
Benchmark: Acronym Query Expansion vs Raw Query Baseline.
Evaluates the 10 domain ACRONYM_GAP queries from data/golden_eval_set.json.
Measures Cross-Encoder confidence scores and top-1 relevance lifts.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import FinancialAcronymExpander


def run_acronym_benchmark() -> Dict[str, Any]:
    golden_path = BASE_DIR / "data" / "golden_eval_set.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    acronym_queries = [d for d in golden_set if d.get("category") == "ACRONYM_GAP"]
    print(f"Loaded {len(acronym_queries)} ACRONYM_GAP queries.")

    hybrid = HybridRetriever()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    expander = FinancialAcronymExpander()

    results = []

    for item in acronym_queries:
        qid = item["question_id"]
        q_raw = item["question"]
        q_exp, expanded_terms = expander.expand(q_raw)

        # 1. Baseline: Raw Query
        t0 = time.perf_counter()
        raw_cands = hybrid.retrieve(q_raw, dense_k=8, bm25_k=8)
        raw_pairs = [[q_raw, c["text"]] for c in raw_cands]
        raw_scores = reranker.predict(raw_pairs)
        t_raw = (time.perf_counter() - t0) * 1000
        best_raw_score = float(max(raw_scores)) if len(raw_scores) > 0 else -99.0
        best_raw_doc = raw_cands[int(np.argmax(raw_scores))]["text"][:80]

        # 2. Treatment: Acronym Expanded Query
        t0 = time.perf_counter()
        exp_cands = hybrid.retrieve(q_exp, dense_k=8, bm25_k=8)
        exp_pairs = [[q_exp, c["text"]] for c in exp_cands]
        exp_scores = reranker.predict(exp_pairs)
        t_exp = (time.perf_counter() - t0) * 1000
        best_exp_score = float(max(exp_scores)) if len(exp_scores) > 0 else -99.0
        best_exp_doc = exp_cands[int(np.argmax(exp_scores))]["text"][:80]

        score_diff = best_exp_score - best_raw_score

        results.append({
            "question_id": qid,
            "raw_query": q_raw[:50] + "...",
            "expanded_terms": ", ".join(expanded_terms),
            "baseline_score": round(best_raw_score, 3),
            "expanded_score": round(best_exp_score, 3),
            "confidence_lift": round(score_diff, 3),
            "improved": score_diff > 0.05,
            "raw_latency_ms": round(t_raw, 1),
            "exp_latency_ms": round(t_exp, 1),
        })

    df = pd.DataFrame(results)
    mean_baseline = float(df["baseline_score"].mean())
    mean_expanded = float(df["expanded_score"].mean())
    net_lift = mean_expanded - mean_baseline
    improved_count = int(df["improved"].sum())

    summary = {
        "total_acronym_queries": len(results),
        "mean_baseline_confidence": round(mean_baseline, 3),
        "mean_expanded_confidence": round(mean_expanded, 3),
        "net_confidence_lift": round(net_lift, 3),
        "queries_with_positive_lift": f"{improved_count}/{len(results)} ({improved_count/len(results)*100:.1f}%)",
        "detailed_results": results,
    }

    out_file = BASE_DIR / "results" / "acronym_expansion_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("FINANCIAL ACRONYM QUERY EXPANSION BENCHMARK RESULTS")
    print("=" * 90)
    print(df[["question_id", "expanded_terms", "baseline_score", "expanded_score", "confidence_lift", "improved"]].to_string())
    print("-" * 90)
    print(f"Mean Baseline Score: {mean_baseline:.3f} | Mean Expanded Score: {mean_expanded:.3f} | Lift: +{net_lift:.3f}")
    print(f"Queries with Significant Lift: {improved_count}/{len(results)}")
    print("=" * 90)

    return summary


if __name__ == "__main__":
    run_acronym_benchmark()

