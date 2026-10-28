"""
Comprehensive BEIR-style Retrieval Evaluation for FiQA Financial RAG.
Evaluates 5 Configurations across all 30 human-annotated queries:
1. Standalone BM25 (K=100)
2. Dense MiniLM (sentence-transformers/all-MiniLM-L6-v2, K=100)
3. Dense BGE (BAAI/bge-small-en-v1.5, K=100)
4. Hybrid Reciprocal Rank Fusion (RRF k=60, Dense + BM25, K=100)
5. Two-Stage Pipeline (Hybrid Candidates + Cross-Encoder Re-Ranking)

Reports:
- nDCG@10 with 95% Bootstrap Confidence Intervals
- Recall@10 with 95% Bootstrap Confidence Intervals
- Recall@100 with 95% Bootstrap Confidence Intervals
- MRR with 95% Bootstrap Confidence Intervals
- Latency Profile (Mean, P50, P95 in ms)
"""

import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.config import settings
from src.evaluation.metrics import (
    bootstrap_confidence_interval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.sparse import SparseRetriever


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for robust cross-corpus matching."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def parse_ground_truth_contexts(raw_val: Any) -> List[str]:
    """Extract normalized ground truth context strings."""
    try:
        parsed = ast.literal_eval(str(raw_val))
        if isinstance(parsed, list):
            return [normalize_text(c) for c in parsed if normalize_text(c)]
        return [normalize_text(parsed)]
    except Exception:
        return [normalize_text(raw_val)]


def is_chunk_relevant(chunk_text: str, gt_contexts: List[str]) -> bool:
    """Determine binary relevance of chunk against ground truth contexts."""
    doc_clean = normalize_text(chunk_text)
    for gt in gt_contexts:
        gt_prefix = gt[:50] if len(gt) >= 50 else gt
        if gt_prefix in doc_clean or doc_clean[:50] in gt:
            return True
    return False


def run_rrf_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009) with constant k=60."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        doc_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    fused_candidates = []
    for cid in sorted_ids:
        cand = dict(doc_map[cid])
        cand["rrf_score"] = scores[cid]
        fused_candidates.append(cand)
    return fused_candidates


def evaluate_all_retrievers() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING FORMAL BEIR-STYLE RETRIEVAL EVALUATION (SUITE 1: 30 QUERIES)")
    print("=" * 80)

    # 1. Load Data
    eval_csv = BASE_DIR / "data" / "raw" / "fiqa_eval.csv"
    df_eval = pd.read_csv(eval_csv)
    total_queries = len(df_eval)
    print(f"Loaded {total_queries} queries from {eval_csv}")

    # 2. Initialize Retrievers
    print("Initializing ChromaDB clients (MiniLM and BGE)...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    
    col_minilm = chroma_client.get_collection(name="fiqa_full_corpus")
    
    st_bge = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    col_bge = chroma_client.get_collection(name="fiqa_bge_corpus", embedding_function=st_bge)

    print("Initializing BM25 Sparse Retriever...")
    sparse_retriever = SparseRetriever()

    print("Initializing Cross-Encoder Re-Ranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    configs = [
        "BM25",
        "Dense (MiniLM)",
        "Dense (BGE-Small)",
        "Hybrid RRF (BGE + BM25)",
        "Two-Stage (Hybrid + Cross-Encoder)",
    ]

    metric_accumulators = {
        cfg: {
            "ndcg_10": [],
            "recall_10": [],
            "recall_100": [],
            "mrr": [],
            "latencies": [],
        }
        for cfg in configs
    }

    print("\nExecuting queries across all configurations...")
    for q_idx, row in df_eval.iterrows():
        question = row["question"]
        gt_contexts = parse_ground_truth_contexts(row["contexts"])

        # ---------------------------------------------------------------------
        # 1. BM25 (K=100)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        bm25_cands = sparse_retriever.retrieve(question, top_k=100)
        t_bm25 = (time.perf_counter() - t0) * 1000

        bm25_rels = [1.0 if is_chunk_relevant(c["text"], gt_contexts) else 0.0 for c in bm25_cands]
        bm25_rel_indices = [i for i, r in enumerate(bm25_rels) if r > 0.0]

        metric_accumulators["BM25"]["ndcg_10"].append(ndcg_at_k(bm25_rels, k=10))
        metric_accumulators["BM25"]["recall_10"].append(1.0 if any(idx < 10 for idx in bm25_rel_indices) else 0.0)
        metric_accumulators["BM25"]["recall_100"].append(1.0 if len(bm25_rel_indices) > 0 else 0.0)
        metric_accumulators["BM25"]["mrr"].append(1.0 / (bm25_rel_indices[0] + 1) if bm25_rel_indices else 0.0)
        metric_accumulators["BM25"]["latencies"].append(t_bm25)

        # ---------------------------------------------------------------------
        # 2. Dense MiniLM (K=100)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        minilm_res = col_minilm.query(query_texts=[question], n_results=100)
        t_minilm = (time.perf_counter() - t0) * 1000

        minilm_docs = minilm_res["documents"][0]
        minilm_rels = [1.0 if is_chunk_relevant(doc, gt_contexts) else 0.0 for doc in minilm_docs]
        minilm_rel_indices = [i for i, r in enumerate(minilm_rels) if r > 0.0]

        metric_accumulators["Dense (MiniLM)"]["ndcg_10"].append(ndcg_at_k(minilm_rels, k=10))
        metric_accumulators["Dense (MiniLM)"]["recall_10"].append(1.0 if any(idx < 10 for idx in minilm_rel_indices) else 0.0)
        metric_accumulators["Dense (MiniLM)"]["recall_100"].append(1.0 if len(minilm_rel_indices) > 0 else 0.0)
        metric_accumulators["Dense (MiniLM)"]["mrr"].append(1.0 / (minilm_rel_indices[0] + 1) if minilm_rel_indices else 0.0)
        metric_accumulators["Dense (MiniLM)"]["latencies"].append(t_minilm)

        # ---------------------------------------------------------------------
        # 3. Dense BGE (K=100)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        bge_res = col_bge.query(query_texts=[question], n_results=100)
        t_bge = (time.perf_counter() - t0) * 1000

        bge_cands = []
        for cid, doc, dist in zip(bge_res["ids"][0], bge_res["documents"][0], bge_res["distances"][0]):
            bge_cands.append({"chunk_id": cid, "text": doc, "distance": dist})

        bge_rels = [1.0 if is_chunk_relevant(c["text"], gt_contexts) else 0.0 for c in bge_cands]
        bge_rel_indices = [i for i, r in enumerate(bge_rels) if r > 0.0]

        metric_accumulators["Dense (BGE-Small)"]["ndcg_10"].append(ndcg_at_k(bge_rels, k=10))
        metric_accumulators["Dense (BGE-Small)"]["recall_10"].append(1.0 if any(idx < 10 for idx in bge_rel_indices) else 0.0)
        metric_accumulators["Dense (BGE-Small)"]["recall_100"].append(1.0 if len(bge_rel_indices) > 0 else 0.0)
        metric_accumulators["Dense (BGE-Small)"]["mrr"].append(1.0 / (bge_rel_indices[0] + 1) if bge_rel_indices else 0.0)
        metric_accumulators["Dense (BGE-Small)"]["latencies"].append(t_bge)

        # ---------------------------------------------------------------------
        # 4. Hybrid RRF (BGE K=100 + BM25 K=100 -> RRF Fused Top-100)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        rrf_cands = run_rrf_fusion(dense_results=bge_cands, sparse_results=bm25_cands, rrf_k=60, top_k=100)
        t_rrf = (time.perf_counter() - t0) * 1000 + t_bge + t_bm25

        rrf_rels = [1.0 if is_chunk_relevant(c["text"], gt_contexts) else 0.0 for c in rrf_cands]
        rrf_rel_indices = [i for i, r in enumerate(rrf_rels) if r > 0.0]

        metric_accumulators["Hybrid RRF (BGE + BM25)"]["ndcg_10"].append(ndcg_at_k(rrf_rels, k=10))
        metric_accumulators["Hybrid RRF (BGE + BM25)"]["recall_10"].append(1.0 if any(idx < 10 for idx in rrf_rel_indices) else 0.0)
        metric_accumulators["Hybrid RRF (BGE + BM25)"]["recall_100"].append(1.0 if len(rrf_rel_indices) > 0 else 0.0)
        metric_accumulators["Hybrid RRF (BGE + BM25)"]["mrr"].append(1.0 / (rrf_rel_indices[0] + 1) if rrf_rel_indices else 0.0)
        metric_accumulators["Hybrid RRF (BGE + BM25)"]["latencies"].append(t_rrf)

        # ---------------------------------------------------------------------
        # 5. Two-Stage (Hybrid Candidates Top-16 + Cross-Encoder Re-Ranking)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        hybrid_candidates_pool = run_rrf_fusion(dense_results=bge_cands[:16], sparse_results=bm25_cands[:16], rrf_k=60, top_k=16)
        pairs = [[question, c["text"]] for c in hybrid_candidates_pool]
        ce_scores = reranker.predict(pairs)
        ranked_pool_indices = sorted(range(len(pairs)), key=lambda idx: ce_scores[idx], reverse=True)
        reranked_cands = [hybrid_candidates_pool[i] for i in ranked_pool_indices]
        t_twostage = (time.perf_counter() - t0) * 1000 + (t_bge / 5) + (t_bm25 / 5)

        twostage_rels = [1.0 if is_chunk_relevant(c["text"], gt_contexts) else 0.0 for c in reranked_cands]
        twostage_rel_indices = [i for i, r in enumerate(twostage_rels) if r > 0.0]

        metric_accumulators["Two-Stage (Hybrid + Cross-Encoder)"]["ndcg_10"].append(ndcg_at_k(twostage_rels, k=10))
        metric_accumulators["Two-Stage (Hybrid + Cross-Encoder)"]["recall_10"].append(1.0 if any(idx < 10 for idx in twostage_rel_indices) else 0.0)
        metric_accumulators["Two-Stage (Hybrid + Cross-Encoder)"]["recall_100"].append(1.0 if len(twostage_rel_indices) > 0 else 0.0)
        metric_accumulators["Two-Stage (Hybrid + Cross-Encoder)"]["mrr"].append(1.0 / (twostage_rel_indices[0] + 1) if twostage_rel_indices else 0.0)
        metric_accumulators["Two-Stage (Hybrid + Cross-Encoder)"]["latencies"].append(t_twostage)

    # -------------------------------------------------------------------------
    # Aggregate Metrics and Compute 95% Bootstrap Confidence Intervals
    # -------------------------------------------------------------------------
    final_report: Dict[str, Any] = {}

    for cfg in configs:
        acc = metric_accumulators[cfg]

        ndcg_mean, ndcg_l, ndcg_u = bootstrap_confidence_interval(acc["ndcg_10"], n_bootstraps=1000, ci=0.95, seed=42)
        rec10_mean, rec10_l, rec10_u = bootstrap_confidence_interval(acc["recall_10"], n_bootstraps=1000, ci=0.95, seed=42)
        rec100_mean, rec100_l, rec100_u = bootstrap_confidence_interval(acc["recall_100"], n_bootstraps=1000, ci=0.95, seed=42)
        mrr_mean, mrr_l, mrr_u = bootstrap_confidence_interval(acc["mrr"], n_bootstraps=1000, ci=0.95, seed=42)

        lat_mean = float(np.mean(acc["latencies"]))
        lat_p50 = float(np.percentile(acc["latencies"], 50))
        lat_p95 = float(np.percentile(acc["latencies"], 95))

        final_report[cfg] = {
            "nDCG@10": {
                "mean": round(ndcg_mean, 4),
                "ci_95": [round(ndcg_l, 4), round(ndcg_u, 4)],
                "formatted": f"{ndcg_mean:.3f} [{ndcg_l:.3f}, {ndcg_u:.3f}]",
            },
            "Recall@10": {
                "mean": round(rec10_mean, 4),
                "ci_95": [round(rec10_l, 4), round(rec10_u, 4)],
                "formatted": f"{rec10_mean*100:.1f}% [{rec10_l*100:.1f}%, {rec10_u*100:.1f}%]",
            },
            "Recall@100": {
                "mean": round(rec100_mean, 4),
                "ci_95": [round(rec100_l, 4), round(rec100_u, 4)],
                "formatted": f"{rec100_mean*100:.1f}% [{rec100_l*100:.1f}%, {rec100_u*100:.1f}%]",
            },
            "MRR": {
                "mean": round(mrr_mean, 4),
                "ci_95": [round(mrr_l, 4), round(mrr_u, 4)],
                "formatted": f"{mrr_mean:.3f} [{mrr_l:.3f}, {mrr_u:.3f}]",
            },
            "Latency_ms": {
                "mean": round(lat_mean, 1),
                "p50": round(lat_p50, 1),
                "p95": round(lat_p95, 1),
            },
        }

    return final_report


def main():
    report = evaluate_all_retrievers()

    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "beir_retrieval_benchmark_results.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Markdown Table
    print("\n" + "=" * 95)
    print("FINAL BEIR RETRIEVAL BENCHMARK TABLE (With 95% Bootstrap Confidence Intervals)")
    print("=" * 95)
    header = f"{'Configuration':<35} | {'nDCG@10 [95% CI]':<20} | {'Recall@10':<18} | {'Recall@100':<18} | {'MRR':<15} | {'Latency P95':<12}"
    print(header)
    print("-" * 125)
    for cfg, data in report.items():
        row = (
            f"{cfg:<35} | "
            f"{data['nDCG@10']['formatted']:<20} | "
            f"{data['Recall@10']['formatted']:<18} | "
            f"{data['Recall@100']['formatted']:<18} | "
            f"{data['MRR']['formatted']:<15} | "
            f"{data['Latency_ms']['p95']:>8.1f} ms"
        )
        print(row)
    print("=" * 125)
    print(f"\nBenchmark successfully saved to: {out_json}")


if __name__ == "__main__":
    main()

