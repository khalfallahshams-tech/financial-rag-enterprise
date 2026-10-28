from typing import Any

from src.config import settings

from .dense import DenseRetriever
from .sparse import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60
) -> list[dict[str, Any]]:
    """
    Computes standard Reciprocal Rank Fusion (RRF) across multiple candidate lists:
        RRF_score(d) = sum_{l in lists} 1.0 / (k + rank_l(d))
    where rank is 1-indexed.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for rank_list in ranked_lists:
        for rank_1idx, item in enumerate(rank_list, start=1):
            doc_id = item["chunk_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(item)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank_1idx))

    # Sort descending by fused RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
    fused_results = []
    for doc_id in sorted_ids:
        item = doc_map[doc_id]
        item["rrf_score"] = rrf_scores[doc_id]
        item["score"] = rrf_scores[doc_id]
        fused_results.append(item)

    return fused_results

class HybridRetriever:
    """
    Candidate Pool Generator combining Dense Vector Search (K=8) and BM25 Lexical Search (K=8).
    Delivers a deduplicated union pool of 12-16 high-signal candidates to the Cross-Encoder.
    """
    def __init__(self, dense_retriever: DenseRetriever = None, sparse_retriever: SparseRetriever = None):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()

    def retrieve(self, query: str, dense_k: int = settings.dense_k, bm25_k: int = settings.bm25_k) -> list[dict[str, Any]]:
        # 1. Retrieve dense semantic candidates
        dense_candidates = self.dense.retrieve(query, top_k=dense_k)

        # 2. Retrieve sparse lexical candidates
        sparse_candidates = self.sparse.retrieve(query, top_k=bm25_k)

        # 3. Deduplicated union pool
        pool_dict = {}
        for cand in dense_candidates:
            pool_dict[cand["chunk_id"]] = cand

        for cand in sparse_candidates:
            cid = cand["chunk_id"]
            if cid not in pool_dict:
                pool_dict[cid] = cand
            else:
                # Chunk found by both dense and sparse!
                pool_dict[cid]["source_channel"] = "hybrid_overlap"

        return list(pool_dict.values())
