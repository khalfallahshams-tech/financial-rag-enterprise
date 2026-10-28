from typing import List, Dict, Any
from src.config import settings
from .dense import DenseRetriever
from .sparse import SparseRetriever

class HybridRetriever:
    """
    Candidate Pool Generator combining Dense Vector Search (K=8) and BM25 Lexical Search (K=8).
    Delivers a deduplicated union pool of 12-16 high-signal candidates to the Cross-Encoder.
    """
    def __init__(self, dense_retriever: DenseRetriever = None, sparse_retriever: SparseRetriever = None):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()

    def retrieve(self, query: str, dense_k: int = settings.dense_k, bm25_k: int = settings.bm25_k) -> List[Dict[str, Any]]:
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

