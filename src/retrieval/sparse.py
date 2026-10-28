import os
import re
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import chromadb
from src.config import settings, DATA_DIR

BM25_CACHE_PATH = DATA_DIR / "bm25_index.pkl"

class SparseRetriever:
    """
    Manages BM25Okapi sparse lexical retrieval over all corpus chunks.
    Bridges the Acronym Blind Spot by guaranteeing exact keyword and regulatory code recall.
    Persists tokenized index to disk for sub-second startup times.
    """
    def __init__(self, chroma_path: str = settings.chroma_path, collection_name: str = settings.dense_collection_name):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.corpus_documents: List[str] = []
        self.corpus_ids: List[str] = []
        self.bm25: BM25Okapi = None
        self._load_or_build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _load_or_build_index(self):
        if os.path.exists(BM25_CACHE_PATH):
            try:
                with open(BM25_CACHE_PATH, "rb") as f:
                    cache_data = pickle.load(f)
                    self.corpus_documents = cache_data["documents"]
                    self.corpus_ids = cache_data["ids"]
                    self.bm25 = cache_data["bm25"]
                return
            except Exception:
                pass  # Fall back to rebuilding if corrupted

        # Rebuild from ChromaDB collection
        client = chromadb.PersistentClient(path=self.chroma_path)
        col = client.get_collection(name=self.collection_name)
        data = col.get(include=["documents"])
        self.corpus_documents = data["documents"]
        self.corpus_ids = data["ids"]

        tokenized_corpus = [self.tokenize(doc) for doc in self.corpus_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Cache to disk
        try:
            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "documents": self.corpus_documents,
                    "ids": self.corpus_ids,
                    "bm25": self.bm25
                }, f)
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = settings.bm25_k) -> List[Dict[str, Any]]:
        query_tokens = self.tokenize(query)
        top_indices = self.bm25.get_top_n(
            query_tokens,
            list(range(len(self.corpus_documents))),
            n=top_k
        )

        candidates = []
        for idx in top_indices:
            candidates.append({
                "chunk_id": self.corpus_ids[idx],
                "text": self.corpus_documents[idx],
                "distance": 0.0,
                "metadata": {"doc_id": self.corpus_ids[idx]},
                "source_channel": "sparse_bm25"
            })
        return candidates

