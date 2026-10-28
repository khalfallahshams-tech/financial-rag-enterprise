import chromadb
from typing import List, Dict, Any
from src.config import settings

class DenseRetriever:
    """
    Manages semantic vector retrieval against local ChromaDB HNSW collection.
    """
    def __init__(self, chroma_path: str = settings.chroma_path, collection_name: str = settings.dense_collection_name):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int = settings.dense_k) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K dense semantic candidates.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        candidates = []
        docs = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, doc_text, dist, meta in zip(ids, docs, distances, metadatas):
            candidates.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "distance": dist,
                "metadata": meta,
                "source_channel": "dense"
            })

        return candidates

    def count(self) -> int:
        return self.collection.count()

