"""French Multilingual RAG Engine.

Provides semantic indexing, hybrid retrieval (BM25 + Dense RRF), confidence gating,
and French grounded generation for small-business e-commerce policy inquiries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrenchDoc:
    """Document schema for French shop policies."""

    doc_id: str
    category: str
    title: str
    content: str


@dataclass
class FrenchQueryResult:
    """Standardized response schema for French queries."""

    question: str
    answer: str
    abstained: bool
    confidence_score: float
    retrieval_method: str
    sources: list[dict[str, Any]]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize_french(text: str) -> list[str]:
    """Tokenize French text with accent-preserving lowercase regex."""
    clean = re.sub(r"[^\w\s-]", " ", text.lower(), flags=re.UNICODE)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
    return tokens


class FrenchShopRAGEngine:
    """Production-grade Multilingual RAG Engine for French E-commerce policies."""

    DEFAULT_DATA_PATH = Path("data/french_shop_faq.json")
    DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    ABSTAIN_THRESHOLD = 0.38  # Minimum cosine similarity required to pass gate

    FRENCH_ABSTAIN_TEMPLATE = (
        "Désolé, aucune information officielle dans les conditions générales et politiques de "
        "la boutique ne permet de répondre à cette demande. Notre service client reste joignable "
        "du lundi au vendredi au 01 89 54 20 00 pour vous assister."
    )

    def __init__(
        self,
        data_path: str | Path | None = None,
        model_name: str | None = None,
        abstain_threshold: float = ABSTAIN_THRESHOLD,
    ) -> None:
        self.data_path = Path(data_path) if data_path else self.DEFAULT_DATA_PATH
        self.abstain_threshold = abstain_threshold
        self.docs: list[FrenchDoc] = []
        self.bm25: BM25Okapi | None = None
        self.doc_embeddings: np.ndarray | None = None
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.encoder: SentenceTransformer | None = None

        self._load_data()
        self._init_encoder()
        self._build_indexes()

    def _load_data(self) -> None:
        """Load and validate the 30 synthetic French policy documents."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"French FAQ dataset not found at {self.data_path}")

        with open(self.data_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        self.docs = [
            FrenchDoc(
                doc_id=d["doc_id"],
                category=d.get("category", "GENERAL"),
                title=d.get("title", ""),
                content=d["content"],
            )
            for d in raw_data
        ]
        logger.info(f"Loaded {len(self.docs)} French policy documents from {self.data_path}")

    def _init_encoder(self) -> None:
        """Initialize multilingual sentence transformer with automatic fallback."""
        try:
            logger.info(f"Loading multilingual embedding model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(
                f"Failed to load {self.model_name} ({e}). Falling back to {self.FALLBACK_MODEL_NAME}"
            )
            self.encoder = SentenceTransformer(self.FALLBACK_MODEL_NAME)

    def _build_indexes(self) -> None:
        """Build BM25 keyword index and pre-compute normalized dense embeddings."""
        # 1. BM25 Index
        corpus_texts = [f"{d.title} {d.content}" for d in self.docs]
        tokenized_corpus = [tokenize_french(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Normalized Vectors
        raw_embs = self.encoder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.doc_embeddings = raw_embs / norms
        logger.info(f"Indexed {len(self.docs)} documents in French BM25 and Dense spaces.")

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Retrieve documents using French tokenized BM25."""
        tokens = tokenize_french(query)
        if not tokens or not self.bm25:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Retrieve documents using multilingual cosine similarity."""
        if self.encoder is None or self.doc_embeddings is None:
            return []
        q_emb = self.encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm

        sims = np.dot(self.doc_embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def retrieve_hybrid(
        self, query: str, top_k: int = 3, rrf_k: int = 60
    ) -> list[dict[str, Any]]:
        """Hybrid Reciprocal Rank Fusion of BM25 and multilingual Dense retrieval."""
        bm25_matches = self.search_bm25(query, top_k=10)
        dense_matches = self.search_dense(query, top_k=10)

        # Dense score map for confidence checking
        dense_scores_map = {idx: score for idx, score in dense_matches}

        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_matches):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (idx, _) in enumerate(dense_matches):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        ranked_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = self.docs[idx]
            dense_sim = dense_scores_map.get(idx, 0.0)
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "category": doc.category,
                    "title": doc.title,
                    "content": doc.content,
                    "rrf_score": round(rrf_scores[idx], 5),
                    "dense_similarity": round(float(dense_sim), 4),
                }
            )

        return results

    def verify_confidence(
        self, query: str, retrieved_docs: list[dict[str, Any]]
    ) -> tuple[bool, float]:
        """Circuit breaker: verify whether top match meets semantic relevance threshold."""
        if not retrieved_docs:
            return False, 0.0

        top_match = retrieved_docs[0]
        dense_sim = top_match.get("dense_similarity", 0.0)
        bm25_hits = len(self.search_bm25(query, top_k=1))

        # Pass gate if dense cosine similarity >= threshold or strong keyword match
        passes = (dense_sim >= self.abstain_threshold) or (bm25_hits > 0 and dense_sim >= 0.30)
        return passes, float(dense_sim)

    def generate_french_response(
        self,
        query: str,
        sources: list[dict[str, Any]],
        use_llm: bool = True,
    ) -> str:
        """Synthesize response with French citations [Source N: Titre]."""
        if not sources:
            return self.FRENCH_ABSTAIN_TEMPLATE

        # 1. Check if Gemini LLM generation is requested and available
        if use_llm and settings.gemini_api_key:
            try:
                from google import genai

                client = genai.Client(api_key=settings.gemini_api_key)
                context_blocks = "\n\n".join(
                    [
                        f"[Source {s['rank']}: {s['title']}]\n{s['content']}"
                        for s in sources
                    ]
                )
                prompt = (
                    "Tu es l'assistant officiel du service client de la boutique e-commerce 'Boutique Étoile'. "
                    "Réponds à la question du client UNIQUEMENT à partir des extraits officiels ci-dessous. "
                    "Sois courtois, clair, précis et cite systématiquement ta source sous la forme [Source 1] ou [Source 2]. "
                    "Si les extraits ne contiennent pas l'information demandée, indique que tu ne peux pas répondre.\n\n"
                    f"Extraits officiels:\n{context_blocks}\n\n"
                    f"Question du client: {query}\n\n"
                    "Réponse professionnelle en français:"
                )

                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"LLM generation failed ({e}), falling back to extractive synthesis.")

        # 2. Extractive grounded synthesis fallback (deterministic)
        top = sources[0]
        answer = (
            f"D'après notre politique officielle concernant '{top['title']}' [Source 1]:\n\n"
            f"{top['content']}"
        )
        if len(sources) > 1 and sources[1].get("dense_similarity", 0.0) >= 0.45:
            sec = sources[1]
            answer += f"\n\nComplément d'information ('{sec['title']}' [Source 2]):\n{sec['content']}"

        return answer

    def run(
        self,
        query: str,
        top_k: int = 3,
        use_llm: bool = True,
    ) -> FrenchQueryResult:
        """Execute end-to-end French RAG pipeline with circuit breaker guardrail."""
        start_time = time.perf_counter()

        sources = self.retrieve_hybrid(query, top_k=top_k)
        passed_gate, confidence = self.verify_confidence(query, sources)

        if not passed_gate:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return FrenchQueryResult(
                question=query,
                answer=self.FRENCH_ABSTAIN_TEMPLATE,
                abstained=True,
                confidence_score=confidence,
                retrieval_method="hybrid_rrf",
                sources=sources,
                latency_ms=round(latency_ms, 2),
            )

        answer = self.generate_french_response(query, sources, use_llm=use_llm)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return FrenchQueryResult(
            question=query,
            answer=answer,
            abstained=False,
            confidence_score=confidence,
            retrieval_method="hybrid_rrf",
            sources=sources,
            latency_ms=round(latency_ms, 2),
        )
