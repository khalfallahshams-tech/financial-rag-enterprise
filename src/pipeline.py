import time
import asyncio
from typing import Optional
from src.config import settings
from src.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    GateStatus,
    CitationAuditReport
)
from src.cache import query_cache, InMemoryQueryCache
from src.retrieval.hybrid import HybridRetriever
from src.ranking.reranker import CrossEncoderReranker
from src.guardrails.confidence_gate import ConfidenceGate
from src.guardrails.citation_auditor import CitationAuditor
from src.generation.llm import DefensiveGenerator

class FinancialRAGPipeline:
    """
    Unified Production Pipeline Orchestrator.
    Seamlessly executes the Two-Stage Hybrid Architecture:
      Query -> Cache Check -> Hybrid Retrieval -> Cross-Encoder -> Confidence Gate -> LLM -> Citation Audit.
    Supports both synchronous (.run()) and asynchronous (.arun()) non-blocking execution.
    """
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        gate: Optional[ConfidenceGate] = None,
        generator: Optional[DefensiveGenerator] = None,
        cache: Optional[InMemoryQueryCache] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.gate = gate or ConfidenceGate()
        self.generator = generator or DefensiveGenerator()
        self.cache = cache or query_cache

    def run(self, request: QueryRequest) -> QueryResponse:
        t_start = time.time()
        question = request.question

        # 1. In-Memory Cache Check
        if request.use_cache:
            cached_response = self.cache.get(question)
            if cached_response is not None:
                return cached_response

        # 2. Stage 1: Hybrid Candidate Retrieval (Dense K=8 + BM25 K=8)
        t_ret = time.time()
        candidates = self.retriever.retrieve(
            query=question,
            dense_k=settings.dense_k,
            bm25_k=settings.bm25_k
        )
        retrieval_ms = (time.time() - t_ret) * 1000

        # 3. Stage 2: Cross-Encoder Neural Re-Ranking
        t_rerank = time.time()
        ranked_top = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_n=settings.top_n_final
        )
        rerank_ms = (time.time() - t_rerank) * 1000

        # 4. Stage 3: Confidence Gate & Circuit Breaker (tau = 0.0)
        gate_status, top_score, refusal_reason = self.gate.evaluate(ranked_top)

        # Build SourceCitation models
        citations = []
        for rank_idx, cand in enumerate(ranked_top, 1):
            tag = f"Source {rank_idx}"
            cand["source_tag"] = tag
            citations.append(SourceCitation(
                source_tag=tag,
                chunk_id=cand["chunk_id"],
                score=cand["score"],
                snippet=cand["text"][:180] + "..."
            ))

        # 5. Circuit Breaker Activated (Early Exit - 0 LLM Cost)
        if gate_status == GateStatus.BLOCKED:
            total_ms = (time.time() - t_start) * 1000
            response = QueryResponse(
                question=question,
                answer=refusal_reason,
                gate_decision=GateStatus.BLOCKED,
                confidence_score=top_score,
                citations=citations,
                citation_audit=CitationAuditReport(),
                is_cached=False,
                retrieval_latency_ms=retrieval_ms,
                rerank_latency_ms=rerank_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_ms
            )
            if request.use_cache:
                self.cache.set(question, response)
            return response

        # 6. Stage 4: Defensive LLM Generation
        answer_text, llm_ms = self.generator.generate(
            question=question,
            sources=ranked_top
        )

        # 7. Citation Attribution Audit
        audit_report = CitationAuditor.audit(
            text=answer_text,
            retrieved_sources=ranked_top
        )

        total_ms = (time.time() - t_start) * 1000
        response = QueryResponse(
            question=question,
            answer=answer_text,
            gate_decision=GateStatus.PASSED,
            confidence_score=top_score,
            citations=citations,
            citation_audit=audit_report,
            is_cached=False,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms
        )

        # Save to cache
        if request.use_cache:
            self.cache.set(question, response)

        return response

    async def arun(self, request: QueryRequest) -> QueryResponse:
        """
        Asynchronous non-blocking wrapper running heavy CPU/network operations
        in thread pools to prevent blocking FastAPI's async event loop.
        """
        return await asyncio.to_thread(self.run, request)

