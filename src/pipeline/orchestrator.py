import asyncio
import time

from src.cache import InMemoryQueryCache, query_cache
from src.config import settings
from src.generation.llm import DefensiveGenerator
from src.guardrails.citation_auditor import CitationAuditor
from src.guardrails.confidence_gate import ConfidenceGate
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import FinancialAcronymExpander
from src.schemas import (
    CitationAuditReport,
    GateStatus,
    QueryRequest,
    QueryResponse,
    RetrieverType,
    SourceCitation,
    SourceItem,
)


class FinancialRAGPipeline:
    """
    Unified Production Pipeline Orchestrator.
    Seamlessly executes the Two-Stage Hybrid Architecture:
      Query -> Cache Check -> Acronym Expansion -> Hybrid Retrieval -> Cross-Encoder -> Confidence Gate -> LLM -> Citation Audit.
    Supports both synchronous (.run()) and asynchronous (.arun()) non-blocking execution.
    """
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        gate: ConfidenceGate | None = None,
        generator: DefensiveGenerator | None = None,
        cache: InMemoryQueryCache | None = None,
        expander: FinancialAcronymExpander | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.gate = gate or ConfidenceGate()
        self.generator = generator or DefensiveGenerator()
        self.cache = cache or query_cache
        self.expander = expander or FinancialAcronymExpander()

    def run(self, request: QueryRequest) -> QueryResponse:
        t_start = time.time()
        question = request.question

        # 1. In-Memory Cache Check (Scoped by retriever, top_k, and normalized query)
        cache_key = f"{request.retriever.value}:{request.top_k}:{question}"
        if request.use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                return cached_response

        # 2. Query Pre-Processing: Domain Acronym Expansion
        expanded_query, expanded_acronyms = self.expander.expand(question)

        # 3. Execution according to Retriever Strategy
        retrieval_ms = 0.0
        rerank_ms = 0.0
        top_score = 0.0
        gate_status = GateStatus.PASSED
        refusal_reason = ""
        ranked_top = []

        retriever_mode = request.retriever

        if retriever_mode == RetrieverType.BM25:
            t_ret = time.time()
            candidates = self.retriever.sparse.retrieve(expanded_query, top_k=request.top_k)
            retrieval_ms = (time.time() - t_ret) * 1000
            for cand in candidates:
                cand.setdefault("score", 1.0)
            ranked_top = candidates[:request.top_k]
            if not ranked_top:
                gate_status = GateStatus.BLOCKED
                top_score = -99.0
                refusal_reason = "No candidate documents retrieved by BM25."
            else:
                top_score = ranked_top[0].get("score", 1.0)

        elif retriever_mode == RetrieverType.DENSE:
            t_ret = time.time()
            candidates = self.retriever.dense.retrieve(expanded_query, top_k=request.top_k)
            retrieval_ms = (time.time() - t_ret) * 1000
            for cand in candidates:
                cand["score"] = float(1.0 - cand.get("distance", 1.0))
            ranked_top = candidates[:request.top_k]
            if not ranked_top:
                gate_status = GateStatus.BLOCKED
                top_score = -99.0
                refusal_reason = "No candidate documents retrieved by Dense vector search."
            else:
                top_score = ranked_top[0].get("score", 0.5)

        elif retriever_mode == RetrieverType.HYBRID:
            t_ret = time.time()
            candidates = self.retriever.retrieve(
                query=expanded_query,
                dense_k=request.top_k,
                bm25_k=request.top_k
            )
            retrieval_ms = (time.time() - t_ret) * 1000
            for idx, cand in enumerate(candidates):
                cand.setdefault("score", float(1.0 / (idx + 1)))
            ranked_top = candidates[:request.top_k]
            if not ranked_top:
                gate_status = GateStatus.BLOCKED
                top_score = -99.0
                refusal_reason = "No candidate documents retrieved by Hybrid search."
            else:
                top_score = ranked_top[0].get("score", 0.5)

        else:  # RetrieverType.HYBRID_RERANK (Two-Stage Champion)
            t_ret = time.time()
            candidates = self.retriever.retrieve(
                query=expanded_query,
                dense_k=settings.dense_k,
                bm25_k=settings.bm25_k
            )
            retrieval_ms = (time.time() - t_ret) * 1000

            t_rerank = time.time()
            ranked_top = self.reranker.rerank(
                query=expanded_query,
                candidates=candidates,
                top_n=request.top_k
            )
            rerank_ms = (time.time() - t_rerank) * 1000

            gate_status, top_score, refusal_reason = self.gate.evaluate(ranked_top)

        # Build SourceCitation models and standard SourceItem list
        citations = []
        sources = []
        for rank_idx, cand in enumerate(ranked_top, 1):
            tag = f"Source {rank_idx}"
            cand["source_tag"] = tag
            s_val = float(cand.get("score", 0.0))
            citations.append(SourceCitation(
                source_tag=tag,
                chunk_id=cand["chunk_id"],
                score=s_val,
                snippet=cand["text"][:180] + "..."
            ))
            sources.append(SourceItem(
                id=cand["chunk_id"],
                text=cand["text"][:300],
                score=s_val
            ))

        cfg_summary = {
            "retriever": retriever_mode.value,
            "top_k": request.top_k,
            "reranker": settings.reranker_model_name if retriever_mode == RetrieverType.HYBRID_RERANK else None,
            "llm_model": settings.gemini_model_name
        }

        # Circuit Breaker Activated (Early Exit - 0 LLM Cost)
        if gate_status == GateStatus.BLOCKED:
            total_ms = (time.time() - t_start) * 1000
            response = QueryResponse(
                question=question,
                answer=refusal_reason,
                abstained=True,
                sources=sources,
                latency_ms=total_ms,
                config=cfg_summary,
                gate_decision=GateStatus.BLOCKED,
                confidence_score=top_score,
                citations=citations,
                citation_audit=CitationAuditReport(),
                expanded_acronyms=expanded_acronyms,
                is_cached=False,
                retrieval_latency_ms=retrieval_ms,
                rerank_latency_ms=rerank_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_ms
            )
            if request.use_cache:
                self.cache.set(cache_key, response)
            return response

        # Defensive LLM Generation
        answer_text, llm_ms = self.generator.generate(
            question=question,
            sources=ranked_top
        )

        # Check if generation abstained
        is_abstention = (
            "does not contain sufficient information" in answer_text.lower()
            or "not enough information" in answer_text.lower()
            or "cannot answer" in answer_text.lower()
        )

        # Citation Attribution Audit
        audit_report = CitationAuditor.audit(
            text=answer_text,
            retrieved_sources=ranked_top
        )

        total_ms = (time.time() - t_start) * 1000
        response = QueryResponse(
            question=question,
            answer=answer_text,
            abstained=is_abstention,
            sources=sources,
            latency_ms=total_ms,
            config=cfg_summary,
            gate_decision=GateStatus.PASSED,
            confidence_score=top_score,
            citations=citations,
            citation_audit=audit_report,
            expanded_acronyms=expanded_acronyms,
            is_cached=False,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms
        )

        # Save to cache
        if request.use_cache:
            self.cache.set(cache_key, response)

        return response

    async def arun(self, request: QueryRequest) -> QueryResponse:
        """
        Asynchronous non-blocking wrapper running heavy CPU/network operations
        in thread pools to prevent blocking FastAPI's async event loop.
        """
        return await asyncio.to_thread(self.run, request)

