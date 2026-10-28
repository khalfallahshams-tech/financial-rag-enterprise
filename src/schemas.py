from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from src.config import settings

class GateStatus(str, Enum):
    PASSED = "PASSED (Gate Open)"
    BLOCKED = "BLOCKED (Circuit Breaker Tripped)"

class RetrieverType(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"

class SourceItem(BaseModel):
    """
    Standardized source item schema matching the portfolio specification:
    sources[{id, text, score}]
    """
    id: str = Field(..., description="Document or chunk identifier.")
    text: str = Field(..., description="Passage text snippet used for answer grounding.")
    score: float = Field(..., description="Relevance score (BM25 score, cosine similarity, or cross-encoder logit).")

class QueryRequest(BaseModel):
    """
    Validated user input request schema (The Bouncer).
    Enforces minimum and maximum character lengths to block malicious or malformed inputs.
    """
    question: str = Field(
        ...,
        min_length=settings.min_query_len,
        max_length=settings.max_query_len,
        description="The financial question to be evaluated and answered by the RAG pipeline.",
        examples=["How do I properly endorse a check over to a third party?"]
    )
    top_k: int = Field(
        default=settings.top_n_final,
        ge=1,
        le=20,
        description="Number of top source passages to return for grounding."
    )
    retriever: RetrieverType = Field(
        default=RetrieverType.HYBRID_RERANK,
        description="Retrieval strategy: bm25, dense, hybrid, or hybrid_rerank."
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check the in-memory cache before executing full retrieval."
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        clean_text = v.strip()
        if not clean_text:
            raise ValueError("Question cannot be empty or purely whitespace.")
        return clean_text

class SourceCitation(BaseModel):
    """
    A single verified source chunk returned from the retrieval and re-ranking stage.
    """
    source_tag: str = Field(..., description="Prompt source identifier (e.g., 'Source 1').")
    chunk_id: str = Field(..., description="ChromaDB persistent chunk ID (e.g., 'doc_6301_c0').")
    score: float = Field(..., description="Cross-encoder joint attention confidence score.")
    snippet: str = Field(..., description="Preview snippet of the chunk text.")

class CitationAuditReport(BaseModel):
    """
    Metrics computed by the Citation Attribution Engine.
    """
    total_citations: int = 0
    valid_citations: int = 0
    hallucinated_citations: int = 0
    citation_precision: float = 1.0
    total_sentences: int = 0
    cited_sentences: int = 0
    sentence_coverage: float = 1.0

class QueryResponse(BaseModel):
    """
    Strictly formatted enterprise output response schema conforming to both:
    1. Spec format: {question, answer, abstained, sources[{id, text, score}], latency_ms, config}
    2. Deep attribution fields: {gate_decision, confidence_score, citations, citation_audit, ...}
    """
    question: str
    answer: str
    abstained: bool = Field(default=False, description="Whether the system abstained due to circuit breaker or low relevance.")
    sources: List[SourceItem] = Field(default_factory=list, description="Grounding source passages according to API spec.")
    latency_ms: float = Field(default=0.0, description="Total latency in milliseconds.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Active runtime configuration used for this query.")

    # Deep attribution & latency decomposition
    gate_decision: GateStatus = GateStatus.PASSED
    confidence_score: float = Field(default=0.0, description="Top candidate's confidence score.")
    citations: List[SourceCitation] = Field(default_factory=list)
    citation_audit: CitationAuditReport = Field(default_factory=CitationAuditReport)
    expanded_acronyms: List[str] = Field(default_factory=list, description="Financial acronyms expanded prior to retrieval.")
    is_cached: bool = Field(default=False, description="Whether the response was served from memory cache.")
    
    # Latency decomposition in milliseconds
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

class HealthResponse(BaseModel):
    """
    Health check payload verifying vector database and model availability.
    """
    status: str = "healthy"
    chroma_db_status: str
    indexed_chunks_count: int
    reranker_model: str
    llm_model: str
    cache_entries_count: int

class ConfigResponse(BaseModel):
    """
    System configuration inspection payload for GET /config and GET /v1/config.
    """
    embedding_model: str
    reranker_model: str
    llm_model: str
    dense_k: int
    bm25_k: int
    top_n_final: int
    confidence_threshold: float
    cache_enabled: bool
    cache_ttl_seconds: int
    cache_max_size: int
    api_timeout_seconds: float
    default_retriever: str
    supported_retrievers: List[str]
    min_query_len: int
    max_query_len: int


