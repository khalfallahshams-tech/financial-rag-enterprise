from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from src.config import settings

class GateStatus(str, Enum):
    PASSED = "PASSED (Gate Open)"
    BLOCKED = "BLOCKED (Circuit Breaker Tripped)"

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
    Strictly formatted enterprise output response schema.
    """
    question: str
    answer: str
    gate_decision: GateStatus
    confidence_score: float = Field(..., description="Top candidate's Cross-Encoder logit score.")
    citations: List[SourceCitation] = Field(default_factory=list)
    citation_audit: CitationAuditReport = Field(default_factory=CitationAuditReport)
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

