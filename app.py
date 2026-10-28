import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import sys
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import settings
from src.schemas import (
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieverType,
)
from src.pipeline import FinancialRAGPipeline
from src.cache import query_cache

# Structured JSON Logger Setup
class JSONFormatter(logging.Formatter):
    """Formats log records as structured, machine-parseable JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_payload.update(record.props)
        return json.dumps(log_payload)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(JSONFormatter())
    logger.addHandler(_handler)
logger.propagate = False

# Global pipeline instance (loaded once at startup)
pipeline: FinancialRAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan: Models and indices are warmed up once into memory at startup.
    """
    global pipeline
    logger.info("Initializing Enterprise Financial RAG Service models and vector index...")
    pipeline = FinancialRAGPipeline()
    logger.info("Service initialized: Models, ChromaDB vector store, and BM25 index ready!")
    yield
    logger.info("Shutting down Financial RAG Service...")

app = FastAPI(
    title="Enterprise Financial RAG API",
    description="Production Multi-Retriever Financial Intelligence API with Citation Attribution & Confidence Guardrails.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend applications (React, Vue, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def json_logging_middleware(request: Request, call_next):
    """
    Structured HTTP request/response logging middleware.
    """
    t0 = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - t0) * 1000, 2)
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
        extra={
            "props": {
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": client_host
            }
        }
    )
    return response

# ---------------------------------------------------------
# Root & Information Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Enterprise Financial RAG API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "config": "/config",
        "query": "/query"
    }

# ---------------------------------------------------------
# Health Endpoints (GET /health and GET /v1/health)
# ---------------------------------------------------------
async def _get_health() -> HealthResponse:
    try:
        count = pipeline.retriever.dense.count() if pipeline else 0
        return HealthResponse(
            status="healthy",
            chroma_db_status="connected",
            indexed_chunks_count=count,
            reranker_model=settings.reranker_model_name,
            llm_model=settings.gemini_model_name,
            cache_entries_count=query_cache.size()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", extra={"props": {"error": str(e)}})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Validates vector database connectivity, chunk count, and model readiness."""
    return await _get_health()

@app.get("/v1/health", response_model=HealthResponse, tags=["Health"], include_in_schema=False)
async def health_check_v1():
    """Versioned health check alias."""
    return await _get_health()

# ---------------------------------------------------------
# Config Endpoints (GET /config and GET /v1/config)
# ---------------------------------------------------------
def _get_config() -> ConfigResponse:
    return ConfigResponse(
        embedding_model=settings.embedding_model_name,
        reranker_model=settings.reranker_model_name,
        llm_model=settings.gemini_model_name,
        dense_k=settings.dense_k,
        bm25_k=settings.bm25_k,
        top_n_final=settings.top_n_final,
        confidence_threshold=settings.confidence_threshold,
        cache_enabled=settings.cache_enabled,
        cache_ttl_seconds=settings.cache_ttl_seconds,
        cache_max_size=settings.cache_max_size,
        api_timeout_seconds=settings.api_timeout_seconds,
        default_retriever=settings.default_retriever,
        supported_retrievers=[r.value for r in RetrieverType],
        min_query_len=settings.min_query_len,
        max_query_len=settings.max_query_len
    )

@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Inspects active pipeline hyperparameters and model settings."""
    return _get_config()

@app.get("/v1/config", response_model=ConfigResponse, tags=["Configuration"], include_in_schema=False)
async def get_config_v1():
    """Versioned configuration alias."""
    return _get_config()

# ---------------------------------------------------------
# Query Endpoints (POST /query and POST /v1/query)
# ---------------------------------------------------------
async def _execute_query(request: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not yet initialized. Please wait for warm-up."
        )

    try:
        # Enforce request timeout SLA (default 30s)
        response = await asyncio.wait_for(
            pipeline.arun(request),
            timeout=settings.api_timeout_seconds
        )

        logger.info(
            f"Query completed: '{request.question[:45]}...' [{request.retriever.value}] ({response.latency_ms:.1f}ms)",
            extra={
                "props": {
                    "event": "rag_query_success",
                    "retriever": request.retriever.value,
                    "top_k": request.top_k,
                    "is_cached": response.is_cached,
                    "abstained": response.abstained,
                    "gate_decision": response.gate_decision.value,
                    "confidence_score": round(response.confidence_score, 4),
                    "total_latency_ms": response.latency_ms
                }
            }
        )
        return response

    except asyncio.TimeoutError:
        logger.error(
            f"Query timed out after {settings.api_timeout_seconds}s SLA: '{request.question[:45]}...'",
            extra={
                "props": {
                    "event": "rag_query_timeout",
                    "timeout_seconds": settings.api_timeout_seconds,
                    "retriever": request.retriever.value
                }
            }
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Query processing SLA exceeded maximum timeout of {settings.api_timeout_seconds}s."
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as e:
        logger.error(
            f"Pipeline unhandled error: {str(e)}",
            extra={"props": {"event": "rag_query_error", "error": str(e)}}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing error: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, tags=["RAG Engine"])
async def query_endpoint(request: QueryRequest):
    """
    Executes the Financial RAG Pipeline with user-specified retrieval strategy:
    - `hybrid_rerank`: Hybrid (Dense K=8 + BM25 K=8) + Cross-Encoder Reranker + Confidence Gate (Champion).
    - `hybrid`: Hybrid Reciprocal Rank Fusion without re-ranking.
    - `dense`: Pure semantic vector search via ChromaDB HNSW index.
    - `bm25`: Pure lexical keyword search via BM25Okapi index.
    """
    return await _execute_query(request)

@app.post("/v1/query", response_model=QueryResponse, tags=["RAG Engine"], include_in_schema=False)
async def query_endpoint_v1(request: QueryRequest):
    """Versioned query endpoint alias."""
    return await _execute_query(request)

# ---------------------------------------------------------
# Cache Endpoints (DELETE /cache and DELETE /v1/cache)
# ---------------------------------------------------------
@app.delete("/cache", tags=["Cache Management"])
async def clear_cache():
    """Clears the LRU query cache in memory."""
    query_cache.clear()
    logger.info("In-memory query cache cleared", extra={"props": {"event": "cache_cleared"}})
    return {"message": "In-memory query cache cleared successfully."}

@app.delete("/v1/cache", tags=["Cache Management"], include_in_schema=False)
async def clear_cache_v1():
    """Versioned cache clear alias."""
    return await clear_cache()

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=False)
