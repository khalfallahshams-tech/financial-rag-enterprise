from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import settings
from src.schemas import QueryRequest, QueryResponse, HealthResponse
from src.pipeline import FinancialRAGPipeline
from src.cache import query_cache

# Global pipeline instance
pipeline: FinancialRAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warm up pipeline and load models into memory
    global pipeline
    print("Starting Enterprise Financial RAG Service...")
    pipeline = FinancialRAGPipeline()
    print("Service initialized and models loaded successfully!")
    yield
    # Shutdown: Clean up resources
    print("Shutting down Financial RAG Service...")

app = FastAPI(
    title="Enterprise Financial RAG API",
    description="Production Two-Stage Hybrid Financial Intelligence API with Citation Attribution & Confidence Guardrails.",
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

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Enterprise Financial RAG API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/v1/health"
    }

@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Validates vector database connectivity, chunk count, and model readiness.
    """
    try:
        count = pipeline.retriever.dense.count()
        return HealthResponse(
            status="healthy",
            chroma_db_status="connected",
            indexed_chunks_count=count,
            reranker_model=settings.reranker_model_name,
            llm_model=settings.gemini_model_name,
            cache_entries_count=query_cache.size()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/v1/query", response_model=QueryResponse, tags=["RAG Engine"])
async def query_financial_intelligence(request: QueryRequest):
    """
    Executes the Two-Stage Hybrid Financial RAG Pipeline asynchronously:
    1. Check LRU In-Memory Cache (sub-millisecond hit).
    2. Hybrid Candidate Retrieval (Dense K=8 + BM25 K=8).
    3. Cross-Encoder Re-Ranking (ms-marco-MiniLM).
    4. Confidence Gate evaluation (stops irrelevant/fake inputs).
    5. Defensive LLM Generation (Gemini 3.5 Flash Lite).
    6. Citation Attribution & Sentence Coverage Audit.
    """
    try:
        response = await pipeline.arun(request)
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing error: {str(e)}"
        )

@app.delete("/v1/cache", tags=["Cache Management"])
async def clear_cache():
    """
    Clears the LRU query cache in memory.
    """
    query_cache.clear()
    return {"message": "In-memory query cache cleared successfully."}

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=False)

