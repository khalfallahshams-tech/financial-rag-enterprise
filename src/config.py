import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
RAW_DATA_DIR = DATA_DIR / "raw"

@dataclass(frozen=True)
class Settings:
    """
    Enterprise RAG System Configuration.
    All hyperparameters are empirically grounded from our benchmark experiments.
    """
    # 1. Vector Database & Embedding Models
    chroma_path: str = str(CHROMA_DIR)
    dense_collection_name: str = "fiqa_full_corpus"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # 2. Retrieval Parameters (Pareto-Optimal sweep results)
    dense_k: int = 8          # Top candidates from dense vector search (slashed latency by -56%)
    bm25_k: int = 8           # Top candidates from BM25 lexical search (bridges the Acronym Gap)
    top_n_final: int = 3      # Final chunks passed to the LLM (eliminates context clutter)

    # 3. Cross-Encoder Re-Ranking Parameters
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    confidence_threshold: float = 0.0  # Logit threshold for Circuit Breaker (tau=0.0)

    # 4. LLM Generation Parameters
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0           # Strict, deterministic responses for financial compliance
    max_retries: int = 3               # Exponential backoff retry limit
    retry_delay_seconds: float = 2.0   # Base delay for rate-limit retries

    # 5. In-Memory Cache Parameters
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600      # 1-hour time-to-live for cached queries
    cache_max_size: int = 1000         # Maximum number of queries held in LRU memory

    # 6. Pydantic Validation Bounds (The Bouncer)
    min_query_len: int = 3             # Rejects empty or single-character inputs
    max_query_len: int = 1000          # Rejects buffer overflow / DOS inputs

    # 7. Server Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_timeout_seconds: float = 60.0  # Maximum request timeout SLA before 504
    default_retriever: str = "hybrid_rerank"

# Global immutable settings instance
settings = Settings()

