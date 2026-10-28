from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from src.cache import query_cache
from src.schemas import GateStatus


@pytest.fixture(scope="module")
def client():
    """
    Spins up the FastAPI TestClient, triggering the lifespan handler to warm up models.
    """
    with TestClient(app) as test_client:
        # Clear cache before running test suite
        query_cache.clear()
        yield test_client
        query_cache.clear()

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Enterprise Financial RAG API"
    assert "health" in data
    assert "config" in data
    assert "query" in data

def test_health_endpoints(client):
    for path in ["/health", "/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["chroma_db_status"] == "connected"
        assert data["indexed_chunks_count"] > 0
        assert "reranker_model" in data
        assert "llm_model" in data

def test_config_endpoints(client):
    for path in ["/config", "/v1/config"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["default_retriever"] == "hybrid_rerank"
        assert "hybrid_rerank" in data["supported_retrievers"]
        assert "bm25" in data["supported_retrievers"]
        assert "dense" in data["supported_retrievers"]
        assert "hybrid" in data["supported_retrievers"]
        assert data["dense_k"] == 8
        assert data["bm25_k"] == 8
        assert data["top_n_final"] == 3
        assert data["confidence_threshold"] == 0.0

def test_query_validation_errors(client):
    # Empty query
    r_empty = client.post("/query", json={"question": ""})
    assert r_empty.status_code == 422

    # Whitespace query
    r_ws = client.post("/query", json={"question": "     "})
    assert r_ws.status_code == 422

    # Too short (< 3 chars)
    r_short = client.post("/query", json={"question": "ab"})
    assert r_short.status_code == 422

    # Invalid retriever name
    r_inv_ret = client.post("/query", json={"question": "What is an ETF?", "retriever": "invalid_engine"})
    assert r_inv_ret.status_code == 422

    # Invalid top_k (< 1)
    r_inv_k = client.post("/query", json={"question": "What is an ETF?", "top_k": 0})
    assert r_inv_k.status_code == 422

def test_query_hybrid_rerank_with_mocked_llm(client):
    # Mock LLM generation to test full pipeline without external network calls
    with patch.object(
        app_module.pipeline.generator,
        "generate",
        return_value=("An ETF is an Exchange Traded Fund [Source 1].", 125.0)
    ):
        payload = {
            "question": "What is an ETF and how does it trade?",
            "top_k": 3,
            "retriever": "hybrid_rerank",
            "use_cache": False
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["question"] == payload["question"]
        assert "ETF" in data["answer"]
        assert data["abstained"] is False
        assert len(data["sources"]) > 0
        assert "id" in data["sources"][0]
        assert "text" in data["sources"][0]
        assert "score" in data["sources"][0]
        assert data["config"]["retriever"] == "hybrid_rerank"
        assert data["latency_ms"] > 0
        assert data["gate_decision"] == GateStatus.PASSED.value

def test_query_retriever_modes(client):
    """Verifies that all 4 specified retriever modes function through POST /query."""
    modes = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    with patch.object(
        app_module.pipeline.generator,
        "generate",
        return_value=("Financial assets are diversified using asset allocation [Source 1].", 100.0)
    ):
        for mode in modes:
            payload = {
                "question": "How should an investor diversify a stock portfolio?",
                "top_k": 2,
                "retriever": mode,
                "use_cache": False
            }
            resp = client.post("/query", json=payload)
            assert resp.status_code == 200, f"Retriever mode {mode} failed with {resp.text}"
            data = resp.json()
            assert data["config"]["retriever"] == mode
            assert len(data["sources"]) <= 2
            assert data["abstained"] is False

def test_circuit_breaker_abstention(client):
    """Nonsense / out-of-domain query should trip the circuit breaker and return abstained=True."""
    payload = {
        "question": "quantum supersymmetric gauge string theory multiversal entropy calculations",
        "retriever": "hybrid_rerank",
        "use_cache": False
    }
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["gate_decision"] == GateStatus.BLOCKED.value
    assert data["abstained"] is True
    assert "not contain sufficient information" in data["answer"].lower()

def test_cache_hit_and_clear(client):
    question = "Can I roll over a traditional 401k to a Roth IRA?"
    with patch.object(
        app_module.pipeline.generator,
        "generate",
        return_value=("Yes, rolling over a 401k to a Roth IRA is allowed [Source 1].", 110.0)
    ):
        # 1st call: fresh run
        r1 = client.post("/query", json={"question": question, "use_cache": True})
        assert r1.status_code == 200
        assert r1.json()["is_cached"] is False

        # 2nd call: cache hit
        r2 = client.post("/query", json={"question": question, "use_cache": True})
        assert r2.status_code == 200
        assert r2.json()["is_cached"] is True

        # Clear cache via DELETE /cache
        r_del = client.delete("/cache")
        assert r_del.status_code == 200

        # 3rd call: fresh run after cache clear
        r3 = client.post("/query", json={"question": question, "use_cache": True})
        assert r3.status_code == 200
        assert r3.json()["is_cached"] is False

def test_timeout_handling(client):
    """Simulates a pipeline hang exceeding timeout SLA to verify HTTP 504 handling."""
    import asyncio

    async def mock_wait_for(fut, timeout):
        # Cleanly close the passed coroutine to avoid unawaited coroutine RuntimeWarning
        if hasattr(fut, "close"):
            fut.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        resp = client.post("/query", json={"question": "What is compounding interest?"})
        assert resp.status_code == 504
        assert "exceeded maximum timeout" in resp.json()["detail"].lower()
