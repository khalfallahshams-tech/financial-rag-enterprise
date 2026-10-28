import time

from src.cache import InMemoryQueryCache
from src.schemas import GateStatus, QueryResponse


def create_mock_response(question: str) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer="Mock answer.",
        gate_decision=GateStatus.PASSED,
        confidence_score=4.5
    )

def test_cache_set_and_get():
    cache = InMemoryQueryCache(max_size=10, ttl_seconds=60)
    resp = create_mock_response("How to endorse a check?")
    cache.set("How to endorse a check?", resp)

    cached = cache.get("How to endorse a check?")
    assert cached is not None
    assert cached.is_cached is True
    assert cached.answer == "Mock answer."

def test_cache_normalization():
    cache = InMemoryQueryCache(max_size=10, ttl_seconds=60)
    resp = create_mock_response("How to endorse a check?")
    cache.set("How to endorse a check?", resp)

    # Different casing and extra punctuation
    cached = cache.get("   how to endorse a check???   ")
    assert cached is not None
    assert cached.is_cached is True

def test_cache_lru_eviction():
    cache = InMemoryQueryCache(max_size=2, ttl_seconds=60)
    cache.set("q1", create_mock_response("q1"))
    cache.set("q2", create_mock_response("q2"))
    # Access q1 so q2 becomes oldest
    cache.get("q1")
    # Insert q3, which should evict q2
    cache.set("q3", create_mock_response("q3"))

    assert cache.get("q1") is not None
    assert cache.get("q3") is not None
    assert cache.get("q2") is None  # Evicted!

def test_cache_ttl_expiration():
    cache = InMemoryQueryCache(max_size=10, ttl_seconds=1)
    cache.set("q1", create_mock_response("q1"))
    time.sleep(1.1)
    assert cache.get("q1") is None  # Expired!

