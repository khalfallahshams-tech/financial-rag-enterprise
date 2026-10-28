import time
import re
import threading
from collections import OrderedDict
from typing import Optional
from src.config import settings
from src.schemas import QueryResponse

class InMemoryQueryCache:
    """
    Thread-Safe Least-Recently-Used (LRU) In-Memory Cache with TTL.
    Answers frequently asked financial questions in < 1 ms with $0.00 cloud API cost.
    """
    def __init__(self, max_size: int = settings.cache_max_size, ttl_seconds: int = settings.cache_ttl_seconds):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def normalize_key(query: str) -> str:
        """
        Normalizes query text (lowercase, stripped, collapsed whitespace, punctuation trimmed)
        so semantically identical strings produce cache hits.
        """
        text = query.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get(self, query: str) -> Optional[QueryResponse]:
        if not settings.cache_enabled:
            return None

        key = self.normalize_key(query)
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            # Check expiration (TTL)
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Return response marked as cached
            cached_resp: QueryResponse = entry["response"].model_copy(deep=True)
            cached_resp.is_cached = True
            return cached_resp

    def set(self, query: str, response: QueryResponse):
        if not settings.cache_enabled:
            return

        key = self.normalize_key(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = {
                "response": response,
                "timestamp": time.time()
            }
            # Evict oldest entry if max capacity reached
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

# Global singleton cache instance
query_cache = InMemoryQueryCache()

