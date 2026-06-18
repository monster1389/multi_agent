"""Unit tests for ModelPool and OpenAIProvider 403 retry."""

import threading
import pytest

from src.config import ModelPool


class TestModelPool:
    """Test ModelPool allocation, replacement, and thread safety."""

    def test_allocate_pops_models_in_order(self):
        pool = ModelPool(["a", "b", "c"])
        assert len(pool) == 3
        assert pool.allocate() == "a"
        assert len(pool) == 2
        assert pool.allocate() == "b"
        assert pool.allocate() == "c"
        assert len(pool) == 0

    def test_allocate_empty_raises(self):
        pool = ModelPool(["x"])
        pool.allocate()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate()

    def test_replace_removes_failed_and_returns_other(self):
        pool = ModelPool(["a", "b", "c"])
        first = pool.allocate()  # "a"
        result = pool.replace(first)
        assert result in ("b", "c")
        remaining = pool.allocate()
        assert remaining not in (first, result)
        assert len(pool) == 0

    def test_replace_last_model_raises(self):
        pool = ModelPool(["x"])
        first = pool.allocate()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.replace(first)

    def test_replace_unallocated_model_raises(self):
        pool = ModelPool(["a", "b"])
        with pytest.raises(ValueError, match="never allocated"):
            pool.replace("a")

    def test_thread_safety(self):
        pool = ModelPool([str(i) for i in range(100)])
        results = []
        lock = threading.Lock()

        def worker():
            try:
                while True:
                    m = pool.allocate()
                    with lock:
                        results.append(m)
            except RuntimeError:
                pass

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 100
        assert len(set(results)) == 100  # all unique

    def test_empty_pool_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one model"):
            ModelPool([])


class TestProvider403Retry:
    """Test OpenAIProvider quota exhaustion detection."""

    def test_detects_403_status_code(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="test", api_key="sk-test")

        class Fake403(Exception):
            pass

        exc = Fake403("forbidden")
        exc.status_code = 403
        assert provider._is_quota_exhausted(exc) is True

    def test_rejects_401_status_code(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="test", api_key="sk-test")

        class Fake401(Exception):
            pass

        exc = Fake401("unauthorized")
        exc.status_code = 401
        assert provider._is_quota_exhausted(exc) is False

    def test_no_pool_no_retry(self):
        """Without a pool, 403 is raised normally."""
        from src.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="test", api_key="sk-test")

        class Fake403(Exception):
            pass

        # No pool -> should just raise
        assert provider._model_pool is None
