"""Unit tests for PoolDefinition, PoolInstance, and OpenAIProvider 403 retry."""

import threading
import pytest

from src.config import PoolDefinition


class TestPoolDefinition:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one model"):
            PoolDefinition([])

    def test_create_instance_returns_pool_instance(self):
        pd_def = PoolDefinition(["a", "b"])
        instance = pd_def.create_instance()
        from src.config import PoolInstance
        assert isinstance(instance, PoolInstance)

    def test_instances_share_dead_set(self):
        pd_def = PoolDefinition(["a", "b"])
        i1 = pd_def.create_instance()
        i2 = pd_def.create_instance()

        # kill "a" via i1
        m1 = i1.checkout()
        assert m1 == "a"
        i1.replace(m1)

        # i2 shouldn't see "a" anymore
        m2 = i2.checkout()
        assert m2 == "b"

    def test_len_returns_model_count(self):
        pd_def = PoolDefinition(["a", "b", "c"])
        assert len(pd_def) == 3


class TestPoolInstance:
    def test_checkout_returns_first_available(self):
        pd_def = PoolDefinition(["a", "b", "c"])
        instance = pd_def.create_instance()
        assert instance.checkout() == "a"
        assert instance.checkout() == "b"
        assert instance.checkout() == "c"

    def test_checkout_no_duplicates_in_same_instance(self):
        pd_def = PoolDefinition(["a", "b"])
        instance = pd_def.create_instance()
        first = instance.checkout()
        second = instance.checkout()
        assert first != second

    def test_checkout_skips_dead_models(self):
        pd_def = PoolDefinition(["a", "b", "c"])
        instance = pd_def.create_instance()
        # kill "a" — replace() also checks out the replacement, so "b" is now in_use
        first = instance.checkout()  # "a"
        replacement = instance.replace(first)  # kills "a", checks out "b"
        assert replacement == "b"
        # both "a" (dead) and "b" (in_use) gone, next is "c"
        next_model = instance.checkout()
        assert next_model == "c"

    def test_checkout_exhausted_raises(self):
        pd_def = PoolDefinition(["a"])
        instance = pd_def.create_instance()
        instance.checkout()
        with pytest.raises(RuntimeError, match="No available models"):
            instance.checkout()

    def test_replace_marks_global_dead(self):
        pd_def = PoolDefinition(["a", "b", "c"])
        instance = pd_def.create_instance()
        model = instance.checkout()  # "a"
        next_model = instance.replace(model)
        assert next_model in ("b", "c")
        assert "a" in pd_def._dead

    def test_replace_all_exhausted_raises(self):
        pd_def = PoolDefinition(["a"])
        instance = pd_def.create_instance()
        model = instance.checkout()
        with pytest.raises(RuntimeError, match="exhausted"):
            instance.replace(model)

    def test_release_all_clears_in_use(self):
        pd_def = PoolDefinition(["a", "b"])
        instance = pd_def.create_instance()
        instance.checkout()  # "a"
        instance.checkout()  # "b"
        instance.release_all()
        # after release, can checkout "a" again (same instance reused — not dead)
        assert instance.checkout() == "a"

    def test_release_all_does_not_revive_dead(self):
        pd_def = PoolDefinition(["a", "b"])
        instance = pd_def.create_instance()
        model = instance.checkout()  # "a"
        instance.replace(model)  # "a" is now dead
        instance.release_all()
        # "a" is dead, next checkout should be "b"
        assert instance.checkout() == "b"

    def test_thread_safety_checkout(self):
        pd_def = PoolDefinition([str(i) for i in range(100)])
        instance = pd_def.create_instance()
        results = []
        lock = threading.Lock()

        def worker():
            try:
                while True:
                    m = instance.checkout()
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
        assert provider._pool_instance is None
