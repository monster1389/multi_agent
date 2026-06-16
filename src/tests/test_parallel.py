"""Test for parallel execution helper."""
import pytest
from src.debate import _run_parallel


def slow_double(agent, multiplier: int) -> int:
    """Slow function that returns agent.id * multiplier."""
    import time
    time.sleep(0.1)
    return agent.id * multiplier


class FakeAgent:
    """Minimal agent stub for testing _run_parallel."""
    def __init__(self, aid: int):
        self.id = aid


class TestRunParallel:
    def test_returns_dict_keyed_by_agent_id(self):
        agents = [FakeAgent(0), FakeAgent(1), FakeAgent(2)]
        results = _run_parallel([
            (slow_double, (a, 2)) for a in agents
        ])
        assert results == {0: 0, 1: 2, 2: 4}

    def test_single_agent(self):
        a = FakeAgent(5)
        results = _run_parallel([(slow_double, (a, 3))])
        assert results == {5: 15}

    def test_first_failure_cancels_others(self):
        def fail_first(agent):
            if agent.id == 0:
                raise ValueError("boom")
            import time
            time.sleep(0.2)
            return agent.id

        agents = [FakeAgent(0), FakeAgent(1)]
        with pytest.raises(ValueError, match="boom"):
            _run_parallel([(fail_first, (a,)) for a in agents])
