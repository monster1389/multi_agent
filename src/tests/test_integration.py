"""End-to-end integration tests using Mock providers.

Verifies the full pipeline: config → dataset → agents → debate/baseline → sandbox → result.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.config import ExperimentConfig, ProviderConfig, DebateParams, DatasetConfig
from src.experiment import run_single_problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(name: str, N: int, K: int = 0, baseline: bool = False) -> ExperimentConfig:
    """Build an ExperimentConfig with MockProvider-compatible settings.

    Uses 'openai' type with a fake api_key — MockProvider is injected later
    by monkeypatching create_provider.
    """
    providers = [
        ProviderConfig(
            name=f"agent-{i}",
            type="openai",
            model="mock-model",
            api_key="sk-mock",
            base_url="",
        )
        for i in range(N)
    ]
    return ExperimentConfig(
        name=name,
        N=N,
        K=K,
        baseline=baseline,
        providers=providers,
        debate_params=DebateParams(alpha=0.5, gamma=0.3),
        dataset=DatasetConfig(subset_size=10, random_seed=42),
    )


def _make_test_problem(problem_id: str = "test-problem") -> dict:
    """Create a minimal problem dict that passes sandbox execution."""
    return {
        "problem_id": problem_id,
        "difficulty": "Easy",
        "title": "Test Problem",
        "description": "Return the sum of two numbers.",
        "constraints": "",
        "function_signature": "solve(self, a: int, b: int) -> int",
        "starter_code": "class Solution:\n    def solve(self, a: int, b: int) -> int:\n        pass\n",
        "entry_point": "Solution().solve",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(a=1, b=2) == 3\n"
            "    assert candidate(a=-1, b=1) == 0\n"
        ),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselinePipeline:
    """Test the baseline (no debate) pipeline end-to-end."""

    def test_baseline_picks_best_solution(self, monkeypatch):
        """Baseline should generate 5 solutions and pick the one that passes tests."""
        from src.tests.mock_provider import MockProvider

        problem = _make_test_problem()
        config = _make_mock_config("baseline-test", N=2, baseline=True)

        # Create mock providers that return correct/wrong code
        providers = []
        # Agent 0: correct code
        p0 = MockProvider(name="agent-0")
        p0.generate_responses = [
            "class Solution:\n    def solve(self, a, b):\n        return a + b\n"
        ]
        providers.append(p0)
        # Agent 1: wrong code
        p1 = MockProvider(name="agent-1")
        p1.generate_responses = [
            "class Solution:\n    def solve(self, a, b):\n        return a - b\n"
        ]
        providers.append(p1)

        # Monkeypatch create_provider to return our mocks
        call_count = [0]

        def mock_create_provider(pcfg, pools=None):
            idx = call_count[0]
            call_count[0] += 1
            return providers[idx]

        monkeypatch.setattr("src.experiment.create_provider", mock_create_provider)

        result = run_single_problem(problem, config)

        assert result["baseline"] is True
        assert result["pass_at_1"] is True  # Agent 0's code passes
        assert result["debate_rounds"] == 1
        assert result["total_llm_calls"] <= 2

    def test_baseline_result_format(self, monkeypatch):
        """Baseline result should have all required JSONL fields."""
        from src.tests.mock_provider import MockProvider

        problem = _make_test_problem()
        config = _make_mock_config("baseline-test", N=1, baseline=True)

        p = MockProvider(name="agent-0")
        p.generate_responses = [
            "class Solution:\n    def solve(self, a, b):\n        return a + b\n"
        ]

        def mock_create_provider(pcfg, pools=None):
            return p

        monkeypatch.setattr("src.experiment.create_provider", mock_create_provider)

        result = run_single_problem(problem, config)

        expected_keys = {
            "problem_id", "difficulty", "experiment", "N", "K",
            "baseline", "pass_at_1", "test_cases_passed", "test_cases_total",
            "debate_rounds", "total_llm_calls", "winner_provider", "solution_code",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["problem_id"] == "test-problem"
        assert result["difficulty"] == "Easy"


class TestDebatePipeline:
    """Test the debate pipeline end-to-end."""

    def test_debate_produces_valid_result(self, monkeypatch):
        """Full debate with mock providers should produce a valid result."""
        from src.tests.mock_provider import MockProvider

        problem = _make_test_problem()
        config = _make_mock_config("debate-test", N=3, K=1, baseline=False)

        providers = []
        for i in range(3):
            p = MockProvider(name=f"agent-{i}")
            p.generate_responses = [
                "class Solution:\n    def solve(self, a, b):\n        return a + b\n"
            ]
            p.refine_responses = [
                "class Solution:\n    def solve(self, a, b):\n        return a + b\n"
            ] * 5
            p.vote_responses = ["[1]"] * 5  # Always vote for display index 1
            providers.append(p)

        call_count = [0]

        def mock_create_provider(pcfg, pools=None):
            idx = call_count[0]
            call_count[0] += 1
            return providers[idx]

        monkeypatch.setattr("src.experiment.create_provider", mock_create_provider)

        result = run_single_problem(problem, config)

        assert result["baseline"] is False
        assert result["N"] == 3
        assert result["K"] == 1
        # The correct solution should pass tests
        assert result["pass_at_1"] is True
        assert result["debate_rounds"] > 0
        assert result["total_llm_calls"] > 3  # at least initial gen (3) + some rounds

    def test_debate_handles_wrong_solutions(self, monkeypatch):
        """Debate with all-wrong solutions should report pass_at_1=False."""
        from src.tests.mock_provider import MockProvider

        problem = _make_test_problem()
        config = _make_mock_config("debate-test", N=2, K=1, baseline=False)

        providers = []
        for i in range(2):
            p = MockProvider(name=f"agent-{i}")
            p.generate_responses = [
                "class Solution:\n    def solve(self, a, b):\n        return 0  # always wrong\n"
            ]
            p.refine_responses = [
                "class Solution:\n    def solve(self, a, b):\n        return 0\n"
            ] * 5
            p.vote_responses = ["1"] * 5
            providers.append(p)

        call_count = [0]

        def mock_create_provider(pcfg, pools=None):
            idx = call_count[0]
            call_count[0] += 1
            return providers[idx]

        monkeypatch.setattr("src.experiment.create_provider", mock_create_provider)

        result = run_single_problem(problem, config)

        assert result["pass_at_1"] is False
        assert result["test_cases_passed"] == 0


class TestConfigIntegration:
    """Test that real config files can be loaded and used."""

    def test_all_configs_load_and_validate(self):
        """All 4 experiment configs should load without error."""
        from src.config import load_config

        for name in ["baseline", "k5", "k1", "k2"]:
            cfg = load_config(f"configs/{name}.yaml")
            assert cfg.N == 5
            assert len(cfg.providers) == 5
            if name == "baseline":
                assert cfg.baseline is True
            else:
                assert cfg.baseline is False
                assert cfg.K > 0

    def test_dataset_cached_correctly(self):
        """Cached dataset should have exactly 10 problems."""
        from src.dataset import load_leetcode_dataset
        problems = load_leetcode_dataset(10, seed=42)
        assert len(problems) == 10
        for p in problems:
            assert "problem_id" in p
            assert "test" in p
            assert "entry_point" in p


class TestErrorHandling:
    """Test that errors are handled gracefully."""

    def test_missing_entry_point(self, monkeypatch):
        """Problem without entry_point should still produce a result dict."""
        from src.tests.mock_provider import MockProvider

        problem = {
            "problem_id": "bad-problem",
            "difficulty": "Unknown",
            "title": "Bad Problem",
            "description": "no tests",
            "constraints": "",
            "function_signature": "solve(self)",
            "starter_code": "",
            "entry_point": "",  # missing!
            "test": "",
        }

        config = _make_mock_config("test", N=1, baseline=True)
        p = MockProvider(name="agent-0")
        p.generate_responses = ["class Solution:\n    def solve(self):\n        return 1\n"]

        def mock_create_provider(pcfg, pools=None):
            return p
        monkeypatch.setattr("src.experiment.create_provider", mock_create_provider)

        result = run_single_problem(problem, config)
        assert result["pass_at_1"] is False
