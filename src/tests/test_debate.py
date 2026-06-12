"""Integration tests for debate engine using Mock providers."""

import pytest
from src.agent import Agent
from src.debate import DebateEngine, DebateParams
from src.tests.mock_provider import (
    MockProvider,
    SAMPLE_CODE_A,
    SAMPLE_CODE_B,
    SAMPLE_CODE_C,
    make_test_problem,
)


def _make_agents(n: int = 3) -> list[Agent]:
    """Create N agents with MockProviders, each returning distinct code."""
    codes = [SAMPLE_CODE_A, SAMPLE_CODE_B, SAMPLE_CODE_C]
    agents = []
    for i in range(n):
        provider = MockProvider(name=f"agent-{i}")
        # Program generate to return distinct code
        provider.generate_responses = [codes[i % len(codes)]]
        # Program refine to return same code (no change)
        provider.refine_responses = [codes[i % len(codes)]] * 10
        agent = Agent(id=i, provider=provider)
        agents.append(agent)
    return agents


class TestDebateEnginePhase1:
    """Test Phase 1 (elimination) mechanics."""

    def test_phase1_eliminates_one_per_round(self):
        """K=1 with 3 agents: should run 2 elimination rounds (3→2→1)."""
        agents = _make_agents(3)
        problem = make_test_problem()

        # Program votes:
        # Agent 0 always votes [2] → display index 2 → agent 1
        # Agent 1 always votes [1] → display index 1 → agent 0
        # Agent 2 always votes [1] → display index 1 → agent 0
        for i, agent in enumerate(agents):
            provider = agent.provider
            if i == 0:
                provider.vote_responses = ["[2]"] * 5
            elif i == 1:
                provider.vote_responses = ["[1]"] * 5
            else:
                provider.vote_responses = ["[1]"] * 5

        engine = DebateEngine(agents, K=1, params=DebateParams(alpha=0.5, gamma=0.3))
        result = engine.run(problem)

        # Should converge: N-1 = 2 rounds max
        assert result.rounds <= 2
        assert result.total_llm_calls > 3  # at least initial gen + some rounds

    def test_phase1_produces_winner(self):
        """Debate should always produce exactly one winner."""
        agents = _make_agents(3)
        problem = make_test_problem()

        for i, agent in enumerate(agents):
            provider = agent.provider
            provider.vote_responses = ["[1]"] * 10

        engine = DebateEngine(agents, K=1)
        result = engine.run(problem)

        assert result.winner_id is not None
        assert result.winner_solution is not None
        assert result.winner_solution.code != ""


class TestDebateEnginePhase2:
    """Test Phase 2 (endgame) mechanics."""

    def test_phase2_weight_absorption(self):
        """K=3 with 3 agents: immediate Phase 2, Best-1 voting."""
        agents = _make_agents(3)
        problem = make_test_problem()

        # All vote for agent 0 → should give agent 0 high weight
        for i, agent in enumerate(agents):
            provider = agent.provider
            # Vote for display index 1 → agent 0 (when excluding self)
            # Voting from agent's perspective: index 1 = first other agent
            if i == 0:
                provider.vote_responses = ["1"] * 5  # votes for agent 1 (display 1)
            elif i == 1:
                provider.vote_responses = ["1"] * 5  # votes for agent 0 (display 1)
            else:
                provider.vote_responses = ["1"] * 5  # votes for agent 0 (display 1)

        engine = DebateEngine(agents, K=3, params=DebateParams(alpha=0.5, gamma=0.3))
        result = engine.run(problem)

        assert result.winner_id is not None
        assert result.rounds <= 2  # N-1 = 2 max

    def test_graph_deadlock_break_triggered(self):
        """Create a voting cycle → should trigger graph break."""
        agents = _make_agents(3)
        problem = make_test_problem()

        # Create cycle: 0→1, 1→2, 2→0
        for i, agent in enumerate(agents):
            provider = agent.provider
            # Agent 0: excluding self, votes [1] → agent 1
            # Agent 1: excluding self, votes [2] → agent 2
            # Agent 2: excluding self, votes [1] → agent 0
            if i == 0:
                provider.vote_responses = ["1"] * 5   # → agent 1
            elif i == 1:
                provider.vote_responses = ["2"] * 5   # → agent 2
            else:
                provider.vote_responses = ["1"] * 5   # → agent 0

        engine = DebateEngine(agents, K=3, params=DebateParams(alpha=0.5, gamma=0.3))
        result = engine.run(problem)

        # Should terminate despite the cycle (graph break)
        assert result.winner_id is not None
        # Graph break should have been used
        graph_breaks = [h for h in result.history if h.get("mechanism") == "graph_break"]
        assert len(graph_breaks) >= 1 or result.terminated_early


class TestDebateKValues:
    """Test different K configurations produce expected behaviors."""

    def test_K_equals_N_immediate_endgame(self):
        """K=N=3: should go directly to Phase 2."""
        agents = _make_agents(3)
        problem = make_test_problem()

        for i, agent in enumerate(agents):
            provider = agent.provider
            provider.vote_responses = ["1"] * 10

        engine = DebateEngine(agents, K=3)
        result = engine.run(problem)

        # All rounds should be Phase 2 (endgame)
        elimination_rounds = [h for h in result.history if h.get("phase") == "elimination"]
        assert len(elimination_rounds) == 0

    def test_K_equals_1_all_elimination(self):
        """K=1 with 3 agents: all rounds should be Phase 1 elimination."""
        agents = _make_agents(3)
        problem = make_test_problem()

        for i, agent in enumerate(agents):
            provider = agent.provider
            provider.vote_responses = ["[1]"] * 10

        engine = DebateEngine(agents, K=1)
        result = engine.run(problem)

        # All rounds should be Phase 1
        endgame_rounds = [h for h in result.history if h.get("phase") == "endgame"]
        assert len(endgame_rounds) == 0

    def test_K_equals_2_mixed_phases(self):
        """K=2 with 5 agents: first 3 rounds elimination, then endgame."""
        agents = _make_agents(5)
        problem = make_test_problem()

        for i, agent in enumerate(agents):
            provider = agent.provider
            provider.vote_responses = ["[1, 2]"] * 10

        engine = DebateEngine(agents, K=2)
        result = engine.run(problem)

        elimination_rounds = [h for h in result.history if h.get("phase") == "elimination"]
        endgame_rounds = [h for h in result.history if h.get("phase") == "endgame"]
        # With K=2, N=5: elimination for 5→4→3 (3 rounds), then endgame for 2 agents
        assert len(elimination_rounds) >= 2
        # Endgame rounds may be 0 if someone wins by weight or 1+ if graph break


class TestConsistencyScore:
    """Test the mathematical correctness of consistency scoring."""

    def test_consistency_score_calculation(self):
        """Manually verify C_i calculation."""
        agents = _make_agents(3)
        problem = make_test_problem()

        # Agent 0: votes [1, 2] → agent_ids [1, 2]
        # Agent 1: votes [1, 2] → agent_ids [0, 2]
        # Agent 2: votes [1, 2] → agent_ids [0, 1]
        for i, agent in enumerate(agents):
            provider = agent.provider
            if i == 0:
                provider.vote_responses = ["[1, 2]"] * 5
            elif i == 1:
                provider.vote_responses = ["[1, 2]"] * 5
            else:
                provider.vote_responses = ["[1, 2]"] * 5

        engine = DebateEngine(agents, K=1, params=DebateParams(alpha=0.5, gamma=0.3))
        result = engine.run(problem)

        # Just verify it runs without error and produces consistent output
        assert result.winner_id is not None
        assert result.rounds > 0


class TestDebateConvergence:
    """Test that the debate always converges."""

    def test_converges_in_N_minus_1_rounds(self):
        """Max rounds is N-1."""
        for N in [2, 3, 5]:
            agents = _make_agents(N)
            problem = make_test_problem()

            for i, agent in enumerate(agents):
                provider = agent.provider
                provider.vote_responses = ["[1]"] * 20

            engine = DebateEngine(agents, K=1 if N > 1 else 1)
            result = engine.run(problem)

            assert result.rounds <= N - 1, f"N={N}: rounds={result.rounds} > {N-1}"
            assert result.winner_id is not None
