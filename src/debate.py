"""Debate engine — core two-phase multi-agent debate algorithm.

Implements the algorithm from 多Agent协商框架-final.pdf:
  Phase 1: Monotonic elimination (N_alive > K)
    Top-K voting → weighted scoring → consistency → weight update → eliminate 1
  Phase 2: Endgame single-vote collapse (N_alive ≤ K)
    Best-1 voting → weight absorption → termination check → graph deadlock break

Guaranteed to converge in at most N-1 rounds.
"""

from dataclasses import dataclass, field

from .agent import Agent, Solution, generate_initial_solution, refine_solution, vote
from .graph_utils import find_cycles


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DebateResult:
    """Result of a single debate run."""
    winner_id: int
    winner_solution: Solution
    rounds: int                          # debate rounds (not counting initial gen)
    total_llm_calls: int                 # includes initial generation
    transcript: list[dict] = field(default_factory=list)  # full interaction log
    history: list[dict] = field(default_factory=list)
    terminated_early: bool = False       # True if convergence before N-1 rounds


@dataclass
class DebateParams:
    """Debate hyperparameters."""
    alpha: float = 0.5    # weight update learning rate
    gamma: float = 0.3    # endgame weight absorption rate


# ---------------------------------------------------------------------------
# Debate Engine
# ---------------------------------------------------------------------------

class DebateEngine:
    """Two-phase multi-agent debate engine.

    Usage:
        engine = DebateEngine(agents, K=2, params=DebateParams())
        result = engine.run(problem)
    """

    def __init__(
        self,
        agents: list[Agent],
        K: int,
        params: DebateParams | None = None,
    ):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if len(agents) < 2:
            raise ValueError(f"Need at least 2 agents, got {len(agents)}")

        self.agents = agents
        self.K = K
        self.params = params or DebateParams()
        self._llm_calls = 0
        self._transcript: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, problem: dict) -> DebateResult:
        """Run the full debate for one problem.

        Args:
            problem: Dict with 'title', 'description', 'starter_code', etc.

        Returns:
            DebateResult with winner and statistics.
        """
        N = len(self.agents)
        self._llm_calls = 0
        history: list[dict] = []

        # --- Step 1: Initial solution generation (N LLM calls) ---
        round0_agents: dict[str, dict] = {}
        for agent in self.agents:
            agent.weight = 1.0 / N
            agent.cumulative_gamma = 0.0
            agent.alive = True
            sol, prompt, response = generate_initial_solution(agent, problem)
            round0_agents[str(agent.id)] = {
                "provider": agent.provider.model_name,
                "code": sol.code,
                "prompt": prompt,
                "response": response,
            }
            self._llm_calls += 1
        self._transcript.append({"round": 0, "phase": "initial", "agents": round0_agents})

        round_num = 0
        N_alive = N

        # --- Main debate loop ---
        while True:
            round_num += 1
            alive = self._alive_agents()

            # PHASE DETERMINATION
            if N_alive > self.K:
                phase = "elimination"
                self._phase1_round(alive, problem, round_num, history)
            else:
                phase = "endgame"
                terminated = self._phase2_round(alive, problem, round_num, history)
                if terminated:
                    break

            # Check N_alive == 1 after elimination round
            N_alive = len(self._alive_agents())
            if N_alive == 1:
                break

        winner = self._alive_agents()[0]
        return DebateResult(
            winner_id=winner.id,
            winner_solution=winner.current_solution,  # type: ignore[arg-type]
            rounds=round_num,
            total_llm_calls=self._llm_calls,
            transcript=self._transcript,
            history=history,
            terminated_early=round_num < N - 1,
        )

    # ------------------------------------------------------------------
    # Phase 1: Elimination
    # ------------------------------------------------------------------

    def _phase1_round(
        self,
        alive: list[Agent],
        problem: dict,
        round_num: int,
        history: list[dict],
    ) -> None:
        """One round of Phase 1: refine → vote Top-K → score → eliminate."""
        # --- 1. Refine: each agent improves its solution ---
        solutions = {a.id: a.current_solution for a in alive if a.current_solution}
        weights_before = {str(a.id): round(a.weight, 4) for a in alive}
        refine_data: dict[str, dict] = {}
        for agent in alive:
            sol, prompt, response = refine_solution(agent, problem, solutions)
            refine_data[str(agent.id)] = {
                "code": sol.code, "prompt": prompt, "response": response
            }
            self._llm_calls += 1

        # --- 2. Vote Top-K ---
        solutions = {a.id: a.current_solution for a in alive if a.current_solution}
        votes: dict[int, list[int]] = {}
        vote_data: dict[str, dict] = {}
        for agent in alive:
            voted, prompt, response = vote(agent, problem, solutions, self.K, my_index=agent.id)
            votes[agent.id] = voted
            vote_data[str(agent.id)] = {
                "voted_for": voted, "raw_response": response,
                "prompt": prompt, "response": response,
            }
            self._llm_calls += 1

        # --- 3. Weighted scoring ---
        scores = self._compute_weighted_scores(alive, votes)

        # --- 4. Global ranking S_global (top-K by score) ---
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        S_global_ids = [sid for sid, _ in sorted_by_score[:self.K]]

        # --- 5. Consistency scores & weight update ---
        consistency = {}
        for agent in alive:
            C = self._consistency_score(agent.id, votes.get(agent.id, []), S_global_ids)
            consistency[agent.id] = C
            agent.cumulative_gamma += C

        self._update_weights_phase1(alive, consistency)
        weights_after = {str(a.id): round(a.weight, 4) for a in alive}

        # --- 6. Deterministic elimination ---
        eliminated = self._eliminate_lowest_weight(alive)
        eliminated.alive = False

        self._transcript.append({
            "round": round_num,
            "phase": "elimination",
            "refine": refine_data,
            "vote": vote_data,
            "scores": {str(k): round(v, 4) for k, v in scores.items()},
            "consistency": {str(k): round(v, 4) for k, v in consistency.items()},
            "weights_before": weights_before,
            "weights_after": weights_after,
            "eliminated": {"agent_id": eliminated.id, "reason": "lowest_weight"},
        })

        history.append({
            "round": round_num,
            "phase": "elimination",
            "alive_before": len(alive),
            "eliminated": eliminated.id,
            "weights": {a.id: round(a.weight, 4) for a in alive},
            "llm_calls_so_far": self._llm_calls,
        })

    # ------------------------------------------------------------------
    # Phase 2: Endgame
    # ------------------------------------------------------------------

    def _phase2_round(
        self,
        alive: list[Agent],
        problem: dict,
        round_num: int,
        history: list[dict],
    ) -> bool:
        """One round of Phase 2: refine → vote Best-1 → absorb → check → break.

        Returns True if debate terminated (winner found).
        """
        # --- 1. Refine ---
        solutions = {a.id: a.current_solution for a in alive if a.current_solution}
        refine_data: dict[str, dict] = {}
        for agent in alive:
            sol, prompt, response = refine_solution(agent, problem, solutions)
            refine_data[str(agent.id)] = {
                "code": sol.code, "prompt": prompt, "response": response
            }
            self._llm_calls += 1

        # --- 2. Vote Best-1 ---
        solutions = {a.id: a.current_solution for a in alive if a.current_solution}
        best_votes: dict[int, int] = {}
        vote_data: dict[str, dict] = {}
        for agent in alive:
            voted_list, prompt, response = vote(agent, problem, solutions, K=1, my_index=agent.id)
            if voted_list:
                best_votes[agent.id] = voted_list[0]
            vote_data[str(agent.id)] = {
                "voted_for": voted_list, "raw_response": response,
                "prompt": prompt, "response": response,
            }
            self._llm_calls += 1

        # --- 3. Weight absorption ---
        self._weight_absorption(alive, best_votes)
        weights_absorbed = {str(a.id): round(a.weight, 4) for a in alive}

        # --- 4. Termination check: any agent > 50%? ---
        for agent in alive:
            if agent.weight > 0.5:
                self._transcript.append({
                    "round": round_num,
                    "phase": "endgame",
                    "refine": refine_data,
                    "vote": vote_data,
                    "best_votes": {str(k): v for k, v in best_votes.items()},
                    "weights_after_absorption": weights_absorbed,
                    "terminated": True,
                    "winner": {"agent_id": agent.id, "mechanism": "weight_threshold"},
                })
                history.append({
                    "round": round_num,
                    "phase": "endgame",
                    "winner": agent.id,
                    "winner_weight": agent.weight,
                    "mechanism": "weight_threshold",
                    "llm_calls_so_far": self._llm_calls,
                })
                return True

        # --- 5. Graph deadlock break ---
        if len(alive) == 1:
            self._transcript.append({
                "round": round_num,
                "phase": "endgame",
                "refine": refine_data,
                "vote": vote_data,
                "best_votes": {str(k): v for k, v in best_votes.items()},
                "weights_after_absorption": weights_absorbed,
                "terminated": True,
                "winner": {"agent_id": alive[0].id, "mechanism": "last_standing"},
            })
            return True

        eliminated_id = self._graph_deadlock_break(alive, best_votes)
        for agent in alive:
            if agent.id == eliminated_id:
                agent.alive = False
                break

        self._transcript.append({
            "round": round_num,
            "phase": "endgame",
            "refine": refine_data,
            "vote": vote_data,
            "best_votes": {str(k): v for k, v in best_votes.items()},
            "weights_after_absorption": weights_absorbed,
            "terminated": False,
            "eliminated": {"agent_id": eliminated_id, "reason": "graph_break"},
        })

        history.append({
            "round": round_num,
            "phase": "endgame",
            "alive_before": len(alive) + 1,  # +1 because we just eliminated
            "eliminated": eliminated_id,
            "mechanism": "graph_break",
            "weights": {a.id: round(a.weight, 4) for a in alive if a.alive},
            "llm_calls_so_far": self._llm_calls,
        })

        return False  # not terminated

    # ------------------------------------------------------------------
    # Mathematical helpers
    # ------------------------------------------------------------------

    def _compute_weighted_scores(
        self,
        alive: list[Agent],
        votes: dict[int, list[int]],
    ) -> dict[int, float]:
        """Compute weighted score for each solution.

        Score(x) = Σ W_i × (1 / Pos(x, S_i)) × I(x ∈ S_i)
        """
        scores: dict[int, float] = {}
        for agent in alive:
            S_i = votes.get(agent.id, [])
            for rank, voted_id in enumerate(S_i, start=1):
                score_contrib = agent.weight * (1.0 / rank)
                scores[voted_id] = scores.get(voted_id, 0.0) + score_contrib
        return scores

    def _consistency_score(
        self,
        agent_id: int,
        S_i: list[int],
        S_global: list[int],
    ) -> float:
        """Compute single-round consistency score for one agent.

        C_i = Σ_{x∈(S_i∩S_global)} (1/Pos(x,S_i) + 1/Pos(x,S_global))
        """
        total = 0.0
        for x in set(S_i) & set(S_global):
            pos_i = S_i.index(x) + 1      # 1-indexed
            pos_g = S_global.index(x) + 1  # 1-indexed
            total += (1.0 / pos_i) + (1.0 / pos_g)
        return total

    def _update_weights_phase1(
        self,
        alive: list[Agent],
        consistency: dict[int, float],
    ) -> None:
        """Update weights: W_i = W_i × (1 + α·C_i), then normalize."""
        alpha = self.params.alpha
        for agent in alive:
            C = consistency.get(agent.id, 0.0)
            agent.weight *= (1.0 + alpha * C)

        # Normalize
        total_w = sum(a.weight for a in alive)
        if total_w > 0:
            for agent in alive:
                agent.weight /= total_w

    def _eliminate_lowest_weight(self, alive: list[Agent]) -> Agent:
        """Deterministic elimination: lowest weight → lowest Γ → highest ID."""
        def sort_key(a: Agent) -> tuple[float, float, int]:
            # We want LOWEST weight first, so use positive weight
            # Then LOWEST gamma, then HIGHEST id (reverse)
            return (a.weight, a.cumulative_gamma, -a.id)

        sorted_agents = sorted(alive, key=sort_key)
        return sorted_agents[0]  # lowest weight (ties broken as above)

    def _weight_absorption(
        self,
        alive: list[Agent],
        best_votes: dict[int, int],
    ) -> None:
        """Weight absorption: W_j^* = W_j + γ·Σ_{i∈V_j} W_i, then normalize.

        V_j = set of agents who voted for j.
        """
        gamma = self.params.gamma

        # Group voters by their vote target
        voters_for: dict[int, list[Agent]] = {}
        for voter_id, voted_id in best_votes.items():
            if voted_id not in voters_for:
                voters_for[voted_id] = []
            voter = next((a for a in alive if a.id == voter_id), None)
            if voter:
                voters_for[voted_id].append(voter)

        # Apply absorption
        for agent in alive:
            voters = voters_for.get(agent.id, [])
            transfer = gamma * sum(v.weight for v in voters)
            agent.weight += transfer

        # Normalize
        total_w = sum(a.weight for a in alive)
        if total_w > 0:
            for agent in alive:
                agent.weight /= total_w

    def _graph_deadlock_break(
        self,
        alive: list[Agent],
        best_votes: dict[int, int],
    ) -> int:
        """Find cycles → eliminate cycle agent with lowest Γ.

        Returns the agent_id to eliminate.
        """
        alive_ids = {a.id for a in alive}

        # Build adjacency for alive agents only
        adjacency: dict[int, int] = {}
        for vid, target in best_votes.items():
            if vid in alive_ids and target in alive_ids:
                adjacency[vid] = target

        cycles = find_cycles(adjacency)

        if not cycles:
            # Degenerate case: no cycles found (shouldn't happen in
            # a proper functional graph, but handle gracefully).
            # Eliminate the agent with lowest weight.
            return self._eliminate_lowest_weight(alive).id

        # Collect all agents in any cycle
        cycle_agents: set[int] = set()
        for cycle in cycles:
            cycle_agents |= cycle

        # Eliminate the cycle agent with lowest cumulative Γ
        # Tie-break: highest initial ID
        def sort_key(aid: int) -> tuple[float, int]:
            agent = next(a for a in alive if a.id == aid)
            return (agent.cumulative_gamma, -aid)

        sorted_cycle = sorted(cycle_agents, key=sort_key)
        return sorted_cycle[0]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _alive_agents(self) -> list[Agent]:
        """Return list of currently alive agents."""
        return [a for a in self.agents if a.alive]
