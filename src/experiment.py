"""Experiment orchestration: run a single experiment version across all problems."""

import json
import sys
from pathlib import Path
from typing import Any

from .agent import Agent, generate_initial_solution
from .config import ExperimentConfig, load_config
from .dataset import load_leetcode_dataset, get_function_signature
from .debate import DebateEngine, DebateParams
from .providers import create_provider
from .sandbox import run_all_tests


def run_single_problem(
    problem: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run one problem through the debate framework (or baseline).

    Args:
        problem: Problem dict from dataset.
        config: Experiment configuration.

    Returns:
        Result dict (one JSONL line).
    """
    # Create agents with their providers
    agents = []
    for i, pcfg in enumerate(config.providers):
        provider = create_provider(pcfg)
        agent = Agent(id=i, provider=provider)
        agents.append(agent)

    total_llm_calls = 0
    debate_rounds = 0
    winner_code = ""
    winner_provider = ""
    winner_id = -1

    if config.baseline:
        # --- BASELINE: generate once, pick best by test pass rate ---
        solutions = []
        round0_agents: dict[str, dict] = {}
        for agent in agents:
            sol, prompt, response = generate_initial_solution(agent, problem)
            total_llm_calls += 1
            solutions.append((agent, sol))
            round0_agents[str(agent.id)] = {
                "provider": agent.provider.model_name,
                "code": sol.code,
                "prompt": prompt,
                "response": response,
            }

        # Test all solutions, pick best
        best_agent = None
        best_code = ""
        best_passed = -1
        test_results: dict[str, dict] = {}
        for agent, sol in solutions:
            passed, total, _ = run_all_tests(sol.code, problem)
            test_results[str(agent.id)] = {"passed": passed, "total": total}
            if passed > best_passed:
                best_passed = passed
                best_agent = agent
                best_code = sol.code

        debate_rounds = 1
        if best_agent:
            winner_code = best_code
            winner_provider = best_agent.provider.model_name
            winner_id = best_agent.id

        # Build Baseline transcript
        transcript = [
            {"round": 0, "phase": "initial", "agents": round0_agents},
            {
                "round": "result", "phase": "selection",
                "test_results": test_results,
                "selected": {
                    "agent_id": best_agent.id if best_agent else -1,
                    "reason": "highest_pass_rate",
                },
            },
        ]
    else:
        # --- DEBATE MODE: full multi-agent debate ---
        params = DebateParams(
            alpha=config.debate_params.alpha,
            gamma=config.debate_params.gamma,
        )
        engine = DebateEngine(agents, K=config.K, params=params)
        result = engine.run(problem)

        total_llm_calls = result.total_llm_calls
        debate_rounds = result.rounds
        winner_code = result.winner_solution.code
        winner_provider = next(
            (a.provider.model_name for a in agents if a.id == result.winner_id),
            "unknown",
        )
        winner_id = result.winner_id
        transcript = result.transcript

    # --- Sandbox verification ---
    passed, total, output = run_all_tests(winner_code, problem)
    pass_at_1 = (passed == total and total > 0)

    # --- Write transcript ---
    safe_exp = _safe_name(config.name)
    transcript_dir = Path("results/transcripts") / safe_exp
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{problem.get('problem_id', 'unknown')}.json"
    full_transcript = {
        "problem_id": problem.get("problem_id", ""),
        "difficulty": problem.get("difficulty", ""),
        "experiment": config.name,
        "N": config.N,
        "K": config.K if not config.baseline else 0,
        "winner_id": winner_id,
        "pass_at_1": pass_at_1,
        "rounds": transcript,
    }
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(full_transcript, f, ensure_ascii=False, indent=2)

    return {
        "problem_id": problem.get("problem_id", ""),
        "difficulty": problem.get("difficulty", ""),
        "experiment": config.name,
        "N": config.N,
        "K": config.K if not config.baseline else 0,
        "baseline": config.baseline,
        "pass_at_1": pass_at_1,
        "test_cases_passed": passed,
        "test_cases_total": total,
        "debate_rounds": debate_rounds,
        "total_llm_calls": total_llm_calls,
        "winner_provider": winner_provider,
        "solution_code": winner_code,
        "transcript_path": str(transcript_path),
    }


def run_experiment(config: ExperimentConfig, max_problems: int | None = None) -> list[dict[str, Any]]:
    """Run an experiment across all (or a subset of) problems.

    Args:
        config: Experiment configuration.
        max_problems: Limit problems (for quick testing). None = all.

    Returns:
        List of per-problem result dicts.
    """
    problems = load_leetcode_dataset(
        subset_size=config.dataset.subset_size,
        seed=config.dataset.random_seed,
    )

    if max_problems:
        problems = problems[:max_problems]

    results = []
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"{_safe_name(config.name)}.jsonl"

    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem['problem_id']} ({problem['difficulty']}) ...", end=" ", flush=True)
        try:
            result = run_single_problem(problem, config)
            results.append(result)
            status = "PASS" if result["pass_at_1"] else "FAIL"
            print(f"{status} | rounds={result['debate_rounds']} calls={result['total_llm_calls']}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "problem_id": problem.get("problem_id", ""),
                "difficulty": problem.get("difficulty", ""),
                "experiment": config.name,
                "pass_at_1": False,
                "error": str(e),
            })

        # Write incrementally
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def _safe_name(name: str) -> str:
    """Convert experiment name to safe filename."""
    return name.lower().replace(" ", "_").replace("=", "")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a debate experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems (for testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Experiment: {cfg.name} | N={cfg.N} K={cfg.K} baseline={cfg.baseline}")
    print(f"Problems: {cfg.dataset.subset_size} | Seed: {cfg.dataset.random_seed}")
    print("---")

    results = run_experiment(cfg, max_problems=args.max_problems)

    # Quick summary
    from .evaluation import summarize_experiment
    summary = summarize_experiment(results)
    print("---")
    print(f"Pass@1: {summary['pass_at_1']:.2f} | "
          f"Avg Rounds: {summary['avg_rounds']:.1f} | "
          f"Avg LLM Calls: {summary['avg_llm_calls']:.1f}")
