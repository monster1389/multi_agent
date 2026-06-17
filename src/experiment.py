"""Experiment orchestration: run a single experiment version across all problems."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one problem through the debate framework (or baseline).

    Args:
        problem: Problem dict from dataset.
        config: Experiment configuration.
        output_dir: Run directory for transcript output. Required.

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
        from .debate import _run_parallel
        solutions = []
        round0_agents: dict[str, dict] = {}
        gen_results = _run_parallel([
            (generate_initial_solution, (agent, problem)) for agent in agents
        ])
        total_llm_calls += len(agents)
        for agent in agents:
            sol, prompt, response = gen_results[agent.id]
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
    if output_dir is None:
        output_dir = _make_run_dir()
    safe_exp = _safe_name(config.name)
    transcript_dir = output_dir / "transcripts" / safe_exp
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


def _make_run_dir(output_dir: Path | None = None) -> Path:
    """Resolve the output directory for this run.

    Args:
        output_dir: Explicit directory (used by run_all).
                    If None, auto-generate from current timestamp.

    Returns:
        Resolved Path to the run directory (created if needed).
    """
    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(
    config: ExperimentConfig,
    max_problems: int | None = None,
    output_dir: Path | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run an experiment across all (or a subset of) problems.

    Args:
        config: Experiment configuration.
        max_problems: Limit problems (for quick testing). None = all.
        output_dir: Optional run directory. If None, auto-generated timestamp.
        workers: Number of problems to run concurrently (default: 1 = serial).

    Returns:
        List of per-problem result dicts.
    """
    problems = load_leetcode_dataset(
        subset_size=config.dataset.subset_size,
        seed=config.dataset.random_seed,
    )

    if max_problems:
        problems = problems[:max_problems]

    run_dir = _make_run_dir(output_dir)
    results = []
    output_path = run_dir / f"{_safe_name(config.name)}.jsonl"

    if workers <= 1:
        for i, problem in enumerate(problems):
            print(f"[{i+1}/{len(problems)}] {problem['problem_id']} ({problem['difficulty']}) ...", end=" ", flush=True)
            try:
                result = run_single_problem(problem, config, output_dir=run_dir)
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
    else:
        write_lock = threading.Lock()
        results_by_idx: dict[int, dict[str, Any]] = {}

        def _run_one(idx: int, problem: dict[str, Any]) -> None:
            pid = problem.get("problem_id", "unknown")
            with write_lock:
                print(f"[{idx+1}/{len(problems)}] {pid} ({problem.get('difficulty', '?')}) ...", end=" ", flush=True)
            try:
                result = run_single_problem(problem, config, output_dir=run_dir)
                status = "PASS" if result["pass_at_1"] else "FAIL"
                with write_lock:
                    print(f"{status} | rounds={result['debate_rounds']} calls={result['total_llm_calls']}")
                    results_by_idx[idx] = result
            except Exception as e:
                with write_lock:
                    print(f"ERROR: {e}")
                    results_by_idx[idx] = {
                        "problem_id": problem.get("problem_id", ""),
                        "difficulty": problem.get("difficulty", ""),
                        "experiment": config.name,
                        "pass_at_1": False,
                        "error": str(e),
                    }
            # Incremental write under lock
            with write_lock:
                ordered = [results_by_idx[k] for k in sorted(results_by_idx)]
                with open(output_path, "w", encoding="utf-8") as f:
                    for r in ordered:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, i, p) for i, p in enumerate(problems)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    import traceback
                    traceback.print_exc()  # unexpected infrastructure failure

        results = [results_by_idx[k] for k in sorted(results_by_idx)]

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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of problems to run concurrently (default: 1 = serial).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    cfg = load_config(args.config)
    print(f"Experiment: {cfg.name} | N={cfg.N} K={cfg.K} baseline={cfg.baseline}")
    print(f"Problems: {cfg.dataset.subset_size} | Seed: {cfg.dataset.random_seed}")
    print("---")

    results = run_experiment(cfg, max_problems=args.max_problems, workers=args.workers)

    # Quick summary
    from .evaluation import summarize_experiment
    summary = summarize_experiment(results)
    print("---")
    print(f"Pass@1: {summary['pass_at_1']:.2f} | "
          f"Avg Rounds: {summary['avg_rounds']:.1f} | "
          f"Avg LLM Calls: {summary['avg_llm_calls']:.1f}")
