"""Evaluation metrics and result summarization."""

from typing import Any


def pass_at_1(results: list[dict[str, Any]]) -> float:
    """Pass@1: proportion of problems where all tests passed.

    Args:
        results: List of per-problem result dicts, each with 'pass_at_1' bool.

    Returns:
        Float in [0, 1].
    """
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("pass_at_1", False))
    return passed / len(results)


def avg_debate_rounds(results: list[dict[str, Any]]) -> float:
    """Average number of debate rounds across problems."""
    if not results:
        return 0.0
    rounds = [r.get("debate_rounds", 0) for r in results]
    return sum(rounds) / len(rounds)


def avg_llm_calls(results: list[dict[str, Any]]) -> float:
    """Average total LLM calls per problem."""
    if not results:
        return 0.0
    calls = [r.get("total_llm_calls", 0) for r in results]
    return sum(calls) / len(calls)


def summarize_experiment(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for a single experiment.

    Returns dict with keys: pass_at_1, avg_rounds, avg_llm_calls, total_problems.
    """
    return {
        "pass_at_1": pass_at_1(results),
        "avg_rounds": avg_debate_rounds(results),
        "avg_llm_calls": avg_llm_calls(results),
        "total_problems": len(results),
    }


def generate_summary_table(
    all_results: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate a Markdown comparison table across experiments.

    Args:
        all_results: Mapping experiment_name → list of per-problem results.

    Returns:
        Markdown table string.
    """
    header = "| Experiment | Pass@1 | Avg Rounds | Avg LLM Calls | Problems |"
    sep =    "|------------|--------|------------|---------------|----------|"

    rows = [header, sep]
    for name, results in all_results.items():
        s = summarize_experiment(results)
        rows.append(
            f"| {name} | {s['pass_at_1']:.2f} | {s['avg_rounds']:.1f} "
            f"| {s['avg_llm_calls']:.1f} | {s['total_problems']} |"
        )

    return "\n".join(rows)
