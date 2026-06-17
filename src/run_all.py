"""Batch experiment runner: execute all 4 experiment versions + generate summary."""

import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .evaluation import generate_summary_table
from .experiment import run_experiment


# Ordered list of experiment configs to run
EXPERIMENT_CONFIGS = [
    "configs/baseline.yaml",
    "configs/k5.yaml",
    "configs/k1.yaml",
    "configs/k2.yaml",
]


def run_all(max_problems: int | None = None) -> None:
    """Run all experiments sequentially and produce a comparison summary.

    Args:
        max_problems: If set, limit each experiment to this many problems
                      (for quick smoke tests).
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for config_path in EXPERIMENT_CONFIGS:
        cfg = load_config(config_path)
        print(f"\n{'='*60}")
        print(f"Running: {cfg.name}")
        print(f"N={cfg.N}, K={cfg.K}, baseline={cfg.baseline}")
        print(f"{'='*60}")

        results = run_experiment(cfg, max_problems=max_problems, output_dir=run_dir)
        all_results[cfg.name] = results

        # Per-experiment quick summary
        passed = sum(1 for r in results if r.get("pass_at_1"))
        total = len(results)
        print(f"\n>>> {cfg.name}: {passed}/{total} pass@1")

    # --- Generate final comparison table ---
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    table = generate_summary_table(all_results)
    print(table)

    # Write summary to run directory
    summary_path = run_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Agent Debate Framework — Experiment Results\n\n")
        f.write(table)
        f.write("\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all debate experiments.")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit problems per experiment (for smoke testing).")
    args = parser.parse_args()
    run_all(max_problems=args.max_problems)
