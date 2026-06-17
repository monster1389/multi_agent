"""LeetCode dataset loading from HuggingFace, with local caching."""

import json
import os
import random
from pathlib import Path
from typing import Any

# Use HF mirror by default for faster downloads inside GFW
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset


_HF_SOURCE = "newfacade/LeetCodeDataset"
_CACHE_DIR = Path("data")
_CACHE_FILE = _CACHE_DIR / "leetcode_10.jsonl"


def load_leetcode_dataset(
    subset_size: int = 10,
    seed: int = 42,
    cache_dir: str = "data",
) -> list[dict[str, Any]]:
    """Load a random subset of LeetCode Python problems from HuggingFace.

    Downloads the dataset on first call, then caches the selected subset
    as JSON Lines for fast subsequent loads.

    Args:
        subset_size: Number of problems to sample.
        seed: Random seed for reproducible sampling.
        cache_dir: Directory for local cache.

    Returns:
        List of problem dicts with keys:
            problem_id, difficulty, description, starter_code,
            entry_point, test, input_output
    """
    cache_path = Path(cache_dir) / f"leetcode_{subset_size}.jsonl"

    # Return cached if available
    if cache_path.exists():
        problems = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        if len(problems) == subset_size:
            return problems

    # Load from HuggingFace
    ds = load_dataset(_HF_SOURCE, split="train")

    # Shuffle and sample
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:subset_size]

    problems = []
    for idx in selected:
        row = ds[int(idx)]
        problem = {
            "problem_id": row["task_id"],
            "difficulty": row["difficulty"],
            "description": row["problem_description"],
            "starter_code": row["starter_code"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "input_output": row["input_output"],
            "constraints": "",  # Not directly provided; can be extracted if needed
        }
        problems.append(problem)

    # Write cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return problems


def get_function_signature(problem: dict) -> str:
    """Extract the function signature from starter_code.

    Example: 'def twoSum(self, nums: List[int], target: int) -> List[int]:'
    """
    starter = problem.get("starter_code", "")
    for line in starter.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped.rstrip(":")
    return "solution(self, ...)"
