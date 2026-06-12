## Stack

- **Language:** Python ≥3.12
- **LLM clients:** `openai` ≥1.0, `anthropic` ≥0.39
- **Data:** `datasets` (HuggingFace) for LeetCode problems, `pyyaml` for config
- **Env:** `python-dotenv` loads `.env` automatically in `load_config()`
- **Test:** `pytest` ≥9.0 — no other lint/format/typecheck tools configured

## Layout

```
configs/        YAML experiment configs (baseline, k1, k2, k5)
data/           LeetCode problem JSONL subsets
docs/           多Agent协商框架-final.pdf (algorithm reference)
results/        Generated output — gitignored; transcripts/ + .jsonl + summary.md
src/            All source code (flat package)
  providers/    LLM provider layer: base, openai_provider, anthropic_provider
  tests/        Tests colocated under src/ (mock_provider, test_debate, etc.)
```

## Commands

```bash
# Run a single experiment
python -m src.experiment --config configs/baseline.yaml

# Run all 4 experiments + summary table
python -m src.run_all

# Limit problems for smoke testing
python -m src.experiment --config configs/k1.yaml --max-problems 3
python -m src.run_all --max-problems 3

# Run tests
pytest
```

## Conventions

- **Colocated tests** under `src/tests/` — not a top-level `tests/` dir.
- **Google-style docstrings** throughout: `Args:` / `Returns:` / `Raises:`.
- **Type hints** on all public signatures; `dict[str, Any]` for unstructured data.
- **Dataclasses** for config (`ExperimentConfig`), debate params (`DebateParams`), and results (`DebateResult`).
- **Relative imports** inside `src/` subpackages: `from ..config import ...`.
- **MockProvider** in `src/tests/mock_provider.py` enables deterministic integration tests without API calls.
- **No lint/format config** — no ruff, black, flake8, or mypy settings exist.

## Watch out for

- **`.env` is gitignored.** Copy `.env.example`, fill `DASHSCOPE_API_KEY` and `DEEPSEEK_API_KEY`. `load_config()` calls `load_dotenv()` automatically.
- **`results/` is generated.** Gitignored; contains JSONL results, per-problem transcripts, and `summary.md`. Never edit by hand.
- **Config `${VAR}` substitution** happens at YAML load time via regex — env must be set before `load_config()`.
- **Provider count must equal N.** The validator in `config.py` enforces `len(providers) == N`; a mismatch raises `ConfigError`.
- **K restrictions:** for debate mode, `1 ≤ K ≤ N`; for baseline (`baseline: true`), K is ignored.
