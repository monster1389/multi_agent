"""Configuration management: YAML loading, env var substitution, validation."""

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DebateParams:
    """Debate framework hyperparameters (from PDF)."""
    alpha: float = 0.5   # weight update learning rate
    gamma: float = 0.3   # endgame weight absorption rate
    lambda_coef: int = 2  # LLM calls per agent per round (refine + vote)


@dataclass
class ProviderConfig:
    """Single LLM provider configuration."""
    name: str           # display name e.g. "gpt-4o"
    type: str           # "openai" | "anthropic"
    model: str          # model id e.g. "gpt-4o"
    api_key: str = ""   # resolved from env
    base_url: str = ""  # optional, for OpenAI-compatible endpoints


@dataclass
class DatasetConfig:
    """Dataset loading configuration."""
    source: str = "newfacade/LeetCodeDataset"
    subset_size: int = 10
    random_seed: int = 42


class ModelPool:
    """Thread-safe model pool for automatic fallback on quota exhaustion.

    Agents allocate models via pop; on 403, replace() swaps in a new one.
    """

    def __init__(self, models: list[str]):
        if not models:
            raise ValueError("ModelPool requires at least one model")
        self._models: list[str] = list(models)
        self._lock = threading.Lock()

    def allocate(self) -> str:
        """Pop and return a model from the pool. Raises RuntimeError if empty."""
        with self._lock:
            if not self._models:
                raise RuntimeError("Model pool exhausted — all models returned 403")
            return self._models.pop(0)

    def replace(self, failed_model: str) -> str:
        """Remove failed_model if still in pool, then return the next available model."""
        with self._lock:
            if failed_model in self._models:
                self._models.remove(failed_model)
            if not self._models:
                raise RuntimeError("Model pool exhausted — all models returned 403")
            return self._models.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    N: int                          # initial agent count
    K: int                          # core threshold (0 = baseline)
    baseline: bool = False          # True → skip debate, pick best of N
    providers: list[ProviderConfig] = field(default_factory=list)
    debate_params: DebateParams = field(default_factory=DebateParams)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    pools: dict[str, "ModelPool"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r'\$\{(\w+)\}')


def _resolve_env(value: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""
    def _replacer(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_VAR_RE.sub(_replacer, value)


def _parse_providers(raw: list[dict]) -> list[ProviderConfig]:
    """Parse provider list from YAML, resolving env vars."""
    providers = []
    for item in raw:
        providers.append(ProviderConfig(
            name=item["name"],
            type=item["type"],
            model=item["model"],
            api_key=_resolve_env(item.get("api_key", "")),
            base_url=item.get("base_url", ""),
        ))
    return providers


def _parse_debate_params(raw: dict | None) -> DebateParams:
    """Parse debate params, using defaults for missing keys."""
    if raw is None:
        return DebateParams()
    return DebateParams(
        alpha=float(raw.get("alpha", 0.5)),
        gamma=float(raw.get("gamma", 0.3)),
        lambda_coef=int(raw.get("lambda_coef", 2)),
    )


def _parse_dataset(raw: dict | None) -> DatasetConfig:
    """Parse dataset config, using defaults for missing keys."""
    if raw is None:
        return DatasetConfig()
    return DatasetConfig(
        source=raw.get("source", "newfacade/LeetCodeDataset"),
        subset_size=int(raw.get("subset_size", 10)),
        random_seed=int(raw.get("random_seed", 42)),
    )


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Automatically calls load_dotenv() so ${VAR} placeholders resolve.
    """
    # Ensure .env is loaded (idempotent)
    load_dotenv()

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    experiment = raw.get("experiment", raw)
    name = experiment["name"]
    N = int(experiment.get("N", 5))
    baseline = bool(experiment.get("baseline", False))
    K = int(experiment.get("K", 0))

    providers = _parse_providers(raw.get("providers", []))
    debate_params = _parse_debate_params(raw.get("debate_params"))
    dataset = _parse_dataset(raw.get("dataset"))

    # Parse model pools
    pools: dict[str, ModelPool] = {}
    raw_pools = raw.get("model_pool", {}) or {}
    for pool_name, model_list in raw_pools.items():
        if not isinstance(model_list, list):
            raise ConfigError(
                f"model_pool.{pool_name} must be a list of model names, got {type(model_list).__name__}"
            )
        pools[pool_name] = ModelPool(model_list)

    config = ExperimentConfig(
        name=name,
        N=N,
        K=K,
        baseline=baseline,
        providers=providers,
        debate_params=debate_params,
        dataset=dataset,
        pools=pools,
    )

    validate_config(config)
    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Configuration validation error."""
    pass


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration consistency.

    Raises ConfigError on invalid settings.
    """
    if config.N < 2:
        raise ConfigError(f"N must be >= 2, got {config.N}")

    if not config.baseline:
        if config.K < 1:
            raise ConfigError(f"K must be >= 1 for debate mode, got {config.K}")
        if config.K > config.N:
            raise ConfigError(f"K ({config.K}) must be <= N ({config.N})")

    if len(config.providers) != config.N:
        raise ConfigError(
            f"Number of providers ({len(config.providers)}) "
            f"must equal N ({config.N})"
        )

    valid_types = {"openai", "anthropic"}
    for i, p in enumerate(config.providers):
        if p.type not in valid_types:
            raise ConfigError(
                f"Provider {i} type '{p.type}' not in {valid_types}"
            )
        if not p.api_key:
            raise ConfigError(
                f"Provider {i} ('{p.name}') has empty api_key — "
                f"check .env file and ${'{' + p.name.upper().replace('-','_') + '_API_KEY}'} "
                f"placeholder"
            )

    # Validate pool sizes
    if config.pools:
        pool_usage: dict[str, int] = {}
        for p in config.providers:
            if p.model.startswith("pool:"):
                pool_name = p.model.split(":", 1)[1]
                pool_usage[pool_name] = pool_usage.get(pool_name, 0) + 1

        for pool_name, count in pool_usage.items():
            pool = config.pools.get(pool_name)
            if pool is None:
                raise ConfigError(
                    f"Provider references pool '{pool_name}' but no such pool defined"
                )
            if count > len(pool):
                raise ConfigError(
                    f"Pool '{pool_name}' has {len(pool)} models but {count} agents need one each"
                )
