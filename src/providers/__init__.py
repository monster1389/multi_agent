"""Provider factory — instantiate the right provider from config."""

from ..config import ProviderConfig, ModelPool
from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def create_provider(
    config: ProviderConfig,
    pools: dict[str, ModelPool] | None = None,
) -> BaseLLMProvider:
    """Create a provider instance from its configuration.

    Supports pool:<name> syntax for model — allocates from the named pool.
    Passes pool reference to provider for automatic fallback on 403 errors.

    Args:
        config: ProviderConfig with type, model, api_key, base_url.
        pools: Optional dict of pool_name → ModelPool for dynamic allocation.

    Returns:
        A concrete BaseLLMProvider instance.

    Raises:
        ValueError: If provider type is unknown or pool reference is invalid.
    """
    pools = pools or {}
    actual_model = config.model
    model_pool = None

    if config.model.startswith("pool:"):
        pool_name = config.model.split(":", 1)[1]
        pool = pools.get(pool_name)
        if pool is None:
            raise ValueError(f"Unknown model pool: {pool_name}")
        actual_model = pool.allocate()
        model_pool = pool

    if config.type == "openai":
        return OpenAIProvider(
            model=actual_model,
            api_key=config.api_key,
            base_url=config.base_url,
            model_pool=model_pool,
        )
    elif config.type == "anthropic":
        return AnthropicProvider(
            model=actual_model,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unknown provider type: {config.type}")
