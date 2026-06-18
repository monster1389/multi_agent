"""Provider factory — instantiate the right provider from config."""

from ..config import ProviderConfig, PoolInstance
from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def create_provider(
    config: ProviderConfig,
    pool_instances: dict[str, PoolInstance] | None = None,
) -> BaseLLMProvider:
    """Create a provider instance from its configuration.

    Supports pool:<name> syntax for model — checks out from the named PoolInstance.
    Passes PoolInstance reference to OpenAIProvider for automatic fallback on 403 errors.

    Args:
        config: ProviderConfig with type, model, api_key, base_url.
        pool_instances: Optional dict of pool_name → PoolInstance for checkout.

    Returns:
        A concrete BaseLLMProvider instance.

    Raises:
        ValueError: If provider type is unknown or pool reference is invalid.
    """
    pool_instances = pool_instances or {}
    actual_model = config.model
    pool_instance = None

    if config.model.startswith("pool:"):
        pool_name = config.model.split(":", 1)[1]
        instance = pool_instances.get(pool_name)
        if instance is None:
            raise ValueError(f"Unknown model pool: {pool_name}")
        actual_model = instance.checkout()
        pool_instance = instance

    if config.type == "openai":
        return OpenAIProvider(
            model=actual_model,
            api_key=config.api_key,
            base_url=config.base_url,
            pool_instance=pool_instance,
        )
    elif config.type == "anthropic":
        return AnthropicProvider(
            model=actual_model,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unknown provider type: {config.type}")
