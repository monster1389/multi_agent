"""Provider factory — instantiate the right provider from config."""

from ..config import ProviderConfig
from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def create_provider(config: ProviderConfig) -> BaseLLMProvider:
    """Create a provider instance from its configuration.

    Args:
        config: ProviderConfig with type, model, api_key, base_url.

    Returns:
        A concrete BaseLLMProvider instance.

    Raises:
        ValueError: If provider type is unknown.
    """
    if config.type == "openai":
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif config.type == "anthropic":
        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unknown provider type: {config.type}")
