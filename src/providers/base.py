"""Base LLM provider abstract interface."""

from abc import ABC, abstractmethod

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(3600.0, connect=30.0)


class BaseLLMProvider(ABC):
    """Unified interface for all LLM providers.

    Modeled after OpenAI chat-completions protocol.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate a response synchronously.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User-level input.
            **kwargs: Provider-specific extras (e.g. temperature).

        Returns:
            The model's text response.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""
        ...
