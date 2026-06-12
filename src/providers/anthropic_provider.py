"""Anthropic Claude provider — adapts to OpenAI-compatible interface."""

from anthropic import Anthropic

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models.

    Internally uses Anthropic Messages API, adapting system/user roles.
    """

    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.7)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        # Anthropic returns a list of content blocks
        content = response.content
        if isinstance(content, list):
            return "".join(
                block.text for block in content if hasattr(block, "text")
            )
        return str(content)
