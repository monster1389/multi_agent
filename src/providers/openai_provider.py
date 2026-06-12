"""OpenAI and OpenAI-compatible provider."""

from openai import OpenAI

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models and any OpenAI-compatible API.

    Works with: OpenAI, DeepSeek, vLLM, Ollama, etc.
    """

    def __init__(self, model: str, api_key: str, base_url: str = ""):
        self._model = model
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        return f"openai:{self._model}"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temperature = kwargs.get("temperature", 0.7)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
