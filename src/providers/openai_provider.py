"""OpenAI and OpenAI-compatible provider."""

from openai import OpenAI

from .base import BaseLLMProvider, DEFAULT_TIMEOUT
from ..config import PoolInstance


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models and any OpenAI-compatible API.

    Works with: OpenAI, DeepSeek, vLLM, Ollama, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "",
        pool_instance: "PoolInstance | None" = None,
    ):
        self._model = model
        self._pool_instance = pool_instance
        self._api_key = api_key
        self._base_url = base_url or None
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs, timeout=DEFAULT_TIMEOUT)

    @property
    def model_name(self) -> str:
        return f"openai:{self._model}"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temperature = kwargs.get("temperature", 0.7)

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if self._is_quota_exhausted(e) and self._pool_instance:
                    try:
                        self._model = self._pool_instance.replace(self._model)
                        self._client = OpenAI(
                            api_key=self._api_key,
                            base_url=self._base_url or "",
                            timeout=DEFAULT_TIMEOUT,
                        )
                        continue
                    except RuntimeError as pool_err:
                        raise RuntimeError(
                            f"All models in pool exhausted after {self._model} returned 403"
                        ) from pool_err
                raise

    def _is_quota_exhausted(self, exc: Exception) -> bool:
        """Check if exception indicates quota exhaustion (403)."""
        status_code = getattr(exc, "status_code", None)
        if status_code == 403:
            return True
        msg = str(exc).lower()
        return (
            "quota" in msg
            or "insufficient" in msg
            or "exhausted" in msg
            or "limit" in msg
        ) and "403" in msg
