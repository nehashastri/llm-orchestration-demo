"""
LLM client implementations for OpenAI.

Provides a unified interface for interacting with OpenAI models,
handling response normalization, error handling, and streaming.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from src.utils.config import get_model_config, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All provider-specific clients must implement this interface
    to ensure consistent behavior across providers.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'openai')."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> dict:
        """
        Generate a completion from the LLM.

        Args:
            prompt: User prompt/message
            model: Model identifier (uses default if None)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            system_prompt: System instructions (optional)
            timeout: Request timeout in seconds

        Returns:
            Normalized response dictionary with:
                - content: Generated text
                - model: Model used
                - provider: Provider name
                - usage: Token usage statistics
                - latency_ms: Request latency
                - cost_usd: Estimated cost
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from the LLM.

        Args:
            Same as generate()

        Yields:
            Individual tokens as they are generated
        """
        pass

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for an LLM call.

        Args:
            model: Model identifier
            prompt_tokens: Input tokens
            completion_tokens: Output tokens

        Returns:
            Cost in USD
        """
        config = get_model_config(model)
        prompt_cost = (prompt_tokens / 1_000_000) * config["cost_per_1m_prompt_tokens"]
        completion_cost = (completion_tokens / 1_000_000) * config["cost_per_1m_completion_tokens"]
        return prompt_cost + completion_cost


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for GPT models.

    Supports both chat completions and streaming responses.
    """

    def __init__(self, api_key: str | None = None, api_client: AsyncOpenAI | None = None):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses settings.openai_api_key if None)
            api_client: Pre-configured AsyncOpenAI client (for testing)
        """
        self._client = (
            api_client
            if api_client is not None
            else AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        )

    @property
    def provider(self) -> str:
        return "openai"

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Generate completion using OpenAI."""
        model = model or settings.default_model
        temperature = temperature if temperature is not None else settings.default_temperature
        max_tokens = max_tokens or settings.default_max_tokens
        timeout = timeout or settings.default_timeout

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Track timing
        start_time = time.time()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract data
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Calculate cost
            cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)

            # Log the call
            from src.utils.logger import log_llm_call

            log_llm_call(
                model=model,
                provider=self.provider,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                success=True,
            )

            return {
                "content": content,
                "model": model,
                "provider": self.provider,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "metrics": {"latency_ms": round(latency_ms, 2), "cost_usd": round(cost_usd, 6)},
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            from src.utils.logger import log_llm_call

            log_llm_call(
                model=model,
                provider=self.provider,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                success=False,
                error=str(e),
            )

            raise

    def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using OpenAI."""

        async def _gen() -> AsyncGenerator[str, None]:
            mdl = model or settings.default_model
            temp = temperature if temperature is not None else settings.default_temperature
            max_toks = max_tokens or settings.default_max_tokens

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                stream = await self._client.chat.completions.create(
                    model=mdl,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_toks,
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            except Exception as e:
                logger.error("openai_stream_error", error=str(e))
                raise

        return _gen()


# Factory function to get the appropriate client
def get_client(provider: str) -> BaseLLMClient:
    """
    Get an LLM client for the specified provider.

    Args:
        provider: Provider name ("openai")

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If provider is unknown
    """
    if provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Only 'openai' is supported.")
