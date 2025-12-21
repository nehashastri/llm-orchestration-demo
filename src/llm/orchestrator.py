"""
LLM orchestration strategies for fallback and streaming.

Provides high-level orchestration logic for calling OpenAI with
fallback strategies and streaming support.
"""

from collections.abc import AsyncGenerator

from src.llm.clients import get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Parallel orchestration disabled - only single OpenAI provider supported
# async def parallel_orchestration(
#     prompt: str,
#     providers: list[str] | None = None,
#     models: list[str] | None = None,
#     temperature: float | None = None,
#     max_tokens: int | None = None,
#     system_prompt: str | None = None,
# ) -> dict:
#     """Disabled: Parallel orchestration requires multiple providers."""
#     raise NotImplementedError(
#         "Parallel orchestration is not available in OpenAI-only mode. "
#         "Use fallback_orchestration instead."
#     )


async def fallback_orchestration(
    prompt: str,
    primary_provider: str = "openai",
    fallback_providers: list[str] | None = None,
    primary_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    timeout: int | None = None,
) -> dict:
    """
    Try primary model, fallback to gpt-3.5-turbo, then return default message.

    Fallback chain: primary_model → gpt-3.5-turbo → default message

    Args:
        prompt: User prompt
        primary_provider: Primary provider (only "openai" supported)
        fallback_providers: Ignored (kept for API compatibility)
        primary_model: Specific model for primary attempt (default: gpt-4-turbo)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System instructions
        timeout: Request timeout

    Returns:
        Dictionary with:
            - content: Generated text or default message
            - provider_used: "openai" or "default"
            - primary_success: Whether primary succeeded
            - fallback_triggered: Whether fallback was used
            - primary_error: Error from primary (if failed)
            - is_default_message: True if default message returned
            - metrics: Latency and cost metrics (if available)
    """
    if primary_provider != "openai":
        raise ValueError("Only 'openai' provider is supported")

    # Fallback chain: primary_model → gpt-3.5-turbo → default message
    primary_model = primary_model or "gpt-4-turbo"
    fallback_model = "gpt-3.5-turbo" if primary_model != "gpt-3.5-turbo" else None

    logger.info(
        "fallback_orchestration_start",
        primary_model=primary_model,
        fallback_model=fallback_model,
    )

    client = get_client("openai")
    primary_error = None
    fallback_error = None

    # Try primary model
    try:
        logger.info("trying_primary_model", model=primary_model)

        result = await client.generate(
            prompt=prompt,
            model=primary_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            timeout=timeout,
        )

        logger.info("fallback_orchestration_success", model=primary_model)

        return {
            **result,
            "provider_used": "openai",
            "primary_success": True,
            "fallback_triggered": False,
            "primary_error": None,
            "is_default_message": False,
        }

    except Exception as e:
        primary_error = str(e)
        logger.warning("primary_model_failed", model=primary_model, error=primary_error)

    # Try fallback model (gpt-3.5-turbo)
    if fallback_model:
        try:
            logger.info("trying_fallback_model", model=fallback_model)

            result = await client.generate(
                prompt=prompt,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                timeout=timeout,
            )

            logger.info("fallback_orchestration_success_with_fallback", model=fallback_model)

            return {
                **result,
                "provider_used": "openai",
                "primary_success": False,
                "fallback_triggered": True,
                "primary_error": primary_error,
                "is_default_message": False,
            }

        except Exception as e:
            fallback_error = str(e)
            logger.warning("fallback_model_failed", model=fallback_model, error=fallback_error)

    # All models failed - return default message
    logger.error(
        "all_models_failed",
        primary_error=primary_error,
        fallback_error=fallback_error,
    )

    default_message = "Service temporarily unavailable. Please try again later."

    return {
        "content": default_message,
        "provider_used": "default",
        "model": "none",
        "primary_success": False,
        "fallback_triggered": True,
        "primary_error": primary_error,
        "fallback_error": fallback_error,
        "is_default_message": True,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "metrics": {"latency_ms": 0, "cost_usd": 0},
    }


async def streaming_orchestration(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from an LLM provider.

    Yields tokens as they are generated for real-time responses.

    Args:
        prompt: User prompt
        provider: Provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System instructions

    Yields:
        Individual tokens as strings
    """
    client = get_client(provider)

    logger.info("streaming_orchestration_start", provider=provider, model=model)

    try:
        token_count = 0
        async for token in client.generate_stream(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        ):
            token_count += 1
            yield token

        logger.info(
            "streaming_orchestration_complete", provider=provider, tokens_streamed=token_count
        )

    except Exception as e:
        logger.error("streaming_orchestration_failed", error=str(e))
        raise


# Consensus orchestration disabled - requires parallel orchestration
# async def consensus_orchestration(
#     prompt: str,
#     providers: list[str] | None = None,
#     temperature: float = 0.3,
#     max_tokens: int | None = None,
#     system_prompt: str | None = None,
#     consensus_threshold: float = 0.5,
# ) -> dict:
#     """Disabled: Consensus orchestration requires multiple providers."""
#     raise NotImplementedError(
#         "Consensus orchestration is not available in OpenAI-only mode."
#     )
