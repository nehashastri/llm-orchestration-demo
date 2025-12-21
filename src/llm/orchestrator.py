"""
LLM orchestration strategies for parallel execution, fallback, and streaming.

Provides high-level orchestration logic for calling multiple LLM providers
with different strategies (parallel, fallback, consensus).
"""

import asyncio
from collections.abc import AsyncIterator

from src.llm.clients import get_client
from src.utils.config import get_provider_for_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def parallel_orchestration(
    prompt: str,
    providers: list[str] | None = None,
    models: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Call multiple LLM providers in parallel and return all responses.

    Useful for comparing outputs or getting the fastest response.

    Args:
        prompt: User prompt
        providers: List of provider names (default: ["openai", "anthropic"])
        models: List of specific models (overrides providers if given)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System instructions

    Returns:
        Dictionary with:
            - content: Content from fastest response
            - winner: Information about fastest provider
            - all_responses: List of all responses
            - metrics: Aggregated metrics (total cost, latency)
    """
    # Default to both providers if not specified
    if not providers and not models:
        providers = ["openai", "anthropic"]

    # If models specified, infer providers from models
    if models:
        providers = [get_provider_for_model(model) for model in models]
    else:
        models = [None] * len(providers)  # Use default models

    # Create tasks for all providers
    tasks = []
    for provider, model in zip(providers, models):
        client = get_client(provider)
        task = client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        tasks.append(task)

    logger.info("parallel_orchestration_start", num_providers=len(providers))

    # Execute all tasks in parallel
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful results from errors
        successful_results = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"provider": providers[i], "error": str(result)})
                logger.error("parallel_provider_failed", provider=providers[i], error=str(result))
            else:
                successful_results.append(result)

        if not successful_results:
            raise Exception("All providers failed")

        # Sort by latency to find winner
        successful_results.sort(key=lambda x: x["metrics"]["latency_ms"])
        winner = successful_results[0]

        # Calculate aggregate metrics
        total_cost = sum(r["metrics"]["cost_usd"] for r in successful_results)
        max_latency = max(r["metrics"]["latency_ms"] for r in successful_results)

        logger.info(
            "parallel_orchestration_complete",
            winner_provider=winner["provider"],
            num_successful=len(successful_results),
            num_failed=len(errors),
            total_cost_usd=total_cost,
        )

        return {
            "content": winner["content"],
            "winner": {
                "provider": winner["provider"],
                "model": winner["model"],
                "latency_ms": winner["metrics"]["latency_ms"],
            },
            "all_responses": successful_results,
            "errors": errors if errors else None,
            "metrics": {
                "total_latency_ms": round(max_latency, 2),
                "total_cost_usd": round(total_cost, 6),
                "num_providers_called": len(providers),
                "num_successful": len(successful_results),
                "num_failed": len(errors),
            },
        }

    except Exception as e:
        logger.error("parallel_orchestration_failed", error=str(e))
        raise


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
    Try primary provider, fallback to others on failure.

    Useful for reliability - ensures request succeeds even if primary fails.

    Args:
        prompt: User prompt
        primary_provider: Primary provider to try first
        fallback_providers: List of fallback providers (default: ["anthropic"])
        primary_model: Specific model for primary provider
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System instructions
        timeout: Request timeout

    Returns:
        Dictionary with:
            - content: Generated text
            - provider_used: Which provider succeeded
            - primary_success: Whether primary succeeded
            - fallback_triggered: Whether fallback was used
            - primary_error: Error from primary (if failed)
            - metrics: Latency and cost metrics
    """
    if fallback_providers is None:
        fallback_providers = ["anthropic"] if primary_provider == "openai" else ["openai"]

    all_providers = [primary_provider] + fallback_providers

    logger.info(
        "fallback_orchestration_start", primary=primary_provider, fallbacks=fallback_providers
    )

    primary_error = None

    # Try each provider in order
    for i, provider in enumerate(all_providers):
        is_primary = i == 0
        client = get_client(provider)

        try:
            logger.info("trying_provider", provider=provider, is_primary=is_primary, attempt=i + 1)

            result = await client.generate(
                prompt=prompt,
                model=primary_model if is_primary else None,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                timeout=timeout,
            )

            # Success!
            logger.info(
                "fallback_orchestration_success",
                provider_used=provider,
                primary_success=is_primary,
                fallback_triggered=not is_primary,
            )

            return {
                **result,
                "provider_used": provider,
                "primary_success": is_primary,
                "fallback_triggered": not is_primary,
                "primary_error": primary_error if not is_primary else None,
            }

        except Exception as e:
            error_msg = str(e)

            if is_primary:
                primary_error = error_msg

            logger.warning(
                "provider_failed",
                provider=provider,
                error=error_msg,
                is_primary=is_primary,
                has_more_fallbacks=i < len(all_providers) - 1,
            )

            # If this was the last provider, raise
            if i == len(all_providers) - 1:
                logger.error("fallback_orchestration_failed", error="All providers failed")
                raise Exception(f"All providers failed. Primary error: {primary_error}")


async def streaming_orchestration(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
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


async def consensus_orchestration(
    prompt: str,
    providers: list[str] | None = None,
    temperature: float = 0.3,  # Lower temp for more consistent responses
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    consensus_threshold: float = 0.5,
) -> dict:
    """
    Call multiple providers and return consensus response.

    Useful for important decisions where you want agreement from multiple models.

    Args:
        prompt: User prompt
        providers: List of provider names
        temperature: Sampling temperature (lower for consistency)
        max_tokens: Maximum tokens
        system_prompt: System instructions
        consensus_threshold: Minimum agreement threshold (0.0-1.0)

    Returns:
        Dictionary with:
            - content: Consensus response (or most common response)
            - confidence: Agreement level (0.0-1.0)
            - all_responses: All individual responses
            - metrics: Aggregated metrics
    """
    # Get all responses in parallel
    result = await parallel_orchestration(
        prompt=prompt,
        providers=providers,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    responses = [r["content"] for r in result["all_responses"]]

    # Simple consensus: find most common response
    # In production, you'd use semantic similarity
    from collections import Counter

    response_counts = Counter(responses)
    most_common_response, count = response_counts.most_common(1)[0]

    confidence = count / len(responses)

    logger.info(
        "consensus_orchestration_complete",
        num_responses=len(responses),
        confidence=confidence,
        threshold=consensus_threshold,
        consensus_reached=confidence >= consensus_threshold,
    )

    return {
        "content": most_common_response,
        "confidence": round(confidence, 2),
        "consensus_reached": confidence >= consensus_threshold,
        "all_responses": result["all_responses"],
        "metrics": result["metrics"],
    }
