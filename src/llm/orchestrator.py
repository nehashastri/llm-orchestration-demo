"""
LLM orchestration strategies for fallback, parallel fan-out, and streaming.

Provides high-level orchestration logic for calling OpenAI with
fallback strategies, multi-model parallel fan-out, and streaming support.
"""

import asyncio
import time
from collections.abc import AsyncGenerator

from src.llm.clients import get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Ranked list of OpenAI models from largest → smallest for fallbacks
OPENAI_FALLBACK_CHAIN = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


async def parallel_orchestration(
    prompt: str,
    version: int = 1,
    models: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    timeout: int | None = None,
) -> dict:
    """
    Fan-out a single prompt to multiple OpenAI models in parallel.

    version 1: race gpt-4o vs gpt-4o-mini, return first success.
    version 2: call gpt-4o, gpt-4-turbo, gpt-4o-mini, return fastest success but
               still aggregate all completions for comparison.
    """

    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")

    # Default model sets per version
    default_models = (
        ["gpt-4o", "gpt-4o-mini"]
        if version == 1
        else [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4o-mini",
        ]
    )
    candidate_models = models or default_models

    client = get_client("openai")

    async def call_model(model_id: str) -> tuple[str, dict] | tuple[str, Exception]:
        start = time.time()
        try:
            result = await client.generate(
                prompt=prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                timeout=timeout,
            )
            latency_ms = (time.time() - start) * 1000
            result["metrics"]["latency_ms"] = round(latency_ms, 2)
            return model_id, result
        except Exception as exc:  # noqa: BLE001
            return model_id, exc

    tasks = [asyncio.create_task(call_model(model_id)) for model_id in candidate_models]

    successes: list[dict] = []
    errors: list[dict] = []
    winner: dict | None = None

    for task in asyncio.as_completed(tasks):
        model_id, outcome = await task
        if isinstance(outcome, Exception):
            errors.append({"provider": "openai", "model": model_id, "error": str(outcome)})
            continue

        successes.append(outcome)

        if winner is None:
            winner = outcome
            if version == 1:
                # Version 1 returns immediately on first success while others finish in background
                continue

    # Wait for any remaining tasks to settle to populate errors/success metrics
    for pending in tasks:
        if pending.done():
            continue
        model_id, outcome = await pending
        if isinstance(outcome, Exception):
            errors.append({"provider": "openai", "model": model_id, "error": str(outcome)})
        else:
            successes.append(outcome)

    if winner is None and successes:
        # Pick the fastest among successes
        winner = min(successes, key=lambda r: r["metrics"].get("latency_ms", float("inf")))

    if winner is None:
        raise RuntimeError("All parallel model calls failed")

    total_latency_ms = max(s["metrics"]["latency_ms"] for s in successes)
    total_cost = sum(s["metrics"]["cost_usd"] for s in successes)

    return {
        "content": winner["content"],
        "winner": {
            "provider": winner.get("provider", "openai"),
            "model": winner["model"],
            "latency_ms": winner["metrics"]["latency_ms"],
        },
        "all_responses": successes,
        "errors": errors or None,
        "metrics": {
            "total_latency_ms": round(total_latency_ms, 2),
            "total_cost_usd": round(total_cost, 6),
            "num_providers_called": len(candidate_models),
            "num_successful": len(successes),
            "num_failed": len(errors),
        },
    }


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
    Try a larger OpenAI model, then fall back to smaller ones.

    Fallback chain: primary_model (default gpt-4o) → gpt-4o-mini → gpt-3.5-turbo → default message
    """
    if primary_provider != "openai":
        raise ValueError("Only 'openai' provider is supported")

    primary_model = primary_model or "gpt-4o"

    # Build an ordered chain from the primary down to smaller models
    ordered_chain = [primary_model] + [m for m in OPENAI_FALLBACK_CHAIN if m != primary_model]

    logger.info("fallback_orchestration_start", primary_model=primary_model, chain=ordered_chain)

    client = get_client("openai")
    primary_error = None
    fallback_error = None

    for idx, model_id in enumerate(ordered_chain):
        is_primary = idx == 0
        try:
            logger.info(
                "trying_model", model=model_id, role="primary" if is_primary else "fallback"
            )

            result = await client.generate(
                prompt=prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                timeout=timeout,
            )

            logger.info(
                "fallback_orchestration_success",
                model=model_id,
                role="primary" if is_primary else "fallback",
            )

            return {
                **result,
                "provider_used": "openai",
                "primary_success": is_primary,
                "fallback_triggered": not is_primary,
                "primary_error": primary_error if not is_primary else None,
                "is_default_message": False,
            }

        except Exception as exc:  # noqa: BLE001
            if is_primary:
                primary_error = str(exc)
                logger.warning("primary_model_failed", model=model_id, error=primary_error)
            else:
                fallback_error = str(exc)
                logger.warning("fallback_model_failed", model=model_id, error=fallback_error)

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
