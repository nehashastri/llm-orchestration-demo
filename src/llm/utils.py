"""
Utility functions for LLM operations.

Includes cost calculation, latency tracking, and response normalization.
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.utils.config import get_model_config


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost for an LLM call.

    Args:
        model: Model identifier
        prompt_tokens: Input tokens
        completion_tokens: Output tokens

    Returns:
        Cost in USD

    Raises:
        ValueError: If model is unknown
    """
    try:
        config = get_model_config(model)
    except ValueError as e:
        raise ValueError(f"Unknown model: {model}") from e

    prompt_cost = (prompt_tokens / 1_000_000) * config["cost_per_1m_prompt_tokens"]
    completion_cost = (completion_tokens / 1_000_000) * config["cost_per_1m_completion_tokens"]
    return prompt_cost + completion_cost


def track_latency(func: Callable) -> Callable:
    """
    Decorator to track latency of async functions.

    Adds 'latency_ms' to the returned dictionary.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped function that includes latency tracking
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        if isinstance(result, dict):
            result["latency_ms"] = latency_ms
        return result

    return wrapper


def normalize_response(raw_response: dict[str, Any], provider: str) -> dict[str, Any]:
    """
    Normalize LLM response to a standard format.

    Args:
        raw_response: Raw response from the LLM provider
        provider: Provider name ('openai', 'anthropic', etc.)

    Returns:
        Normalized response dictionary with:
            - content: Generated text
            - usage: Token usage statistics
            - model: Model identifier (if available)
    """
    if provider == "openai":
        return {
            "content": raw_response["choices"][0]["message"]["content"],
            "usage": {
                "prompt_tokens": raw_response["usage"]["prompt_tokens"],
                "completion_tokens": raw_response["usage"]["completion_tokens"],
                "total_tokens": raw_response["usage"].get(
                    "total_tokens",
                    raw_response["usage"]["prompt_tokens"]
                    + raw_response["usage"]["completion_tokens"],
                ),
            },
            "model": raw_response.get("model"),
        }
    elif provider == "anthropic":
        return {
            "content": raw_response["content"][0]["text"],
            "usage": {
                "prompt_tokens": raw_response["usage"]["input_tokens"],
                "completion_tokens": raw_response["usage"]["output_tokens"],
                "total_tokens": raw_response["usage"]["input_tokens"]
                + raw_response["usage"]["output_tokens"],
            },
            "model": raw_response.get("model"),
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")
