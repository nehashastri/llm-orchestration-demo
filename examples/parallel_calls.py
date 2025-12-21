"""
Parallel parameter testing with different OpenAI configurations.
Copilot: Reference this for A/B testing or multi-parameter optimization.

Note: True parallel orchestration requires multiple providers.
This example demonstrates concurrent calls with different parameters.
"""

import asyncio
from typing import cast

from openai import AsyncOpenAI


async def call_openai_conservative(prompt: str) -> dict:
    """Call OpenAI with conservative parameters (deterministic)."""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # More deterministic
    )
    return {
        "variant": "conservative (temp=0.2)",
        "model": "gpt-4-turbo",
        "content": response.choices[0].message.content or "",
    }


async def call_openai_creative(prompt: str) -> dict:
    """Call OpenAI with creative parameters (more varied)."""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # More creative
    )
    return {
        "variant": "creative (temp=0.9)",
        "model": "gpt-4-turbo",
        "content": response.choices[0].message.content or "",
    }


async def call_openai_fast(prompt: str) -> dict:
    """Call OpenAI with fast/cheap model."""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return {
        "variant": "fast (gpt-3.5-turbo)",
        "model": "gpt-3.5-turbo",
        "content": response.choices[0].message.content or "",
    }


async def parallel_parameter_testing(prompt: str) -> list[dict]:
    """
    Call OpenAI with different parameters in parallel for A/B testing.

    Use cases:
    - Compare temperature settings
    - Test different models
    - A/B testing for quality vs speed
    - System prompt variations

    Latency: max(t1, t2, t3) instead of t1 + t2 + t3
    """
    tasks = [
        call_openai_conservative(prompt),
        call_openai_creative(prompt),
        call_openai_fast(prompt),
    ]

    # Run in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return cast(list[dict], valid_results)


async def main():
    """Example usage of parallel parameter testing."""
    print("Testing different OpenAI configurations in parallel...\n")

    results = await parallel_parameter_testing("Write a haiku about programming")

    for i, result in enumerate(results, 1):
        print(f"--- Result {i}: {result['variant']} ---")
        print(f"Model: {result['model']}")
        print(f"Content: {result['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
