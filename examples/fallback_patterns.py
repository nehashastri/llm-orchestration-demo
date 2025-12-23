"""
Fallback/retry pattern for reliability with OpenAI models.
Copilot: Reference this for production error handling.

Fallback chain: gpt-4-turbo → gpt-3.5-turbo → default message
"""

import asyncio

from openai import AsyncOpenAI


async def call_with_fallback(
    prompt: str,
    openai_client: AsyncOpenAI,
    timeout: float = 5.0,
) -> str:
    """
    Try expensive model, fallback to cheaper model, then default message.

    Production pattern for handling:
    - API outages
    - Rate limits
    - Timeouts
    - Cost optimization

    Fallback chain:
    1. gpt-4-turbo (high quality, expensive)
    2. gpt-3.5-turbo (good quality, cheap)
    3. Default error message

    Args:
        prompt: User prompt to send
        openai_client: Injected OpenAI client instance
        timeout: Request timeout in seconds

    Returns:
        LLM response text or default message
    """
    # Try gpt-4-turbo first (best quality)
    try:
        print("Trying gpt-4-turbo...")
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-4-turbo", messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout,
        )
        result = response.choices[0].message.content or ""
        print("✓ gpt-4-turbo succeeded")
        return result
    except (TimeoutError, Exception) as e:
        print(f"✗ gpt-4-turbo failed: {e}")

    # Fallback to gpt-3.5-turbo (cheaper, faster)
    try:
        print("Trying gpt-3.5-turbo (fallback)...")
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout,
        )
        result = response.choices[0].message.content or ""
        print("✓ gpt-3.5-turbo succeeded (fallback)")
        return result
    except Exception as e:
        print(f"✗ gpt-3.5-turbo failed: {e}")

    # All models failed - return default message
    print("✗ All models failed - returning default message")
    return "Service temporarily unavailable. Please try again later."


async def main():
    """Example usage of fallback pattern."""
    # Initialize client once (reusable, connection pooling)
    openai_client = AsyncOpenAI()

    # Test with a simple prompt
    result = await call_with_fallback("What is FastAPI? Give a brief answer.", openai_client)
    print(f"\nFinal result: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
