"""
Fallback/retry pattern for reliability.
Copilot: Reference this for production error handling.
"""

import asyncio

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


async def call_with_fallback(
    prompt: str,
    openai_client: AsyncOpenAI,
    anthropic_client: AsyncAnthropic,
    timeout: float = 5.0,
) -> str:
    """
    Try primary LLM, fallback to secondary on failure.

    Production pattern for handling:
    - API outages
    - Rate limits
    - Timeouts

    Args:
        prompt: User prompt to send
        openai_client: Injected OpenAI client instance
        anthropic_client: Injected Anthropic client instance
        timeout: Request timeout in seconds

    Returns:
        LLM response text
    """
    # Try OpenAI first (cheaper)
    try:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content or ""
    except (TimeoutError, Exception) as e:
        print(f"OpenAI failed: {e}, trying Anthropic...")

        # Fallback to Anthropic
        try:
            response = await asyncio.wait_for(
                anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                ),
                timeout=timeout,
            )
            # Extract text from the first text content block
            # According to Anthropic docs, response.content is a list of content blocks
            for block in response.content:
                if block.type == "text":
                    return block.text  # type: ignore
            return str(response.content[0])  # Fallback if no text block found
        except Exception as e:
            return f"All LLMs failed: {e}"


if __name__ == "__main__":
    # Initialize clients once (reusable, connection pooling)
    openai_client = AsyncOpenAI()
    anthropic_client = AsyncAnthropic()

    result = asyncio.run(
        call_with_fallback("What is FastAPI?", openai_client, anthropic_client)
    )
    print(result)
