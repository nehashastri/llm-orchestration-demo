"""
Basic LLM call with error handling.
Copilot: Reference this for simple LLM interactions.
"""

import asyncio
import os

from openai import AsyncOpenAI


async def basic_llm_call(prompt: str) -> str:
    """
    Make a simple LLM call with error handling.

    Args:
        prompt: User prompt

    Returns:
        LLM response text
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            ),
            timeout=30.0,
        )
        return response.choices[0].message.content or ""
    except TimeoutError:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    result = asyncio.run(basic_llm_call("What is FastAPI?"))
    print(result)
