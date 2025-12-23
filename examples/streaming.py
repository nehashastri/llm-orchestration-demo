"""
Streaming LLM responses token-by-token.
Copilot: Reference this for real-time chat interfaces.
"""

import asyncio

from openai import AsyncOpenAI


async def stream_llm_response(prompt: str):
    """
    Stream LLM response token-by-token.

    Benefits:
    - First token in ~500ms (vs 10s for full response)
    - Better UX (like ChatGPT)
    - Lower perceived latency
    """
    client = AsyncOpenAI()

    stream = await client.chat.completions.create(
        model="gpt-4-turbo-preview", messages=[{"role": "user", "content": prompt}], stream=True
    )

    print("Response: ", end="")
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
    print()  # New line at end


if __name__ == "__main__":
    asyncio.run(stream_llm_response("Write a haiku about coding"))
