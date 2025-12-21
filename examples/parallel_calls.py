"""
Parallel LLM orchestration with asyncio.gather().
Copilot: Reference this for multi-model consensus or parallel tasks.
"""
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

async def call_openai(prompt: str) -> dict:
    """Call OpenAI API."""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"provider": "openai", "content": response.choices[0].message.content}

async def call_anthropic(prompt: str) -> dict:
    """Call Anthropic API."""
    client = AsyncAnthropic()
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return {"provider": "anthropic", "content": response.content[0].text}

async def parallel_orchestration(prompt: str) -> list[dict]:
    """
    Call multiple LLMs in parallel and aggregate results.
    
    Latency: max(2s, 2s) = 2s (instead of 2s + 2s = 4s)
    """
    tasks = [
        call_openai(prompt),
        call_anthropic(prompt)
    ]
    
    # Run in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors
    valid_results = [r for r in results if not isinstance(r, Exception)]
    return valid_results

if __name__ == "__main__":
    results = asyncio.run(parallel_orchestration("Explain async/await"))
    for result in results:
        print(f"{result['provider']}: {result['content'][:100]}...")