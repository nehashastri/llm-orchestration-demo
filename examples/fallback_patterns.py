"""
Fallback/retry pattern for reliability.
Copilot: Reference this for production error handling.
"""
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

async def call_with_fallback(prompt: str) -> str:
    """
    Try primary LLM, fallback to secondary on failure.
    
    Production pattern for handling:
    - API outages
    - Rate limits
    - Timeouts
    """
    # Try OpenAI first (cheaper)
    try:
        client = AsyncOpenAI()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=5.0
        )
        return response.choices[0].message.content
    except (asyncio.TimeoutError, Exception) as e:
        print(f"OpenAI failed: {e}, trying Anthropic...")
        
        # Fallback to Anthropic
        try:
            client = AsyncAnthropic()
            response = await asyncio.wait_for(
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                ),
                timeout=5.0
            )
            return response.content[0].text
        except Exception as e:
            return f"All LLMs failed: {e}"

if __name__ == "__main__":
    result = asyncio.run(call_with_fallback("What is FastAPI?"))
    print(result)