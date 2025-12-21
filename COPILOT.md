# Copilot Instructions for llm-orchestration-demo

## Project Context
FastAPI app for LLM orchestration. Interview demo for AI/ML Engineer role.
Focus: **Speed, async patterns, production-ready code**.

## Critical Rules

### 1. Always Use These Patterns
```python
# ✅ ALWAYS async for I/O
async def call_llm(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
        return response.text

# ✅ ALWAYS type hints
def process(data: dict[str, Any]) -> list[str]:
    ...

# ✅ ALWAYS Pydantic for validation
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

# ✅ ALWAYS timeouts
await asyncio.wait_for(llm_call(), timeout=30)

# ✅ ALWAYS structured logging
logger.info("LLM call", extra={"latency": 1.5, "cost": 0.001})
```

### 2. Project Commands (ALWAYS use pixi)
```bash
pixi run dev        # Start server
pixi run test       # Run tests
pixi run lint       # Lint code
pixi run format     # Format code
```

### 3. Import Order (ALWAYS follow this)
```python
# Standard library
import asyncio
from typing import Any, Optional

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local
from src.llm.clients import OpenAIClient
from src.utils.config import settings
```

### 4. Error Handling Template
```python
try:
    result = await asyncio.wait_for(llm_call(), timeout=30)
    return result
except asyncio.TimeoutError:
    logger.error("LLM timeout")
    raise HTTPException(status_code=504, detail="Request timeout")
except Exception as e:
    logger.error(f"LLM error: {e}")
    raise HTTPException(status_code=500, detail="Internal error")
```

### 5. Test Template
```python
@pytest.mark.asyncio
async def test_llm_call(mock_openai_client):
    """Test successful LLM call."""
    # Arrange
    prompt = "Test prompt"
    
    # Act
    result = await call_llm(prompt)
    
    # Assert
    assert result is not None
    mock_openai_client.create.assert_called_once()
```

### 6. FastAPI Route Template
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: OpenAIClient = Depends(get_openai_client)
) -> ChatResponse:
    """
    Chat endpoint with LLM orchestration.
    
    Args:
        request: Chat request with prompt
        client: OpenAI client (injected)
        
    Returns:
        Chat response with content and metadata
    """
    try:
        content = await client.generate(request.prompt)
        return ChatResponse(content=content, model="gpt-4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Reference Examples
When implementing features, CHECK THESE FIRST:
- `examples/basic_call.py` - Simple LLM call
- `examples/parallel_calls.py` - asyncio.gather() pattern
- `examples/streaming.py` - StreamingResponse pattern
- `examples/fallback_pattern.py` - Retry/fallback logic

## Key Architectural Decisions

### Why Async Everywhere?
- Parallel LLM calls (6s → 2s latency)
- Handle multiple concurrent requests
- Non-blocking I/O

### Why Pydantic?
- Automatic validation (catch bad inputs)
- Auto-generated OpenAPI docs
- Type safety

### Why Dependency Injection?
- Easy mocking in tests
- Centralized configuration
- Reusable across endpoints

## Common Tasks

### Add New LLM Provider
1. Create `src/llm/providers/new_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement `async def generate()`
4. Add tests in `tests/test_new_provider.py`

### Add New API Endpoint
1. Add route in `src/api/routes.py`
2. Add Pydantic models in `src/api/models.py`
3. Add tests in `tests/test_api.py`
4. Update docs in `docs/ARCHITECTURE.md`

### Debug Issues
1. Check logs: `pixi run dev` (in terminal)
2. Set breakpoint in VS Code (F9)
3. Start debugger (F5)
4. Use integrated terminal for `print()` debugging

## Performance Targets (Job Requirements)
- ✅ Sub-500ms first token latency (use streaming)
- ✅ Handle 100 concurrent requests (async/await)
- ✅ Cost < $0.01 per request (model selection)

## Security Checklist
- ❌ NEVER hardcode API keys (use .env)
- ❌ NEVER log sensitive data
- ✅ ALWAYS validate inputs (Pydantic)
- ✅ ALWAYS set timeouts
- ✅ ALWAYS handle errors gracefully

## Code Style
- Max line length: 100 characters
- Double quotes for strings
- Type hints mandatory
- Docstrings for public functions (Google style)

## Git Workflow
```bash
# Always work on feature branch
git checkout -b feature/new-feature

# Commit often with clear messages
git add .
git commit -m "feat: add parallel LLM orchestration"

# Push and create PR
git push -u origin feature/new-feature
```