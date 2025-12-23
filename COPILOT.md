# Copilot Instructions for llm-orchestration-demo

## Project Context
FastAPI app for LLM orchestration. Interview demo for AI/ML Engineer role.
Scope: **OpenAI-only** (4o / 4o-mini / 4-turbo / 3.5).
Focus: **Speed, async patterns, production-ready code**.

## Agent Constitution

**CRITICAL RULE:** If the user corrects you on anything, you must:

- Immediately record the correction in [COPILOT.md](COPILOT.md) in an appropriate section.
- Continue with what you were doing, applying the correction.

This ensures that corrections become part of the permanent knowledge base for all future agent interactions.

**MAINTENANCE RULE:** Periodically review and organize [COPILOT.md](COPILOT.md). As repeat issues emerge, progressively strengthen memory on those matters through:

- Emphasis: Use bold, italics, or formatting to highlight critical information.
- Position: Move important or frequently violated rules to more prominent locations (e.g., top of sections, Agent Constitution).
- Repetition: Reinforce key points by mentioning them in multiple relevant sections.
- Alerts: Add explicit warnings or "CRITICAL" markers for rules that are commonly missed.

This ensures that the most important and frequently needed guidance is easily accessible and hard to miss.

---

## Getting Started

### Quick Setup
1. Install dependencies: `pixi install`
2. Enter environment: `pixi shell` (optional; can use `pixi run` instead)
3. Start dev server: `pixi run dev`
4. View API docs: `start http://localhost:8000/docs`

### Essential Rules
Always use Pixi for commands:
```bash
pixi run test                    # Run tests
pixi run format                  # Format code + markdown
pixi run lint                    # Lint and fix
pixi run dev                     # Start server
```

Never run commands directly (e.g., `pytest`, `python`, `mypy`) — always prefix with `pixi run`.

### Reference Examples
When implementing features, CHECK THESE FIRST:
- `examples/basic_call.py` — Simple LLM call
- `examples/parallel_calls.py` — asyncio.gather() pattern
- `examples/streaming.py` — StreamingResponse pattern
- `examples/fallback_patterns.py` — Retry/fallback logic

---

## How-to Guides

### Add New API Endpoint
1. Add route in `src/api/routes.py`
2. Add Pydantic models in `src/api/models.py`
3. Add tests in `tests/test_api.py` (mirror structure: `test_<module>.py`)
4. Update docs in `ARCHITECTURE.md` and `docs/api_specs.md`

### Add New Provider (if expanding beyond OpenAI)
1. Create `src/llm/providers/new_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement `async def generate()`
4. Add tests in `tests/test_new_provider.py`

### Debug Issues
1. Check logs: `pixi run dev` (in terminal)
2. Set breakpoint in VS Code (F9)
3. Start debugger (F5)
4. Use integrated terminal for `print()` debugging

### Git Workflow
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and write tests
3. Run `pixi run test` to validate
4. Stage and commit: `git add .` then `git commit -m "feat: description"`
5. Push and create PR: `git push -u origin feature/new-feature`

---

## Reference

### Pixi Commands
```bash
# Environment & deps
pixi install                      # Install dependencies and set up envs

# Development server
pixi run dev                      # Start server (127.0.0.1:8000)
pixi run dev-public               # Start server bound to 0.0.0.0:8000

# Testing
pixi run test                     # Run the full pytest suite
pixi run test-cov                 # Tests with coverage (HTML at htmlcov/)
pixi run -e dev pytest tests/path/to/specific_test.py   # Run a single test file
pixi run -e dev pytest -k "substring_or_marker"         # Run tests matching a pattern

# Quality
pixi run format                   # Format code (ruff format) + markdown (mdformat)
pixi run lint                     # Lint and auto-fix (ruff check --fix)
pixi run typecheck                # Type checking (mypy src/)
pixi run check                    # Format + lint + typecheck aggregate

# Docs (open in browser)
start http://localhost:8000/docs  # Open interactive API docs
```

### Code Patterns (ALWAYS Use These)

**Async I/O:**
```python
async def call_llm(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
        return response.text
```

**Type Hints:**
```python
def process(data: dict[str, Any]) -> list[str]:
    ...
```

**Pydantic Validation:**
```python
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
```

**Timeouts:**
```python
await asyncio.wait_for(llm_call(), timeout=30)
```

**Structured Logging:**
```python
logger.info("LLM call", extra={"latency": 1.5, "cost": 0.001})
```

### Import Order
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

### Error Handling Template
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

### Test Template
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

### FastAPI Route Template
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

### Testing Strategy
- **Framework:** pytest with coverage reporting
- **Structure:** Tests mirror source: `src/module.py` → `tests/test_module.py`
- **Run tests:** Use `pixi run test` for suite or `pixi run test-cov` for HTML coverage
- **Coverage:** Aim for high coverage on core logic (`src/llm/`, `src/api/`)
- **Fixtures:** Use fixtures for mocking clients
- **Markers:** Custom markers (e.g., `@pytest.mark.asyncio`) handled via `conftest.py`

### Docstrings
Use Google-style format. Example:
```python
def process_prompt(prompt: str) -> str:
    """Process user prompt and return formatted result.

    Args:
        prompt: Raw user input string.

    Returns:
        Formatted prompt ready for LLM.
    """
```

### Code Style
- Max line length: 100 characters
- Double quotes for strings
- Type hints mandatory
- Docstrings for public functions (Google style)

### Performance Targets
- Sub-500ms first token latency (use streaming)
- Handle 100 concurrent requests (async/await)
- Cost < $0.01 per request (model selection)

### Security Checklist
- ❌ NEVER hardcode API keys (use .env)
- ❌ NEVER log sensitive data
- ✅ ALWAYS validate inputs (Pydantic)
- ✅ ALWAYS set timeouts
- ✅ ALWAYS handle errors gracefully

---

## Explanation

### Why Async Everywhere?
- Parallel LLM calls reduce latency from 6s → 2s
- Handle multiple concurrent requests efficiently
- Non-blocking I/O allows single thread to serve many clients

### Why Pydantic?
- Automatic request/response validation
- Auto-generated OpenAPI docs at `/docs`
- Type safety and IDE support

### Why Dependency Injection?
- Easy mocking in tests (pass fake clients)
- Centralized configuration and setup
- Reusable across endpoints

## Code Requirements

### Completeness & Style
- All code must be **complete and runnable** — no pseudocode or sketches
- Follow consistent coding style throughout the codebase
- Use descriptive variable and function names (avoid single-letter names except in loops)
- Include helpful comments for complex logic, not for obvious code
- Every function should have a docstring (Google-style)

### Data Handling
- Use appropriate data structures (dict, list, Pydantic models, etc.)
- Implement proper validation (leverage Pydantic for request/response data)
- Handle missing or invalid data gracefully with appropriate error responses
- Use efficient lookups and operations (avoid N² loops when O(N) is possible)

### Import Organization
- Group imports: standard library, third-party, local
- Import modules, not individual functions (e.g., `import json` not `from json import loads`)
- Place all imports at the top of the file

---

## Best Practices

### Testing & Changes
- Always write tests when making code changes
- Tests mirror source structure: `src/module.py` → `tests/test_module.py`
- Run `pixi run test` before committing
- Use test templates above as reference

### Code Organization
- Prefer Pydantic models for request/response validation and data structures
- Use dependency injection (FastAPI `Depends()`) for clients and services
- Keep functions focused on single responsibility
- Only edit files directly involved in the feature or fix

### Error Handling & Logging
- Always set timeouts on async I/O operations
- Catch and log specific exceptions, not generic `Exception`
- Use structured logging with relevant context (latency, cost, request ID)
- Return appropriate HTTP status codes (504 for timeout, 500 for errors, 400 for validation)

---

## Documentation Guidelines

### Content Principles
- **Single source of truth:** Update relevant files (ARCHITECTURE.md, COPILOT.md, api_specs.md, README.md)
- **Avoid duplication:** Don't repeat; link instead
- **Keep it simple:** Prefer concise, focused content over comprehensive guides
- **No stale content:** Don't include version-specific information that can't be updated automatically
- **Pixi-only:** Document only Pixi for setup (`pixi install`, `pixi shell`, `pixi run`). Never document pip, conda, or alternatives
- **Avoid file trees:** Don't include detailed file tree structures; they become stale. Reference actual structure or provide high-level overviews instead.

### Writing Style
- Clear and direct: Avoid grandiose phrasing ("serves as", "plays a vital role")
- Avoid formulaic scaffolding: Skip "it's important to note", "In summary", "Overall"
- No vague attributions: Avoid "industry reports", "some argue" without specifics
- No AI disclaimers: Don't mention being an AI or apologize
- Minimal formatting: Use bold sparingly, avoid overuse of em dashes
- Concrete over abstract: Show patterns and examples

---

## Code Requirements
