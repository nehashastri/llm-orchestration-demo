"""
Shared test fixtures for pytest.

This file contains reusable test fixtures that mock external dependencies
(OpenAI API) and provide test clients for the FastAPI app.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Mock API responses
MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4-turbo",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "This is a mock response from OpenAI."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

MOCK_OPENAI_GPT35_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1677652289,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response from gpt-3.5-turbo.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 15, "total_tokens": 23},
}


# ============================================================================
# FastAPI Test Clients
# ============================================================================


@pytest.fixture
def client():
    """
    Synchronous test client for FastAPI app.

    Usage:
        def test_endpoint(client):
            response = client.get("/health")
            assert response.status_code == 200
    """
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
async def async_client():
    """
    Async test client for FastAPI app.

    Usage:
        @pytest.mark.asyncio
        async def test_endpoint(async_client):
            response = await async_client.get("/health")
            assert response.status_code == 200
    """
    from httpx import ASGITransport

    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ============================================================================
# OpenAI Mocks
# ============================================================================


@pytest.fixture
def mock_openai_client():
    """
    Mock OpenAI client with successful responses.

    Usage:
        def test_openai_call(mock_openai_client):
            response = await mock_openai_client.chat.completions.create(...)
            assert response.choices[0].message.content == "..."
    """
    client = AsyncMock()

    # Mock the chat.completions.create method
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = MOCK_OPENAI_RESPONSE["choices"][0]["message"][
        "content"
    ]
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_response.model = "gpt-4-turbo"

    client.chat.completions.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def mock_openai_client_error():
    """
    Mock OpenAI client that raises errors.

    Usage:
        def test_openai_error(mock_openai_client_error):
            with pytest.raises(Exception):
                await mock_openai_client_error.chat.completions.create(...)
    """
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI API error"))
    return client


@pytest.fixture
def mock_openai_client_timeout():
    """
    Mock OpenAI client that times out.
    """
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=TimeoutError("Request timed out"))
    return client


@pytest.fixture
def mock_openai_stream():
    """
    Mock OpenAI streaming response.

    Usage:
        async def test_streaming(mock_openai_stream):
            async for chunk in mock_openai_stream:
                print(chunk.choices[0].delta.content)
    """

    async def mock_stream():
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta = Mock()
            chunk.choices[0].delta.content = token
            yield chunk

    return mock_stream()


# ============================================================================
# Test Data
# ============================================================================


@pytest.fixture
def sample_chat_request():
    """
    Sample chat request payload for testing.
    """
    return {
        "prompt": "What is the capital of France?",
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
    }


@pytest.fixture
def sample_parallel_request():
    """
    Sample parallel orchestration request.
    """
    return {
        "prompt": "Write a haiku about coding",
        "providers": ["openai", "anthropic"],
        "temperature": 0.8,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_fallback_request():
    """
    Sample fallback orchestration request.
    """
    return {
        "prompt": "Explain quantum computing",
        "primary_provider": "openai",
        "fallback_providers": ["anthropic"],
        "timeout": 10,
        "temperature": 0.5,
    }


# ============================================================================
# Environment & Configuration
# ============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Mock environment variables for testing.

    Usage:
        def test_with_env(mock_env_vars):
            # Environment variables are set
            from src.utils.config import settings
            assert settings.openai_api_key == "test-key"
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_settings():
    """
    Mock application settings.
    """
    from src.utils.config import Settings

    return Settings(
        openai_api_key="test-openai-key",
        environment="testing",
        default_model="gpt-4-turbo",
        default_temperature=0.7,
        default_max_tokens=500,
        enable_caching=False,  # Disable cache in tests
    )


# ============================================================================
# Async Test Utilities
# ============================================================================


@pytest.fixture
def event_loop():
    """
    Create an instance of the default event loop for each test case.
    Required for async tests.
    """
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Time & Latency Mocking
# ============================================================================


@pytest.fixture
def mock_time():
    """
    Mock time.time() to return consistent values for latency testing.

    Usage:
        def test_latency(mock_time):
            with mock_time:
                # Latency will be consistent
                pass
    """
    with patch("time.time") as mock:
        mock.side_effect = [1000.0, 1001.234]  # 1234ms latency
        yield mock


# ============================================================================
# Database/Cache Mocks (for future use)
# ============================================================================


@pytest.fixture
def mock_cache():
    """
    Mock cache for testing caching behavior.
    """
    cache = {}

    async def get(key):
        return cache.get(key)

    async def set(key, value, ttl=None):
        cache[key] = value

    async def delete(key):
        cache.pop(key, None)

    async def clear():
        cache.clear()

    mock = Mock()
    mock.get = get
    mock.set = set
    mock.delete = delete
    mock.clear = clear

    return mock


# ============================================================================
# Cleanup
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup():
    """
    Automatically runs after each test to clean up resources.
    """
    yield
    # Cleanup code here (if needed)
    pass
