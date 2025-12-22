"""
LLM orchestration and client tests using Test-Driven Development (TDD).

These tests define the expected behavior of LLM clients and orchestration
logic BEFORE implementation.

Run with: pytest tests/test_llm.py -v
"""

import asyncio
from unittest.mock import Mock

import pytest

# ============================================================================
# OpenAI Client Tests
# ============================================================================


class TestOpenAIClient:
    """Tests for OpenAI API client."""

    @pytest.mark.asyncio
    async def test_openai_generate_success(self, mock_openai_client):
        """Test successful OpenAI text generation."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client)
        response = await client.generate(prompt="Hello, world!", temperature=0.7, max_tokens=100)

        assert response is not None
        assert "content" in response
        assert isinstance(response["content"], str)
        assert len(response["content"]) > 0

    @pytest.mark.asyncio
    async def test_openai_generate_includes_metadata(self, mock_openai_client):
        """Test OpenAI response includes token usage and model info."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client)
        response = await client.generate(prompt="Test")

        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "completion_tokens" in response["usage"]
        assert "model" in response

    @pytest.mark.asyncio
    async def test_openai_generate_respects_temperature(self, mock_openai_client):
        """Test temperature parameter is passed correctly."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client)
        await client.generate(prompt="Test", temperature=0.3)

        # Verify mock was called with correct parameters
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_openai_generate_handles_error(self, mock_openai_client_error):
        """Test error handling for OpenAI API failures."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client_error)

        with pytest.raises(Exception) as exc_info:
            await client.generate(prompt="Test")

        assert "OpenAI API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_generate_handles_timeout(self, mock_openai_client_timeout):
        """Test timeout handling."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client_timeout)

        with pytest.raises(asyncio.TimeoutError):
            await client.generate(prompt="Test", timeout=5)

    @pytest.mark.asyncio
    async def test_openai_stream_generates_tokens(self, mock_openai_stream):
        """Test OpenAI streaming returns tokens incrementally."""
        from unittest.mock import AsyncMock

        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=Mock())
        client._client.chat.completions.create = AsyncMock(return_value=mock_openai_stream)

        tokens = []
        async for token in client.generate_stream(prompt="Test"):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


# ============================================================================
# Fallback Orchestration Tests
# ============================================================================


class TestFallbackOrchestration:
    """Tests for fallback orchestration (OpenAI-only with model fallback)."""

    @pytest.mark.asyncio
    async def test_fallback_succeeds_with_primary_model(self, mock_openai_client):
        """Test fallback orchestration succeeds with primary model."""
        from unittest.mock import patch

        from src.llm.orchestrator import fallback_orchestration

        with patch("src.llm.clients.AsyncOpenAI", return_value=mock_openai_client):
            result = await fallback_orchestration(prompt="Test prompt", primary_model="gpt-4-turbo")

        assert result["content"] is not None
        assert result["primary_success"] is True
        assert result["fallback_triggered"] is False
        assert result["is_default_message"] is False
        assert result["provider_used"] == "openai"

    @pytest.mark.asyncio
    async def test_fallback_uses_gpt35_when_primary_fails(
        self, mock_openai_client_error, mock_openai_client
    ):
        """Test fallback to gpt-3.5-turbo when primary model fails."""
        from unittest.mock import AsyncMock, patch

        from src.llm.orchestrator import fallback_orchestration

        call_count = 0

        async def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (primary) fails
                raise Exception("Primary model failed")
            else:
                # Second call (fallback) succeeds
                return {
                    "content": "Fallback response",
                    "model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 15, "total_tokens": 23},
                    "metrics": {"latency_ms": 150, "cost_usd": 0.00002},
                }

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=mock_generate_side_effect)

        with patch("src.llm.orchestrator.get_client", return_value=mock_client):
            result = await fallback_orchestration(prompt="Test prompt", primary_model="gpt-4-turbo")

        assert call_count == 2  # Primary + fallback
        assert result["content"] == "Fallback response"
        assert result["primary_success"] is False
        assert result["fallback_triggered"] is True
        assert result["is_default_message"] is False

    @pytest.mark.asyncio
    async def test_fallback_returns_default_message_when_all_fail(self):
        """Test default message returned when all models fail."""
        from unittest.mock import AsyncMock, patch

        from src.llm.orchestrator import fallback_orchestration

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("All models failed"))

        with patch("src.llm.orchestrator.get_client", return_value=mock_client):
            result = await fallback_orchestration(prompt="Test prompt")

        assert result["is_default_message"] is True
        assert "Service temporarily unavailable" in result["content"]
        assert result["provider_used"] == "default"
        assert result["primary_success"] is False
        assert result["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_fallback_includes_error_details(self):
        """Test fallback response includes error details from failed attempts."""
        from unittest.mock import AsyncMock, patch

        from src.llm.orchestrator import fallback_orchestration

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("API rate limit exceeded"))

        with patch("src.llm.orchestrator.get_client", return_value=mock_client):
            result = await fallback_orchestration(prompt="Test prompt")

        assert "primary_error" in result
        assert "API rate limit exceeded" in result["primary_error"]


# ============================================================================
# Base LLM Client Tests (Interface)
# ============================================================================


class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract interface."""

    def test_base_client_enforces_generate_method(self):
        """Test BaseLLMClient requires generate() implementation."""
        from src.llm.clients import BaseLLMClient

        with pytest.raises(TypeError):
            # Cannot instantiate abstract class without implementing generate
            BaseLLMClient()

    def test_base_client_enforces_provider_property(self):
        """Test BaseLLMClient requires provider property."""
        from src.llm.clients import BaseLLMClient

        class IncompleteClient(BaseLLMClient):
            async def generate(self, prompt):
                pass

        with pytest.raises(TypeError):
            IncompleteClient()


# ============================================================================
# Parallel Orchestration Tests - DISABLED (OpenAI-only mode)
# ============================================================================
# Parallel orchestration requires multiple providers and is not available
# in OpenAI-only mode. Tests removed.


# ============================================================================
# Streaming Orchestration Tests
# ============================================================================


class TestStreamingOrchestration:
    """Tests for streaming orchestration."""

    @pytest.mark.asyncio
    async def test_streaming_yields_tokens(self, mock_openai_stream):
        """Test streaming orchestration yields tokens incrementally."""
        from src.llm.orchestrator import streaming_orchestration

        tokens = []
        async for token in streaming_orchestration(prompt="Test", provider="openai"):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_streaming_handles_errors_gracefully(self, mock_openai_client_error):
        """Test streaming handles errors without breaking."""
        from unittest.mock import patch

        from src.llm.orchestrator import streaming_orchestration

        mock_client = Mock()
        mock_client.generate_stream = Mock(side_effect=RuntimeError("Streaming failed"))

        with patch("src.llm.orchestrator.get_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Streaming failed"):
                async for _token in streaming_orchestration(prompt="Test", provider="openai"):
                    pass


# ============================================================================
# Cost Calculation Tests
# ============================================================================


class TestCostCalculation:
    """Tests for LLM cost calculation."""

    def test_calculate_openai_cost(self):
        """Test OpenAI cost calculation."""
        from src.llm.utils import calculate_cost

        cost = calculate_cost(model="gpt-4-turbo", prompt_tokens=1000, completion_tokens=500)

        # GPT-4-turbo: $0.03 per 1M prompt, $0.06 per 1M completion
        expected = (1000 / 1_000_000 * 0.03) + (500 / 1_000_000 * 0.06)
        assert abs(cost - expected) < 0.0001

    def test_calculate_openai_mini_cost(self):
        """Test cost calculation for gpt-4o-mini."""
        from src.llm.utils import calculate_cost

        cost = calculate_cost(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)

        expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.6)
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_unknown_model_raises_error(self):
        """Test unknown model raises error."""
        from src.llm.utils import calculate_cost

        with pytest.raises(ValueError) as exc_info:
            calculate_cost(model="unknown-model", prompt_tokens=100, completion_tokens=50)

        assert "Unknown model" in str(exc_info.value)


# ============================================================================
# Latency Tracking Tests
# ============================================================================


class TestLatencyTracking:
    """Tests for latency measurement."""

    @pytest.mark.asyncio
    async def test_track_latency_decorator(self, mock_openai_client, mock_time):
        """Test latency tracking decorator."""
        from src.llm.utils import track_latency

        @track_latency
        async def mock_llm_call():
            await asyncio.sleep(0.1)
            return {"content": "Test"}

        result = await mock_llm_call()

        assert "latency_ms" in result
        assert result["latency_ms"] > 0


# ============================================================================
# Response Normalization Tests
# ============================================================================


class TestResponseNormalization:
    """Tests for normalizing responses from different providers."""

    def test_normalize_openai_response(self):
        """Test OpenAI response normalization."""
        from src.llm.utils import normalize_response

        raw_response = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        normalized = normalize_response(raw_response, provider="openai")

        assert normalized["content"] == "Hello"
        assert normalized["usage"]["prompt_tokens"] == 10


# ============================================================================
# Model Configuration Tests
# ============================================================================


class TestModelConfiguration:
    """Tests for model configuration and metadata."""

    def test_get_model_config(self):
        """Test retrieving model configuration."""
        from src.utils.config import get_model_config

        config = get_model_config("gpt-4-turbo")

        assert config["provider"] == "openai"
        assert "max_tokens" in config
        assert "cost_per_1m_prompt_tokens" in config
        assert "timeout" in config

    def test_list_available_models(self):
        """Test listing all available models."""
        from src.utils.config import list_models

        models = list_models()

        assert len(models) > 0
        assert any(m["id"] == "gpt-4-turbo" for m in models)
        assert any(m["id"] == "gpt-3.5-turbo" for m in models)
        # Should now have Claude models for testing
        assert any("claude" in m["id"] for m in models)

    def test_validate_model_exists(self):
        """Test model validation."""
        from src.utils.config import validate_model

        assert validate_model("gpt-4-turbo") is True
        assert validate_model("unknown-model") is False
