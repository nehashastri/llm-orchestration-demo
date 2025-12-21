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
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=Mock())
        client._client.chat.completions.create = Mock(return_value=mock_openai_stream)

        tokens = []
        async for token in client.generate_stream(prompt="Test"):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


# ============================================================================
# Anthropic Client Tests
# ============================================================================


class TestAnthropicClient:
    """Tests for Anthropic API client."""

    @pytest.mark.asyncio
    async def test_anthropic_generate_success(self, mock_anthropic_client):
        """Test successful Anthropic text generation."""
        from src.llm.clients import AnthropicClient

        client = AnthropicClient(api_client=mock_anthropic_client)
        response = await client.generate(prompt="Hello, Claude!", temperature=0.7, max_tokens=100)

        assert response is not None
        assert "content" in response
        assert isinstance(response["content"], str)
        assert len(response["content"]) > 0

    @pytest.mark.asyncio
    async def test_anthropic_generate_includes_metadata(self, mock_anthropic_client):
        """Test Anthropic response includes token usage."""
        from src.llm.clients import AnthropicClient

        client = AnthropicClient(api_client=mock_anthropic_client)
        response = await client.generate(prompt="Test")

        assert "usage" in response
        assert "input_tokens" in response["usage"]
        assert "output_tokens" in response["usage"]
        assert "model" in response

    @pytest.mark.asyncio
    async def test_anthropic_generate_normalizes_response(self, mock_anthropic_client):
        """Test Anthropic responses are normalized to match OpenAI format."""
        from src.llm.clients import AnthropicClient

        client = AnthropicClient(api_client=mock_anthropic_client)
        response = await client.generate(prompt="Test")

        # Should have same structure as OpenAI response
        assert "content" in response
        assert "usage" in response
        assert "model" in response

    @pytest.mark.asyncio
    async def test_anthropic_generate_handles_error(self, mock_anthropic_client_error):
        """Test error handling for Anthropic API failures."""
        from src.llm.clients import AnthropicClient

        client = AnthropicClient(api_client=mock_anthropic_client_error)

        with pytest.raises(Exception) as exc_info:
            await client.generate(prompt="Test")

        assert "Anthropic API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_anthropic_stream_generates_tokens(self, mock_anthropic_stream):
        """Test Anthropic streaming returns tokens incrementally."""
        from src.llm.clients import AnthropicClient

        client = AnthropicClient(api_client=Mock())
        client._client.messages.create = Mock(return_value=mock_anthropic_stream)

        tokens = []
        async for token in client.generate_stream(prompt="Test"):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


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
# Parallel Orchestration Tests
# ============================================================================


class TestParallelOrchestration:
    """Tests for parallel LLM orchestration."""

    @pytest.mark.asyncio
    async def test_parallel_calls_multiple_providers(self, mock_all_llm_clients):
        """Test parallel orchestration calls multiple providers."""
        from src.llm.orchestrator import parallel_orchestration

        openai_client, anthropic_client = mock_all_llm_clients

        results = await parallel_orchestration(
            prompt="Test prompt", providers=["openai", "anthropic"]
        )

        assert len(results) == 2
        assert results[0]["provider"] in ["openai", "anthropic"]
        assert results[1]["provider"] in ["openai", "anthropic"]

    @pytest.mark.asyncio
    async def test_parallel_returns_fastest_first(self, mock_all_llm_clients):
        """Test parallel orchestration returns fastest response first."""
        from src.llm.orchestrator import parallel_orchestration

        # Configure mock to have different latencies
        openai_client, anthropic_client = mock_all_llm_clients

        results = await parallel_orchestration(prompt="Test", providers=["openai", "anthropic"])

        # Results should be ordered by latency (fastest first)
        assert results[0]["latency_ms"] <= results[1]["latency_ms"]

    @pytest.mark.asyncio
    async def test_parallel_handles_partial_failure(
        self, mock_openai_client, mock_anthropic_client_error
    ):
        """Test parallel orchestration continues if one provider fails."""
        from src.llm.orchestrator import parallel_orchestration

        results = await parallel_orchestration(prompt="Test", providers=["openai", "anthropic"])

        # Should have one successful result
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) == 1
        assert successful_results[0]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_parallel_aggregates_costs(self, mock_all_llm_clients):
        """Test parallel orchestration aggregates costs."""
        from src.llm.orchestrator import parallel_orchestration

        results = await parallel_orchestration(prompt="Test", providers=["openai", "anthropic"])

        total_cost = sum(r.get("cost_usd", 0) for r in results)
        assert total_cost > 0

    @pytest.mark.asyncio
    async def test_parallel_uses_asyncio_gather(self, mock_all_llm_clients):
        """Test parallel orchestration uses asyncio.gather for concurrency."""
        from unittest.mock import patch

        from src.llm.orchestrator import parallel_orchestration

        with patch("asyncio.gather", wraps=asyncio.gather) as mock_gather:
            await parallel_orchestration(prompt="Test", providers=["openai", "anthropic"])

            # Verify asyncio.gather was called
            assert mock_gather.called


# ============================================================================
# Fallback Orchestration Tests
# ============================================================================


class TestFallbackOrchestration:
    """Tests for fallback LLM orchestration."""

    @pytest.mark.asyncio
    async def test_fallback_uses_primary_when_available(self, mock_openai_client):
        """Test fallback uses primary provider when it succeeds."""
        from src.llm.orchestrator import fallback_orchestration

        result = await fallback_orchestration(
            prompt="Test", primary_provider="openai", fallback_providers=["anthropic"]
        )

        assert result["provider"] == "openai"
        assert result["fallback_triggered"] is False

    @pytest.mark.asyncio
    async def test_fallback_triggers_on_primary_failure(
        self, mock_openai_client_error, mock_anthropic_client
    ):
        """Test fallback triggers when primary provider fails."""
        from src.llm.orchestrator import fallback_orchestration

        result = await fallback_orchestration(
            prompt="Test", primary_provider="openai", fallback_providers=["anthropic"]
        )

        assert result["provider"] == "anthropic"
        assert result["fallback_triggered"] is True
        assert "primary_error" in result

    @pytest.mark.asyncio
    async def test_fallback_tries_multiple_fallbacks(
        self, mock_openai_client_error, mock_anthropic_client_error
    ):
        """Test fallback tries multiple fallback providers in order."""
        from src.llm.orchestrator import fallback_orchestration

        # Assuming we add a third provider later
        with pytest.raises(Exception) as exc_info:
            await fallback_orchestration(
                prompt="Test", primary_provider="openai", fallback_providers=["anthropic"]
            )

        assert "All providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_respects_timeout(
        self, mock_openai_client_timeout, mock_anthropic_client
    ):
        """Test fallback triggers on timeout."""
        from src.llm.orchestrator import fallback_orchestration

        result = await fallback_orchestration(
            prompt="Test", primary_provider="openai", fallback_providers=["anthropic"], timeout=5
        )

        assert result["provider"] == "anthropic"
        assert result["fallback_triggered"] is True


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
        from src.llm.orchestrator import streaming_orchestration

        with pytest.raises(Exception):
            async for token in streaming_orchestration(prompt="Test", provider="openai"):
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

    def test_calculate_anthropic_cost(self):
        """Test Anthropic cost calculation."""
        from src.llm.utils import calculate_cost

        cost = calculate_cost(model="claude-3-opus", prompt_tokens=1000, completion_tokens=500)

        # Claude 3 Opus: $0.015 per 1M input, $0.075 per 1M output
        expected = (1000 / 1_000_000 * 0.015) + (500 / 1_000_000 * 0.075)
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

    def test_normalize_anthropic_response(self):
        """Test Anthropic response normalization."""
        from src.llm.utils import normalize_response

        raw_response = {
            "content": [{"text": "Hello from Claude"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        normalized = normalize_response(raw_response, provider="anthropic")

        assert normalized["content"] == "Hello from Claude"
        assert normalized["usage"]["prompt_tokens"] == 10
        assert normalized["usage"]["completion_tokens"] == 20


# ============================================================================
# Model Configuration Tests
# ============================================================================


class TestModelConfiguration:
    """Tests for model configuration and metadata."""

    def test_get_model_config(self):
        """Test retrieving model configuration."""
        from src.llm.config import get_model_config

        config = get_model_config("gpt-4-turbo")

        assert config["provider"] == "openai"
        assert "max_tokens" in config
        assert "cost_per_1m_prompt" in config
        assert "timeout" in config

    def test_list_available_models(self):
        """Test listing all available models."""
        from src.llm.config import list_models

        models = list_models()

        assert len(models) > 0
        assert any(m["id"] == "gpt-4-turbo" for m in models)
        assert any(m["id"] == "claude-3-opus" for m in models)

    def test_validate_model_exists(self):
        """Test model validation."""
        from src.llm.config import validate_model

        assert validate_model("gpt-4-turbo") is True
        assert validate_model("unknown-model") is False
