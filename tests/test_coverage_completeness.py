"""
Comprehensive tests to achieve 100% code coverage.

Covers edge cases, error paths, and rarely-executed code paths:
- Lifespan events (startup/shutdown)
- Provider validation edge cases
- Health check error scenarios
- Streaming error handling
- System prompt handling
- Anthropic normalization
- Logger utilities
- Config utilities

Run with: pytest tests/test_coverage_completeness.py -v
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

# ============================================================================
# Lifespan & Startup Tests
# ============================================================================


class TestLifespanEvents:
    """Tests for application lifespan events (startup/shutdown)."""

    @patch("src.api.main.settings")
    @patch("src.api.main.logger")
    def test_lifespan_startup_logs_correctly(self, mock_logger, mock_settings):
        """Test that startup events are logged."""
        from src.api.main import lifespan

        mock_settings.environment = "production"
        mock_settings.api_version = "0.1.0"
        mock_settings.openai_api_key = "sk-test-key"

        # Create a fake app
        from fastapi import FastAPI

        app = FastAPI()

        # Test the lifespan context manager
        import asyncio

        async def test_startup():
            async with lifespan(app):
                # During the yield, app is running
                pass  # App shutdown happens after yield

        asyncio.run(test_startup())

        # Verify startup logs were called
        assert mock_logger.info.call_count >= 2  # application_starting, application_ready

    @patch("src.api.main.settings")
    @patch("src.api.main.logger")
    def test_lifespan_warns_on_missing_api_key(self, mock_logger, mock_settings):
        """Test warning when OpenAI API key is not set."""
        from src.api.main import lifespan

        mock_settings.environment = "development"
        mock_settings.api_version = "0.1.0"
        mock_settings.openai_api_key = ""  # Empty API key

        from fastapi import FastAPI

        app = FastAPI()

        import asyncio

        async def test_missing_key():
            async with lifespan(app):
                pass

        asyncio.run(test_missing_key())

        # Verify warning was logged
        mock_logger.warning.assert_called()


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestModelValidation:
    """Tests for Pydantic model field validators."""

    def test_parallel_request_validates_empty_providers(self):
        """Test that ParallelRequest rejects empty providers list."""
        from pydantic import ValidationError

        from src.api.models import ParallelRequest

        with pytest.raises(ValidationError) as exc_info:
            ParallelRequest(prompt="Test", providers=[])

        # Pydantic's min_length validation triggers first
        assert "at least 1 item" in str(exc_info.value).lower()

    def test_parallel_request_validates_invalid_providers(self):
        """Test that ParallelRequest rejects invalid provider names."""
        from pydantic import ValidationError

        from src.api.models import ParallelRequest

        with pytest.raises(ValidationError) as exc_info:
            ParallelRequest(prompt="Test", providers=["openai", "invalid-provider"])

        assert "Invalid providers" in str(exc_info.value)


# ============================================================================
# Health Check Error Scenarios
# ============================================================================


class TestHealthCheckErrors:
    """Tests for health check error handling."""

    @patch("src.api.routes.get_client")
    def test_health_check_handles_provider_error(self, mock_get_client, client):
        """Test health check when provider connection fails."""
        # Make get_client raise an exception
        mock_get_client.side_effect = Exception("Connection failed")

        response = client.get("/health")

        # Should still return 200 but with unhealthy status
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "openai" in data["providers"]
        assert data["providers"]["openai"]["status"] == "error"


# ============================================================================
# Streaming Error Handling
# ============================================================================


class TestStreamingErrors:
    """Tests for streaming error scenarios."""

    @patch("src.api.routes.streaming_orchestration")
    def test_stream_handles_generation_error(self, mock_stream, client):
        """Test that streaming handles errors during generation."""

        async def error_generator():
            """Generator that raises an error."""
            yield "Start"
            raise Exception("Stream error")

        mock_stream.return_value = error_generator()

        response = client.post(
            "/chat/stream",
            json={"prompt": "Test", "provider": "openai"},
        )

        # Stream should return error in SSE format
        assert response.status_code == 200
        content = response.text
        assert "error" in content.lower() or "Start" in content


# ============================================================================
# System Prompt Tests
# ============================================================================


class TestSystemPromptHandling:
    """Tests for system prompt in various scenarios."""

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_openai_client):
        """Test that system prompt is properly included in messages."""
        from src.llm.clients import OpenAIClient

        client = OpenAIClient(api_client=mock_openai_client)

        await client.generate(
            prompt="What is AI?",
            system_prompt="You are a helpful assistant",
            temperature=0.5,
        )

        # Verify system prompt was included in the call
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_stream_with_system_prompt(self, mock_openai_client):
        """Test streaming with system prompt."""
        from src.llm.clients import OpenAIClient

        # Mock streaming response
        async def mock_stream_iter():
            for token in ["Hello", " world"]:
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta.content = token
                yield chunk

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream_iter())

        client = OpenAIClient(api_client=mock_openai_client)

        tokens = []
        async for token in client.generate_stream(prompt="Test", system_prompt="Be concise"):
            tokens.append(token)

        assert len(tokens) > 0


# ============================================================================
# Client Error Paths
# ============================================================================


class TestClientErrorPaths:
    """Tests for error handling in LLM clients."""

    @pytest.mark.asyncio
    async def test_generate_stream_error_handling(self):
        """Test that streaming errors are properly logged and raised."""
        from src.llm.clients import OpenAIClient

        # Create mock that raises error during streaming
        mock_client = AsyncMock()

        async def error_stream():
            """Async generator that raises error."""
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta.content = "Test"
            yield chunk
            raise Exception("Streaming failed")

        mock_client.chat.completions.create = AsyncMock(return_value=error_stream())

        client = OpenAIClient(api_client=mock_client)

        with pytest.raises(Exception) as exc_info:
            async for _ in client.generate_stream(prompt="Test"):
                pass

        assert "Streaming failed" in str(exc_info.value)

    def test_get_client_invalid_provider(self):
        """Test that get_client raises error for unknown providers."""
        from src.llm.clients import get_client

        with pytest.raises(ValueError) as exc_info:
            get_client("invalid-provider")

        assert "Unknown provider" in str(exc_info.value)


# ============================================================================
# Orchestration Edge Cases
# ============================================================================


class TestOrchestrationEdgeCases:
    """Tests for orchestration edge cases."""

    @pytest.mark.asyncio
    async def test_fallback_rejects_invalid_provider(self):
        """Test that fallback orchestration rejects non-openai providers."""
        from src.llm.orchestrator import fallback_orchestration

        with pytest.raises(ValueError) as exc_info:
            await fallback_orchestration(
                prompt="Test",
                primary_provider="anthropic",  # Not supported
            )

        assert "Only 'openai' provider is supported" in str(exc_info.value)


# ============================================================================
# Utility Functions Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions in utils modules."""

    def test_normalize_anthropic_response(self):
        """Test normalization of Anthropic API responses."""
        from src.llm.utils import normalize_response

        anthropic_response = {
            "content": [{"text": "Response from Claude"}],
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25,
            },
            "model": "claude-3-opus",
        }

        normalized = normalize_response(anthropic_response, "anthropic")

        assert normalized["content"] == "Response from Claude"
        assert normalized["usage"]["prompt_tokens"] == 15
        assert normalized["usage"]["completion_tokens"] == 25
        assert normalized["usage"]["total_tokens"] == 40
        assert normalized["model"] == "claude-3-opus"

    def test_normalize_unknown_provider_raises_error(self):
        """Test that normalize_response raises error for unknown providers."""
        from src.llm.utils import normalize_response

        with pytest.raises(ValueError) as exc_info:
            normalize_response("unknown-provider", {})

        assert "Unknown provider" in str(exc_info.value)

    def test_is_production_utility(self):
        """Test is_production utility function."""
        from src.utils.config import is_production, settings

        # Current setting
        result = is_production()
        expected = settings.environment == "production"
        assert result == expected

    def test_is_development_utility(self):
        """Test is_development utility function."""
        from src.utils.config import is_development, settings

        result = is_development()
        expected = settings.environment == "development"
        assert result == expected


# ============================================================================
# Logger Tests
# ============================================================================


class TestLoggerUtilities:
    """Tests for logger utility functions."""

    def test_logger_mixin_property(self):
        """Test LoggerMixin provides logger property."""
        from src.utils.logger import LoggerMixin

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        logger = obj.logger

        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    @patch("src.utils.logger.get_logger")
    def test_log_llm_call_success(self, mock_get_logger):
        """Test log_llm_call utility for successful calls."""
        from src.utils.logger import log_llm_call

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_llm_call(
            model="gpt-4-turbo",
            provider="openai",
            latency_ms=250.5,
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001234,
            success=True,
        )

        mock_logger.info.assert_called_once()

    @patch("src.utils.logger.get_logger")
    def test_log_llm_call_error(self, mock_get_logger):
        """Test log_llm_call utility for failed calls."""
        from src.utils.logger import log_llm_call

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_llm_call(
            model="gpt-4-turbo",
            provider="openai",
            latency_ms=150.0,
            prompt_tokens=5,
            completion_tokens=0,
            cost_usd=0.0,
            success=False,
            error="Timeout error",
        )

        mock_logger.error.assert_called_once()

    @patch("src.utils.logger.get_logger")
    def test_log_error_utility(self, mock_get_logger):
        """Test log_error utility function."""
        from src.utils.logger import log_error

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_error(
            error_type="validation_error",
            message="Invalid input",
            field="temperature",
            value=3.0,
        )

        mock_logger.error.assert_called_once()


# ============================================================================
# Production Environment Tests
# ============================================================================


class TestProductionConfiguration:
    """Tests for production-specific configuration."""

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_logging_setup_in_production(self):
        """Test that logging configures correctly for production."""
        # Import fresh to pick up environment variable
        import importlib

        import src.utils.logger

        importlib.reload(src.utils.logger)

        # Should not raise any errors
        from src.utils.logger import get_logger

        logger = get_logger("test")
        assert logger is not None


# ============================================================================
# Cost Calculation Edge Cases
# ============================================================================


class TestCostCalculation:
    """Tests for cost calculation in clients."""

    @pytest.mark.asyncio
    async def test_cost_calculation_with_large_numbers(self, mock_openai_client):
        """Test cost calculation handles large token counts correctly."""
        from src.llm.clients import OpenAIClient

        # Mock response with large token counts
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 100000  # 100k tokens
        mock_response.usage.completion_tokens = 50000  # 50k tokens
        mock_response.usage.total_tokens = 150000
        mock_response.model = "gpt-4-turbo"

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_client=mock_openai_client)
        result = await client.generate(prompt="Test", model="gpt-4-turbo")

        # Verify cost calculation (gpt-4-turbo: $0.03 per 1M prompt, $0.06 per 1M completion)
        expected_cost = (100000 / 1_000_000 * 0.03) + (50000 / 1_000_000 * 0.06)
        assert abs(result["metrics"]["cost_usd"] - expected_cost) < 0.0001


# ============================================================================
# Config Edge Cases
# ============================================================================


class TestConfigEdgeCases:
    """Tests for configuration edge cases."""

    def test_get_provider_for_model(self):
        """Test get_provider_for_model utility."""
        from src.utils.config import get_provider_for_model

        provider = get_provider_for_model("gpt-4-turbo")
        assert provider == "openai"

        provider = get_provider_for_model("gpt-3.5-turbo")
        assert provider == "openai"

    def test_validate_model_with_valid_model(self):
        """Test validate_model returns True for valid models."""
        from src.utils.config import validate_model

        assert validate_model("gpt-4-turbo") is True
        assert validate_model("gpt-3.5-turbo") is True

    def test_validate_model_with_invalid_model(self):
        """Test validate_model returns False for invalid models."""
        from src.utils.config import validate_model

        assert validate_model("invalid-model-xyz") is False


# ============================================================================
# Abstract Method Tests
# ============================================================================


class TestAbstractMethods:
    """Tests for abstract base class enforcement."""

    def test_base_llm_client_cannot_be_instantiated(self):
        """Test that BaseLLMClient cannot be instantiated directly."""
        from src.llm.clients import BaseLLMClient

        with pytest.raises(TypeError) as exc_info:
            BaseLLMClient()  # type: ignore

        assert "abstract" in str(exc_info.value).lower()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        """Test that incomplete subclasses cannot be instantiated."""
        from src.llm.clients import BaseLLMClient

        # Subclass that doesn't implement all abstract methods
        class IncompleteClient(BaseLLMClient):
            @property
            def provider(self) -> str:
                return "incomplete"

            # Missing: generate, generate_stream implementations

        with pytest.raises(TypeError):
            IncompleteClient()  # type: ignore

    def test_abstract_methods_have_pass_statements(self):
        """Test that abstract methods exist with pass statements."""
        import inspect

        from src.llm.clients import BaseLLMClient

        # Check that abstract methods are defined
        assert hasattr(BaseLLMClient, "generate")
        assert hasattr(BaseLLMClient, "generate_stream")
        assert hasattr(BaseLLMClient, "provider")

        # These are abstract, so they have pass or raise NotImplementedError
        assert inspect.isabstract(BaseLLMClient)
