"""
API endpoint tests for the LLM orchestration service.

Tests all REST API endpoints defined in src/api/routes.py
using FastAPI's Tepixi run test tests/test_api.py -vstClient with mocked LLM providers.

Run with: pytest tests/test_api.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest

# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Test health check returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client):
        """Test health check response has correct structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "providers" in data
        assert isinstance(data["providers"], dict)

    def test_health_check_status_when_healthy(self, client):
        """Test health status is 'healthy' when providers are accessible."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "unhealthy"]
        assert "openai" in data["providers"]


class TestMonitoringEndpoints:
    """Tests for monitoring and ops endpoints."""

    def test_detailed_health_structure(self, client):
        """Ensure detailed health includes dependency checks and system metrics."""
        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in data
        assert "system" in data
        assert "openai" in data["checks"]
        assert "cpu_percent" in data["system"]
        assert "memory_percent" in data["system"]

    def test_readiness_endpoint(self, client):
        """Readiness probe should respond with a status field."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_liveness_endpoint(self, client):
        """Liveness probe should always return alive payload."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_metrics_endpoint_structure(self, client):
        """Metrics endpoint should surface core system resource gauges."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        for field in [
            "cpu_percent",
            "memory_percent",
            "memory_used_mb",
            "memory_total_mb",
            "disk_percent",
        ]:
            assert field in data


# ============================================================================
# Root Endpoint Tests
# ============================================================================


class TestRootEndpoint:
    """Tests for the / root endpoint."""

    def test_root_returns_200(self, client):
        """Test root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_structure(self, client):
        """Test root endpoint has API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "docs" in data
        assert data["docs"] == "/docs"


# ============================================================================
# Chat Completion Tests
# ============================================================================


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    @patch("src.api.routes.get_client")
    def test_chat_completion_success(self, mock_get_client, client):
        """Test successful chat completion."""
        # Mock the LLM client
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value={
                "content": "This is a test response",
                "model": "gpt-4-turbo",
                "provider": "openai",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "metrics": {
                    "latency_ms": 250.0,
                    "cost_usd": 0.001,
                    "tokens_per_second": 80.0,
                },
            }
        )
        mock_get_client.return_value = mock_client

        # Make request
        response = client.post(
            "/chat",
            json={
                "prompt": "Hello, world!",
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 500,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "content" in data
        assert "model" in data
        assert "provider" in data
        assert "usage" in data
        assert "metrics" in data
        assert "metadata" in data

        # Verify content
        assert data["content"] == "This is a test response"
        assert data["model"] == "gpt-4-turbo"
        assert data["provider"] == "openai"

    @patch("src.api.routes.get_client")
    def test_chat_completion_with_system_prompt(self, mock_get_client, client):
        """Test chat completion with system prompt."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value={
                "content": "Response with system context",
                "model": "gpt-4-turbo",
                "provider": "openai",
                "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
                "metrics": {"latency_ms": 300.0, "cost_usd": 0.0015, "tokens_per_second": 83.3},
            }
        )
        mock_get_client.return_value = mock_client

        response = client.post(
            "/chat",
            json={
                "prompt": "Explain quantum computing",
                "model": "gpt-4-turbo",
                "system_prompt": "You are a physics professor",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Response with system context"

    def test_chat_completion_invalid_prompt(self, client):
        """Test chat completion with empty prompt returns 422."""
        response = client.post(
            "/chat",
            json={
                "prompt": "",  # Invalid: empty prompt
                "model": "gpt-4-turbo",
            },
        )

        assert response.status_code == 422

    def test_chat_completion_invalid_temperature(self, client):
        """Test chat completion with invalid temperature returns 422."""
        response = client.post(
            "/chat",
            json={
                "prompt": "Test",
                "model": "gpt-4-turbo",
                "temperature": 3.0,  # Invalid: > 2.0
            },
        )

        assert response.status_code == 422

    def test_chat_completion_includes_usage_metrics(self, client):
        """Test chat completion response includes token usage."""
        with patch("src.api.routes.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(
                return_value={
                    "content": "Test response",
                    "model": "gpt-4-turbo",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                    "metrics": {"latency_ms": 200.0, "cost_usd": 0.0005, "tokens_per_second": 75.0},
                }
            )
            mock_get_client.return_value = mock_client

            response = client.post("/chat", json={"prompt": "Test", "model": "gpt-4-turbo"})

            data = response.json()
            assert data["usage"]["prompt_tokens"] == 5
            assert data["usage"]["completion_tokens"] == 10
            assert data["usage"]["total_tokens"] == 15


# ============================================================================
# Fallback Orchestration Tests
# ============================================================================


class TestFallbackEndpoint:
    """Tests for the /chat/fallback endpoint."""

    @patch("src.api.routes.fallback_orchestration")
    def test_fallback_success_no_fallback_needed(self, mock_fallback, client):
        """Test fallback endpoint when primary provider succeeds."""
        mock_fallback.return_value = {
            "content": "Primary response",
            "model": "gpt-4-turbo",
            "provider": "openai",
            "provider_used": "openai",
            "primary_success": True,
            "fallback_triggered": False,
            "usage": {"prompt_tokens": 8, "completion_tokens": 15, "total_tokens": 23},
            "metrics": {"latency_ms": 280.0, "cost_usd": 0.00092, "tokens_per_second": 53.5},
        }

        response = client.post(
            "/chat/fallback",
            json={
                "prompt": "Test fallback",
                "primary_provider": "openai",
                "primary_model": "gpt-4-turbo",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["content"] == "Primary response"
        assert data["primary_success"] is True
        assert data["fallback_triggered"] is False
        assert data["provider_used"] == "openai"

    @patch("src.api.routes.fallback_orchestration")
    def test_fallback_triggered(self, mock_fallback, client):
        """Test fallback endpoint when primary fails and fallback succeeds."""
        mock_fallback.return_value = {
            "content": "Fallback response",
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "provider_used": "openai",
            "primary_success": False,
            "fallback_triggered": True,
            "primary_error": "Primary provider timeout",
            "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            "metrics": {"latency_ms": 5150.0, "cost_usd": 0.00032, "tokens_per_second": 2.3},
        }

        response = client.post(
            "/chat/fallback",
            json={
                "prompt": "Test fallback",
                "primary_provider": "openai",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["fallback_triggered"] is True
        assert data["primary_success"] is False
        assert data["primary_error"] is not None


# ============================================================================
# Parallel Orchestration Tests (Disabled)
# ============================================================================


class TestParallelEndpoint:
    """Tests for the /chat/parallel endpoint."""

    @patch("src.api.routes.parallel_orchestration")
    def test_parallel_version_1_returns_fastest(self, mock_parallel, client):
        """Test parallel endpoint (v1) returns 200 and winner info."""
        mock_parallel.return_value = {
            "content": "Fast response",
            "winner": {"provider": "openai", "model": "gpt-4o", "latency_ms": 120.5},
            "all_responses": [
                {
                    "content": "Fast response",
                    "model": "gpt-4o",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    "metrics": {"latency_ms": 120.5, "cost_usd": 0.0005},
                },
                {
                    "content": "Slower",
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 9, "completion_tokens": 18, "total_tokens": 27},
                    "metrics": {"latency_ms": 200.0, "cost_usd": 0.0001},
                },
            ],
            "errors": None,
            "metrics": {
                "total_latency_ms": 200.0,
                "total_cost_usd": 0.0006,
                "num_providers_called": 2,
                "num_successful": 2,
                "num_failed": 0,
            },
        }

        response = client.post(
            "/chat/parallel",
            json={
                "prompt": "Test parallel",
                "version": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["winner"]["model"] == "gpt-4o"
        assert len(data["all_responses"]) == 2

    @patch("src.api.routes.parallel_orchestration")
    def test_parallel_version_2_runs_three_models(self, mock_parallel, client):
        """Test parallel endpoint (v2) aggregates three models."""
        mock_parallel.return_value = {
            "content": "Turbo response",
            "winner": {"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 90.0},
            "all_responses": [
                {
                    "content": "Turbo response",
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
                    "metrics": {"latency_ms": 90.0, "cost_usd": 0.00008},
                },
                {
                    "content": "4o response",
                    "model": "gpt-4o",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26},
                    "metrics": {"latency_ms": 150.0, "cost_usd": 0.0003},
                },
                {
                    "content": "4-turbo response",
                    "model": "gpt-4-turbo",
                    "provider": "openai",
                    "usage": {"prompt_tokens": 11, "completion_tokens": 14, "total_tokens": 25},
                    "metrics": {"latency_ms": 180.0, "cost_usd": 0.0005},
                },
            ],
            "errors": [],
            "metrics": {
                "total_latency_ms": 180.0,
                "total_cost_usd": 0.00088,
                "num_providers_called": 3,
                "num_successful": 3,
                "num_failed": 0,
            },
        }

        response = client.post(
            "/chat/parallel",
            json={
                "prompt": "Test parallel",
                "version": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["num_providers_called"] == 3


# ============================================================================
# Streaming Tests
# ============================================================================


class TestStreamEndpoint:
    """Tests for the /chat/stream endpoint."""

    @patch("src.api.routes.streaming_orchestration")
    def test_stream_returns_200(self, mock_stream, client):
        """Test streaming endpoint returns 200 OK."""

        async def mock_generator():
            """Mock async generator for streaming tokens."""
            for token in ["Hello", " ", "world", "!"]:
                yield token

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/chat/stream",
            json={
                "prompt": "Stream test",
                "provider": "openai",
                "model": "gpt-4-turbo",
            },
        )

        assert response.status_code == 200

    @patch("src.api.routes.streaming_orchestration")
    def test_stream_content_type(self, mock_stream, client):
        """Test streaming endpoint returns text/event-stream content type."""

        async def mock_generator():
            yield "test"

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/chat/stream",
            json={
                "prompt": "Test",
                "provider": "openai",
            },
        )

        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


# ============================================================================
# Models Endpoint Tests
# ============================================================================


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    def test_models_returns_200(self, client):
        """Test models endpoint returns 200 OK."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_response_structure(self, client):
        """Test models endpoint returns list of models."""
        response = client.get("/models")
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_include_metadata(self, client):
        """Test each model includes required metadata."""
        response = client.get("/models")
        data = response.json()

        first_model = data["models"][0]
        assert "id" in first_model
        assert "provider" in first_model
        assert "max_tokens" in first_model
        assert "cost_per_1m_prompt_tokens" in first_model
        assert "cost_per_1m_completion_tokens" in first_model
        assert "supports_streaming" in first_model


# ============================================================================
# Statistics Endpoint Tests
# ============================================================================


class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_stats_returns_200(self, client):
        """Test stats endpoint returns 200 OK."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_response_structure(self, client):
        """Test stats endpoint has correct structure."""
        response = client.get("/stats")
        data = response.json()

        assert "total_requests" in data
        assert "requests_by_provider" in data
        assert "average_latency_ms" in data
        assert "total_cost_usd" in data
        assert "error_rate" in data
        assert "uptime_seconds" in data

    def test_stats_data_types(self, client):
        """Test stats endpoint returns correct data types."""
        response = client.get("/stats")
        data = response.json()

        assert isinstance(data["total_requests"], int)
        assert isinstance(data["requests_by_provider"], dict)
        assert isinstance(data["average_latency_ms"], (int, float))
        assert isinstance(data["total_cost_usd"], (int, float))
        assert isinstance(data["error_rate"], (int, float))
        assert isinstance(data["uptime_seconds"], (int, float))


# ============================================================================
# Async Tests
# ============================================================================


class TestAsyncEndpoints:
    """Async tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_async_health_check(self, async_client):
        """Test health check using async client."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    @patch("src.api.routes.get_client")
    async def test_async_chat_completion(self, mock_get_client, async_client):
        """Test chat completion using async client."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value={
                "content": "Async response",
                "model": "gpt-4-turbo",
                "provider": "openai",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "metrics": {"latency_ms": 200.0, "cost_usd": 0.0005, "tokens_per_second": 50.0},
            }
        )
        mock_get_client.return_value = mock_client

        response = await async_client.post(
            "/chat",
            json={"prompt": "Async test", "model": "gpt-4-turbo"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Async response"
