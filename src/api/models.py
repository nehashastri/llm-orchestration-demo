"""
Pydantic models for API request/response validation.

Provides type-safe data models with automatic validation,
documentation, and serialization.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Request Models
# ============================================================================


class ChatRequest(BaseModel):
    """
    Request model for chat completion endpoint.

    Example:
        {
            "prompt": "Explain quantum computing",
            "model": "gpt-4-turbo",
            "temperature": 0.7,
            "max_tokens": 500
        }
    """

    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt or question")
    model: str = Field(default="gpt-4-turbo", description="LLM model identifier")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens to generate")
    system_prompt: str | None = Field(
        default=None, max_length=5000, description="System instructions (optional)"
    )


class ParallelRequest(BaseModel):
    """
    Request model for parallel orchestration endpoint.

    Example:
        {
            "prompt": "Write a haiku",
            "version": 1,
            "temperature": 0.8
        }
    """

    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    version: Literal[1, 2] = Field(
        default=1,
        description="Parallel strategy version (1=race 4o vs 4o-mini, 2=all three models)",
    )
    # Optional explicit models for OpenAI; alternatively choose by providers
    models: list[str] | None = Field(
        default=None,
        description="Optional explicit list of OpenAI models to fan out to",
    )
    # Providers to fan out to; must be non-empty and valid
    providers: list[str] = Field(
        default=["openai"],
        min_length=1,
        description="Providers to call in parallel (e.g., 'openai', 'anthropic')",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens")
    system_prompt: str | None = Field(default=None, description="System instructions")

    @field_validator("providers")
    def _validate_providers(cls, v: list[str]) -> list[str]:
        allowed = {"openai", "anthropic"}
        invalid = [p for p in v if p not in allowed]
        if invalid:
            # Match tests expecting this message substring
            raise ValueError(f"Invalid providers: {', '.join(invalid)}")
        return v


class FallbackRequest(BaseModel):
    """
    Request model for fallback orchestration endpoint.

    Example:
        {
            "prompt": "What is AI?",
            "primary_provider": "openai",
            "primary_model": "gpt-4-turbo",
            "timeout": 10
        }
    """

    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    primary_provider: str = Field(
        default="openai", description="Primary provider (only openai supported)"
    )
    fallback_providers: list[str] = Field(
        default=[], description="Deprecated: fallback handled automatically"
    )
    primary_model: str | None = Field(
        default=None, description="Specific model for primary provider"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens")
    timeout: int = Field(default=30, ge=1, le=120, description="Request timeout in seconds")
    system_prompt: str | None = Field(default=None, description="System instructions")


class StreamRequest(BaseModel):
    """
    Request model for streaming endpoint.

    Example:
        {
            "prompt": "Tell me a story",
            "provider": "openai",
            "temperature": 0.9
        }
    """

    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    provider: str = Field(default="openai", description="Provider for streaming")
    model: str | None = Field(default=None, description="Specific model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens")
    system_prompt: str | None = Field(default=None, description="System instructions")


# ============================================================================
# Response Models
# ============================================================================


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(description="Input tokens used")
    completion_tokens: int = Field(description="Output tokens generated")
    total_tokens: int = Field(description="Total tokens used")


class Metrics(BaseModel):
    """Performance metrics."""

    latency_ms: float = Field(description="Request latency in milliseconds")
    cost_usd: float = Field(description="Estimated cost in USD")


class Metadata(BaseModel):
    """Request metadata."""

    request_id: str = Field(description="Unique request identifier")
    timestamp: str = Field(description="ISO timestamp")


class ChatResponse(BaseModel):
    """
    Response model for chat completion.

    Example:
        {
            "content": "Quantum computing uses...",
            "model": "gpt-4-turbo",
            "provider": "openai",
            "usage": {...},
            "metrics": {...},
            "metadata": {...}
        }
    """

    content: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    provider: str = Field(description="Provider name")
    usage: TokenUsage = Field(description="Token usage")
    metrics: Metrics = Field(description="Performance metrics")
    metadata: Metadata = Field(description="Request metadata")


class WinnerInfo(BaseModel):
    """Information about the fastest provider in parallel orchestration."""

    provider: str = Field(description="Provider that won")
    model: str = Field(description="Model used")
    latency_ms: float = Field(description="Latency in milliseconds")


class ParallelMetrics(BaseModel):
    """Aggregated metrics for parallel orchestration."""

    total_latency_ms: float = Field(description="Max latency across all providers")
    total_cost_usd: float = Field(description="Sum of all costs")
    num_providers_called: int = Field(description="Number of providers called")
    num_successful: int = Field(description="Number of successful responses")
    num_failed: int = Field(description="Number of failed responses")


class ParallelResponse(BaseModel):
    """
    Response model for parallel orchestration.

    Example:
        {
            "content": "Winner's response",
            "winner": {...},
            "all_responses": [...],
            "metrics": {...}
        }
    """

    content: str = Field(description="Content from fastest response")
    winner: WinnerInfo = Field(description="Information about winner")
    all_responses: list[ChatResponse] = Field(description="All responses")
    errors: list[dict] | None = Field(default=None, description="Errors from failed providers")
    metrics: ParallelMetrics = Field(description="Aggregated metrics")


class FallbackResponse(BaseModel):
    """
    Response model for fallback orchestration.

    Example:
        {
            "content": "Response text",
            "provider_used": "anthropic",
            "primary_success": false,
            "fallback_triggered": true,
            "primary_error": "Timeout"
        }
    """

    content: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    provider: str = Field(description="Provider name")
    provider_used: str = Field(description="Which provider succeeded")
    primary_success: bool = Field(description="Whether primary succeeded")
    fallback_triggered: bool = Field(description="Whether fallback was used")
    primary_error: str | None = Field(
        default=None, description="Error from primary provider if failed"
    )
    usage: TokenUsage = Field(description="Token usage")
    metrics: Metrics = Field(description="Performance metrics")
    metadata: Metadata = Field(description="Request metadata")


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str = Field(description="Model identifier")
    provider: str = Field(description="Provider name")
    max_tokens: int = Field(description="Maximum context length")
    cost_per_1m_prompt_tokens: float = Field(description="Cost per 1M prompt tokens")
    cost_per_1m_completion_tokens: float = Field(description="Cost per 1M completion tokens")
    supports_streaming: bool = Field(description="Whether streaming is supported")


class ModelsResponse(BaseModel):
    """Response model for models list endpoint."""

    models: list[ModelInfo] = Field(description="List of available models")


class ProviderStatus(BaseModel):
    """Status of a provider."""

    status: Literal["connected", "error", "unknown"] = Field(
        description="Provider connection status"
    )
    error: str | None = Field(default=None, description="Error message if status is error")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Example:
        {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:45Z",
            "version": "0.1.0",
            "providers": {...}
        }
    """

    status: Literal["healthy", "unhealthy"] = Field(description="Overall health status")
    timestamp: str = Field(description="ISO timestamp")
    version: str = Field(description="API version")
    providers: dict[str, ProviderStatus] = Field(description="Status of each provider")


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""

    total_requests: int = Field(description="Total API requests")
    requests_by_provider: dict[str, int] = Field(description="Requests broken down by provider")
    average_latency_ms: float = Field(description="Average latency")
    total_cost_usd: float = Field(description="Total cost")
    cache_hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    error_rate: float = Field(description="Error rate (0.0-1.0)")
    uptime_seconds: int = Field(description="Uptime in seconds")


# ============================================================================
# Error Response Models
# ============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Example:
        {
            "error": "validation_error",
            "message": "temperature must be between 0.0 and 2.0",
            "field": "temperature",
            "request_id": "req_abc123",
            "timestamp": "2024-01-15T10:30:45Z"
        }
    """

    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    field: str | None = Field(
        default=None, description="Field that caused error (for validation errors)"
    )
    request_id: str = Field(description="Request ID for debugging")
    timestamp: str = Field(description="ISO timestamp")
