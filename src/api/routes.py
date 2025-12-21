"""
API route definitions for LLM orchestration endpoints.

Implements all endpoints defined in docs/api_specs.md.
"""

import time

from fastapi import Request, status
from fastapi.responses import StreamingResponse

from src.api.main import app, get_uptime_seconds, stats, update_provider_stats
from src.api.models import (
    ChatRequest,
    ChatResponse,
    FallbackRequest,
    FallbackResponse,
    HealthResponse,
    Metadata,
    Metrics,
    ModelInfo,
    ModelsResponse,
    ParallelRequest,
    ParallelResponse,
    StatsResponse,
    StreamRequest,
    TokenUsage,
)
from src.llm.clients import get_client
from src.llm.orchestrator import (
    fallback_orchestration,
    streaming_orchestration,
)
from src.utils.config import get_provider_for_model, list_models, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Health Check
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Check if the service is running and providers are accessible",
)
async def health_check():
    """
    Health check endpoint.

    Returns service status and provider connectivity.
    """
    providers = {}
    overall_healthy = True

    # Check OpenAI
    try:
        _openai_client = get_client("openai")  # type: ignore
        # Could add actual API ping here
        providers["openai"] = {"status": "connected"}
    except Exception as e:
        providers["openai"] = {"status": "error", "error": str(e)}
        overall_healthy = False

    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version=settings.api_version,
        providers=providers,
    )


# ============================================================================
# Chat Completion
# ============================================================================


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Generate chat completion",
    description="Generate a chat completion using a single LLM provider",
)
async def chat_completion(request_body: ChatRequest, request: Request):
    """
    Generate a chat completion.

    Calls a single LLM provider and returns the response with
    usage metrics, latency, and cost information.
    """
    logger.info(
        "chat_request",
        model=request_body.model,
        temperature=request_body.temperature,
        request_id=request.state.request_id,
    )

    # Get provider from model
    provider = get_provider_for_model(request_body.model)
    client = get_client(provider)

    # Generate completion
    result = await client.generate(
        prompt=request_body.prompt,
        model=request_body.model,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
        system_prompt=request_body.system_prompt,
    )

    # Update stats
    update_provider_stats(provider, result["metrics"]["cost_usd"])

    # Build response
    response = ChatResponse(
        content=result["content"],
        model=result["model"],
        provider=result["provider"],
        usage=TokenUsage(**result["usage"]),
        metrics=Metrics(**result["metrics"]),
        metadata=Metadata(
            request_id=request.state.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )

    return response


# ============================================================================
# Parallel Orchestration - DISABLED (OpenAI-only mode)
# ============================================================================


@app.post(
    "/chat/parallel",
    response_model=ParallelResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    tags=["Chat"],
    summary="Parallel orchestration (disabled)",
    description="Not available in OpenAI-only mode",
)
async def parallel_chat(request_body: ParallelRequest, request: Request):
    """
    Parallel orchestration is disabled in OpenAI-only mode.

    Use /chat/fallback endpoint for reliability instead.
    """
    from fastapi import HTTPException

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Parallel orchestration is not available in OpenAI-only mode. Use /chat/fallback instead.",
    )
        all_responses=all_responses,
        errors=result.get("errors"),
        metrics=result["metrics"],
    )


# ============================================================================
# Fallback Orchestration
# ============================================================================


@app.post(
    "/chat/fallback",
    response_model=FallbackResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Fallback orchestration",
    description="Try primary provider, fallback to others on failure",
)
async def fallback_chat(request_body: FallbackRequest, request: Request):
    """
    Execute fallback orchestration.

    Tries the primary provider first, automatically falling back
    to secondary providers if the primary fails or times out.
    """
    logger.info(
        "fallback_request",
        primary=request_body.primary_provider,
        fallbacks=request_body.fallback_providers,
        request_id=request.state.request_id,
    )

    # Execute fallback orchestration
    result = await fallback_orchestration(
        prompt=request_body.prompt,
        primary_provider=request_body.primary_provider,
        fallback_providers=request_body.fallback_providers,
        primary_model=request_body.primary_model,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
        system_prompt=request_body.system_prompt,
        timeout=request_body.timeout,
    )

    # Update stats
    update_provider_stats(result["provider_used"], result["metrics"]["cost_usd"])

    # Build response
    response = FallbackResponse(
        content=result["content"],
        model=result["model"],
        provider=result["provider"],
        provider_used=result["provider_used"],
        primary_success=result["primary_success"],
        fallback_triggered=result["fallback_triggered"],
        primary_error=result.get("primary_error"),
        usage=TokenUsage(**result["usage"]),
        metrics=Metrics(**result["metrics"]),
        metadata=Metadata(
            request_id=request.state.request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )

    return response


# ============================================================================
# Streaming
# ============================================================================


@app.post(
    "/chat/stream",
    tags=["Chat"],
    summary="Streaming chat completion",
    description="Stream chat completion token-by-token using Server-Sent Events",
)
async def stream_chat(request_body: StreamRequest, request: Request):
    """
    Generate streaming chat completion.

    Returns a Server-Sent Events (SSE) stream with tokens
    generated in real-time.
    """
    logger.info(
        "stream_request", provider=request_body.provider, request_id=request.state.request_id
    )

    async def event_generator():
        """
        Generate Server-Sent Events for streaming.

        Yields:
            SSE-formatted strings with tokens
        """
        try:
            token_index = 0
            async for token in streaming_orchestration(
                prompt=request_body.prompt,
                provider=request_body.provider,
                model=request_body.model,
                temperature=request_body.temperature,
                max_tokens=request_body.max_tokens,
                system_prompt=request_body.system_prompt,
            ):
                # Format as Server-Sent Event
                yield f'data: {{"token": "{token}", "index": {token_index}}}\n\n'
                token_index += 1

            # Send completion signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request.state.request_id,
        },
    )


# ============================================================================
# Models
# ============================================================================


@app.get(
    "/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="List available models",
    description="Get list of all available LLM models with their configurations",
)
async def get_models():
    """
    List all available models.

    Returns model metadata including pricing, capabilities,
    and provider information.
    """
    models_data = list_models()
    models = [ModelInfo(**model) for model in models_data]
    return ModelsResponse(models=models)


# ============================================================================
# Statistics
# ============================================================================


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Monitoring"],
    summary="Get API statistics",
    description="Get usage statistics and metrics",
)
async def get_stats():
    """
    Get API usage statistics.

    Returns aggregated metrics about requests, costs, latency,
    and error rates.
    """
    avg_latency = (
        stats["total_latency_ms"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0
    )

    error_rate = stats["errors"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0

    return StatsResponse(
        total_requests=stats["total_requests"],
        requests_by_provider=stats["requests_by_provider"],
        average_latency_ms=round(avg_latency, 2),
        total_cost_usd=round(stats["total_cost_usd"], 6),
        cache_hit_rate=0.0,  # Implement caching in Phase 6
        error_rate=round(error_rate, 3),
        uptime_seconds=get_uptime_seconds(),
    )


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="API information",
    description="Get basic API information and links to documentation",
)
async def root():
    """
    Root endpoint with API information.

    Provides links to documentation and basic service info.
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "models": "/models",
    }
