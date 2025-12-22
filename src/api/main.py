"""
FastAPI application setup with middleware, monitoring, and exception handling.

This module wires together:
- Request/response middleware (request IDs, logging, performance, rate limiting)
- Monitoring endpoints (detailed health, metrics, readiness/liveness)
- API routers for chat/orchestration features
- Structured error handling for consistent responses
"""

import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.health import (
    DetailedHealthStatus,
    MetricsResponse,
    detailed_health_check,
    liveness_check,
    readiness_check,
)
from src.api.health import (
    get_metrics as get_system_metrics,
)
from src.api.middleware import PerformanceMonitoringMiddleware, get_cors_config
from src.api.routes import router as api_router
from src.api.state import stats
from src.utils.config import settings
from src.utils.logger import get_logger, log_api_request

logger = get_logger(__name__)

# In-memory rate limit tracking per client IP
_rate_limit_records: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_limit = max(settings.rate_limit_requests, 100)  # align with prod expectation
_rate_limit_window = max(settings.rate_limit_window, 60)
_slow_request_threshold = 5.0  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hooks for startup/shutdown logging and validation."""
    logger.info(
        "application_starting", environment=settings.environment, version=settings.api_version
    )

    if not settings.openai_api_key:
        logger.warning("openai_api_key_not_set")

    logger.info("application_ready")
    yield
    logger.info("application_shutting_down")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ----------------------------------------------------------------------------
# Middleware
# ----------------------------------------------------------------------------


app.add_middleware(CORSMiddleware, **get_cors_config())
app.add_middleware(PerformanceMonitoringMiddleware)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request ID, enforce rate limits, log, and collect basic stats."""

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Sliding window rate limiting (in-memory). In production, replace with Redis.
    records = _rate_limit_records[client_ip]
    while records and now - records[0] > _rate_limit_window:
        records.popleft()

    if len(records) >= _rate_limit_limit:
        reset_at = int(records[0] + _rate_limit_window)
        logger.warning(
            "rate_limit_exceeded",
            client_ip=client_ip,
            limit=_rate_limit_limit,
            window_seconds=_rate_limit_window,
            request_id=request_id,
        )

        stats["errors"] += 1
        stats["total_requests"] += 1

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "request_id": request_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            headers={
                "X-Request-ID": request_id,
                "X-RateLimit-Limit": str(_rate_limit_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            },
        )

    records.append(now)

    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    remaining = max(_rate_limit_limit - len(records), 0)
    reset_at = int(records[0] + _rate_limit_window) if records else int(now + _rate_limit_window)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-RateLimit-Limit"] = str(_rate_limit_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_at)

    log_api_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
        request_id=request_id,
        client_ip=client_ip,
    )

    stats["total_requests"] += 1
    stats["total_latency_ms"] += latency_ms

    if latency_ms / 1000 > _slow_request_threshold:
        logger.warning(
            "slow_request",
            path=request.url.path,
            method=request.method,
            latency_ms=round(latency_ms, 2),
            threshold_seconds=_slow_request_threshold,
            request_id=request_id,
        )

    return response


# ----------------------------------------------------------------------------
# Exception Handlers
# ----------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")

    first_error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(loc) for loc in first_error.get("loc", []))
    message = first_error.get("msg", "Validation error")

    logger.warning(
        "validation_error",
        request_id=request_id,
        field=field,
        message=message,
        path=request.url.path,
    )

    stats["errors"] += 1

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": message,
            "field": field,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning("value_error", request_id=request_id, error=str(exc), path=request.url.path)

    stats["errors"] += 1

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "invalid_input",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error("timeout_error", request_id=request_id, error=str(exc), path=request.url.path)

    stats["errors"] += 1

    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "timeout_error",
            "message": "Request timed out",
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")

    error_str = str(exc).lower()
    if any(keyword in error_str for keyword in ["api", "provider", "openai"]):
        status_code = status.HTTP_502_BAD_GATEWAY
        error_type = "provider_error"
        message = "Upstream provider error"
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_type = "internal_error"
        message = "Internal server error"

    logger.error(
        "exception",
        request_id=request_id,
        error_type=error_type,
        error=str(exc),
        path=request.url.path,
    )

    stats["errors"] += 1

    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_type,
            "message": message,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------


# Core API routes (chat, fallback, parallel, stats, root)
app.include_router(api_router)

# Monitoring/ops routes (avoid duplicate /health basic route)
app.add_api_route(
    path="/health/detailed",
    endpoint=detailed_health_check,
    methods=["GET"],
    response_model=DetailedHealthStatus,
    tags=["Monitoring"],
    summary="Detailed health check",
    description="Comprehensive health check for dependencies and system resources.",
)
app.add_api_route(
    path="/health/ready",
    endpoint=readiness_check,
    methods=["GET"],
    tags=["Monitoring"],
    summary="Readiness probe",
    description="Indicates whether the service is ready to receive traffic.",
)
app.add_api_route(
    path="/health/live",
    endpoint=liveness_check,
    methods=["GET"],
    tags=["Monitoring"],
    summary="Liveness probe",
    description="Basic liveness check to ensure the service is responsive.",
)
app.add_api_route(
    path="/metrics",
    endpoint=get_system_metrics,
    methods=["GET"],
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="System metrics",
    description="Current CPU, memory, and disk utilization for the host.",
)
