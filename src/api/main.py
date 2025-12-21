"""
FastAPI application factory and configuration.

Creates and configures the FastAPI application with middleware,
exception handlers, and startup/shutdown events.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.config import settings
from src.utils.logger import get_logger, log_api_request

logger = get_logger(__name__)

# Track app start time for uptime calculation
app_start_time = time.time()

# Statistics tracking (in production, use Redis or database)
stats = {
    "total_requests": 0,
    "requests_by_provider": {},
    "total_cost_usd": 0.0,
    "total_latency_ms": 0.0,
    "errors": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for startup and shutdown events.

    Runs code before the application starts accepting requests
    and after it shuts down.
    """
    # Startup
    logger.info(
        "application_starting", environment=settings.environment, version=settings.api_version
    )

    # Validate API keys are set
    if not settings.openai_api_key:
        logger.warning("openai_api_key_not_set")

    logger.info("application_ready")

    yield  # Application runs here

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# Middleware
# ============================================================================


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Add unique request ID to all requests for tracing.

    Sets X-Request-ID header in both request context and response.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Process request
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    # Add headers
    response.headers["X-Request-ID"] = request_id

    # Log request
    log_api_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
        request_id=request_id,
        client_ip=request.client.host if request.client else None,
    )

    # Update stats
    stats["total_requests"] += 1
    stats["total_latency_ms"] += latency_ms

    return response


@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """
    Add rate limit headers to responses.

    In production, implement actual rate limiting with Redis.
    """
    response = await call_next(request)

    # Mock rate limit headers (implement real rate limiting in production)
    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
    response.headers["X-RateLimit-Remaining"] = "8"  # Mock value
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

    return response


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with detailed field information.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Extract first error for clarity
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
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
    """Handle ValueError exceptions (e.g., invalid model)."""
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
    """Handle timeout errors."""
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
    """Handle all other exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Check if it's a provider error
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


# ============================================================================
# Utility Functions
# ============================================================================


def get_uptime_seconds() -> int:
    """Get application uptime in seconds."""
    return int(time.time() - app_start_time)


def update_provider_stats(provider: str, cost: float):
    """Update statistics for a provider."""
    if provider not in stats["requests_by_provider"]:
        stats["requests_by_provider"][provider] = 0

    stats["requests_by_provider"][provider] += 1
    stats["total_cost_usd"] += cost


# Import routes (must be after app creation)
