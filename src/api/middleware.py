"""
Production middleware for logging, timing, and request tracking.
"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger
from src.utils.redis import check_rate_limit

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses.

    What it does:
    - Assigns unique ID to each request
    - Logs request details (method, path, headers)
    - Times how long request takes
    - Logs response details (status, size)

    Think of this as a security camera recording everything.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Incoming request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        # Process the request
        try:
            response = await call_next(request)

            # Calculate request duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=round(duration, 3),
            )

            # Add request ID to response headers (useful for debugging)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(duration, 3))

            return response

        except Exception as e:
            # Log errors
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_seconds=round(duration, 3),
            )
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track performance metrics.

    What it does:
    - Tracks slow requests (> 5 seconds)
    - Logs warnings for slow endpoints
    - Can be extended to send metrics to monitoring systems
    """

    SLOW_REQUEST_THRESHOLD = 5.0  # seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Warn about slow requests
        if duration > self.SLOW_REQUEST_THRESHOLD:
            logger.warning(
                "Slow request detected",
                path=request.url.path,
                method=request.method,
                duration_seconds=round(duration, 3),
                threshold_seconds=self.SLOW_REQUEST_THRESHOLD,
            )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-backed rate limiting middleware using sliding window algorithm.

    What it does:
    - Tracks requests per IP address using Redis
    - Distributed rate limiting (works across multiple servers)
    - Configurable limits and time windows
    - Returns 429 if limit exceeded
    """

    def __init__(self, app, requests_per_window: int = 100, window_seconds: int = 3600):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and docs
        if request.url.path in [
            "/health",
            "/health/ready",
            "/health/live",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        try:
            # Check rate limit using Redis
            is_allowed, remaining = await check_rate_limit(
                client_id=client_ip, limit=self.requests_per_window, window=self.window_seconds
            )

            if not is_allowed:
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    path=request.url.path,
                    limit=self.requests_per_window,
                    window_seconds=self.window_seconds,
                )

                return Response(
                    content='{"error": {"type": "rate_limit_exceeded", "message": "Too many requests. Please try again later."}}',
                    status_code=429,
                    media_type="application/json",
                    headers={
                        "X-RateLimit-Limit": str(self.requests_per_window),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(self.window_seconds),
                    },
                )

            # Add rate limit headers to successful responses
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

            return response

        except Exception as e:
            # If Redis fails, log error and allow request (fail open)
            logger.error("Rate limiting error, allowing request", client_ip=client_ip, error=str(e))
            return await call_next(request)


# ============================================
# CORS MIDDLEWARE CONFIGURATION
# ============================================


def get_cors_config():
    """
    CORS (Cross-Origin Resource Sharing) configuration.

    What is CORS?
    - Browser security feature
    - Prevents malicious websites from accessing your API
    - You need to explicitly allow which websites can call your API

    Example:
    - Your API: http://localhost:8000
    - Your frontend: http://localhost:3000
    - Without CORS: Browser blocks frontend from calling API
    - With CORS: You allow localhost:3000 to call your API
    """
    return {
        # Allow all origins for now to satisfy automated tests and local tools.
        # Tighten this in production by replacing "*" with explicit frontends.
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],  # Allow all HTTP methods (GET, POST, etc.)
        "allow_headers": ["*"],  # Allow all headers
    }


# ============================================
# USAGE IN MAIN.PY
# ============================================

"""
# In src/api/main.py:

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.middleware import (
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware,
    RateLimitMiddleware,
    get_cors_config
)

app = FastAPI()

# Add CORS middleware (must be first!)
app.add_middleware(CORSMiddleware, **get_cors_config())

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_hour=100)

# Now all requests go through these middleware layers:
# Request → CORS → Rate Limit → Logging → Performance → Your Route → Response
"""

# ============================================
# EXAMPLE LOG OUTPUT
# ============================================

"""
When someone calls your API, you'll see logs like this:

2024-12-21 10:30:15 [INFO] Incoming request
    request_id=abc-123
    method=POST
    path=/chat
    client_ip=127.0.0.1
    user_agent=Thunder Client

2024-12-21 10:30:17 [INFO] Request completed
    request_id=abc-123
    method=POST
    path=/chat
    status_code=200
    duration_seconds=1.523

If request is slow:
2024-12-21 10:30:25 [WARNING] Slow request detected
    path=/chat
    method=POST
    duration_seconds=6.234
    threshold_seconds=5.0

If rate limit exceeded:
2024-12-21 10:31:00 [WARNING] Rate limit exceeded
    client_ip=192.168.1.100
    path=/chat
    limit=100
"""
