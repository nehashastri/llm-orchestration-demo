# src/api/errors.py
"""
Production-grade error handling for FastAPI.
This module defines custom exceptions and error handlers.
"""

import traceback
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ExceptionHandler

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================
# CUSTOM EXCEPTIONS
# ============================================


class LLMAPIError(Exception):
    """Base exception for LLM API errors"""

    def __init__(
        self, message: str, provider: str, original_error: Exception | None = None
    ) -> None:
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(self.message)


class LLMTimeoutError(LLMAPIError):
    """Raised when LLM API call times out"""

    pass


class LLMRateLimitError(LLMAPIError):
    """Raised when LLM API rate limit is exceeded"""

    pass


class LLMAuthenticationError(LLMAPIError):
    """Raised when LLM API authentication fails"""

    pass


class LLMInvalidRequestError(LLMAPIError):
    """Raised when LLM API request is invalid"""

    pass


# ============================================
# ERROR RESPONSE MODELS
# ============================================


class ErrorResponse:
    """Standard error response format"""

    @staticmethod
    def create(
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create standardized error response.

        Example response:
        {
            "error": {
                "type": "validation_error",
                "message": "Invalid temperature value",
                "details": {"field": "temperature", "value": 5.0},
                "request_id": "abc123"
            }
        }
        """
        error_obj: dict[str, Any] = {
            "type": error_type,
            "message": message,
        }

        if details is not None:
            error_obj["details"] = details

        if request_id is not None:
            error_obj["request_id"] = request_id

        response: dict[str, Any] = {"error": error_obj}
        return response


# ============================================
# EXCEPTION HANDLERS
# ============================================


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTPExceptions.

    These are raised by FastAPI for things like:
    - 404 Not Found
    - 403 Forbidden
    - 401 Unauthorized
    """
    logger.warning(
        "HTTP exception", status_code=exc.status_code, detail=exc.detail, path=request.url.path
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.create(
            error_type="http_error",
            message=exc.detail,
            details={"status_code": exc.status_code},
            request_id=request.headers.get("X-Request-ID"),
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    These are raised when request data doesn't match your Pydantic models.
    Example: temperature = 5.0 (should be 0.0-2.0)
    """
    # Extract validation errors
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning("Validation error", path=request.url.path, errors=errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse.create(
            error_type="validation_error",
            message="Request validation failed",
            details={"errors": errors},
            request_id=request.headers.get("X-Request-ID"),
        ),
    )


async def llm_api_exception_handler(request: Request, exc: LLMAPIError) -> JSONResponse:
    """
    Handle LLM API specific errors.

    These are custom exceptions we defined above for:
    - Timeouts
    - Rate limits
    - Authentication failures
    """
    # Map exception types to HTTP status codes
    status_code_map = {
        LLMTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
        LLMRateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
        LLMAuthenticationError: status.HTTP_401_UNAUTHORIZED,
        LLMInvalidRequestError: status.HTTP_400_BAD_REQUEST,
    }

    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.error(
        "LLM API error",
        error_type=type(exc).__name__,
        message=exc.message,
        provider=exc.provider,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse.create(
            error_type="llm_api_error",
            message=exc.message,
            details={"provider": exc.provider, "error_class": type(exc).__name__},
            request_id=request.headers.get("X-Request-ID"),
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unexpected exceptions.

    This is the "catch-all" handler for bugs and unexpected errors.
    We log the full traceback for debugging but return a safe error message.
    """
    # Log full traceback for debugging
    logger.error(
        "Unexpected error",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        traceback=traceback.format_exc(),
    )

    # In production, don't expose internal error details to users
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse.create(
            error_type="internal_error",
            message="An unexpected error occurred. Please try again later.",
            details={"error_id": request.headers.get("X-Request-ID")},
            request_id=request.headers.get("X-Request-ID"),
        ),
    )


# ============================================
# HELPER FUNCTIONS
# ============================================


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers with the FastAPI app.

    Call this in main.py:
        from src.api.errors import register_error_handlers
        register_error_handlers(app)
    """
    app.add_exception_handler(HTTPException, cast(ExceptionHandler, http_exception_handler))
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_exception_handler)
    )
    app.add_exception_handler(LLMAPIError, cast(ExceptionHandler, llm_api_exception_handler))
    app.add_exception_handler(Exception, cast(ExceptionHandler, generic_exception_handler))

    logger.info("Error handlers registered")


# ============================================
# USAGE EXAMPLES
# ============================================

"""
Example 1: Raise custom exception in your code:

# In src/llm/clients.py:
async def call_openai(prompt: str):
    try:
        response = await openai_client.chat.completions.create(...)
        return response
    except openai.Timeout:
        raise LLMTimeoutError(
            message="OpenAI API timeout after 30 seconds",
            provider="openai",
            original_error=e
        )
    except openai.RateLimitError:
        raise LLMRateLimitError(
            message="OpenAI rate limit exceeded",
            provider="openai",
            original_error=e
        )

Example 2: The error handler automatically converts it to JSON:

Request: POST /chat with temperature=5.0
Response: 422 Unprocessable Entity
{
    "error": {
        "type": "validation_error",
        "message": "Request validation failed",
        "details": {
            "errors": [
                {
                    "field": "temperature",
                    "message": "ensure this value is less than or equal to 2.0",
                    "type": "value_error.number.not_le"
                }
            ]
        },
        "request_id": "abc123"
    }
}

Example 3: Generic error handling:

If any unexpected error occurs (like a bug in your code),
the generic_exception_handler catches it:

Response: 500 Internal Server Error
{
    "error": {
        "type": "internal_error",
        "message": "An unexpected error occurred. Please try again later.",
        "details": {
            "error_id": "abc123"
        }
    }
}

Meanwhile, the full traceback is logged for debugging.
"""
