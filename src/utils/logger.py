"""
Structured logging configuration using structlog.

Provides JSON-formatted logs for production with human-readable
output for development. Automatically tracks request IDs, latency,
costs, and other metrics.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("llm_call", model="gpt-4", latency_ms=1234, cost_usd=0.0045)
"""

import logging
import sys
from typing import Any

import structlog

from src.utils.config import is_development, settings


def setup_logging() -> None:
    """
    Configure structlog with appropriate processors for the environment.

    Development: Human-readable console output with colors
    Production: JSON-formatted logs for parsing/aggregation
    """

    # Shared processors for all environments
    shared_processors = [
        structlog.contextvars.merge_contextvars,  # Add context vars
        structlog.stdlib.add_log_level,  # Add log level
        structlog.stdlib.add_logger_name,  # Add logger name
        structlog.processors.TimeStamper(fmt="iso"),  # ISO timestamp
        structlog.processors.StackInfoRenderer(),  # Stack traces
        structlog.processors.format_exc_info,  # Exception formatting
    ]

    # Development: Pretty console output
    if is_development():
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]
    # Production: JSON output
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,  # Format tracebacks as dicts
            structlog.processors.JSONRenderer(),  # JSON output
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to play nice with structlog
    logging.basicConfig(
        format="%(message)s",
        level=logging.getLevelName(settings.log_level),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("user_action", user_id=123, action="login")
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger property to any class.

    Usage:
        class MyService(LoggerMixin):
            def do_something(self):
                self.logger.info("doing_something", param="value")
    """

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


# Convenience functions for common logging patterns


def log_llm_call(
    model: str,
    provider: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    success: bool = True,
    error: str | None = None,
) -> None:
    """
    Log an LLM API call with standardized metrics.

    Args:
        model: Model identifier
        provider: Provider name (openai, anthropic)
        latency_ms: Request latency in milliseconds
        prompt_tokens: Input tokens used
        completion_tokens: Output tokens generated
        cost_usd: Cost in USD
        success: Whether the call succeeded
        error: Error message if failed
    """
    logger = get_logger("llm")

    log_data = {
        "event": "llm_call",
        "model": model,
        "provider": provider,
        "latency_ms": round(latency_ms, 2),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "cost_usd": round(cost_usd, 6),
        "success": success,
    }

    if error:
        log_data["error"] = error
        logger.error(**log_data)
    else:
        logger.info(**log_data)


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    request_id: str,
    client_ip: str | None = None,
) -> None:
    """
    Log an API request with standardized fields.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        latency_ms: Request latency in milliseconds
        request_id: Unique request identifier
        client_ip: Client IP address (optional)
    """
    logger = get_logger("api")

    logger.info(
        "api_request",
        method=method,
        path=path,
        status_code=status_code,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
        client_ip=client_ip,
    )


def log_error(error_type: str, message: str, **extra: Any) -> None:
    """
    Log an error with standardized format.

    Args:
        error_type: Type of error (validation_error, provider_error, etc.)
        message: Error message
        **extra: Additional context fields
    """
    logger = get_logger("error")

    logger.error("error_occurred", error_type=error_type, message=message, **extra)


# Initialize logging when module is imported
setup_logging()
