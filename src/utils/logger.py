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
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

from src.utils.config import is_development, settings


def setup_logging() -> None:
    """
    Configure structlog with appropriate processors for the environment.

    Development: Human-readable console output with colors
    Production: JSON-formatted logs for parsing/aggregation
    """

    # Ensure log directory exists
    log_dir = Path(getattr(settings, "log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    # Renderer selection
    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_development()
        else structlog.processors.JSONRenderer()
    )

    # Shared processors (pre-chain for stdlib logging)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Handlers: console + daily rotating file
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        utc=False,
    )
    # Rotate at midnight and name rotated files with the date
    file_handler.suffix = "%Y-%m-%d"

    def _date_namer(name: str) -> str:
        """Rename rotated files to include date before the extension.

        TimedRotatingFileHandler produces names like:
            <log_dir>/app.log.YYYY-MM-DD
        We convert them to:
            <log_dir>/app-YYYY-MM-DD.log
        """
        p = Path(name)
        fname = p.name
        if ".log." in fname:
            base, date = fname.split(".log.", 1)
            return str(p.parent / f"{base}-{date}.log")
        # Fallback: best-effort transformation
        parts = fname.split(".")
        base = parts[0] if parts else "app"
        date = parts[-1] if parts else "unknown"
        return str(p.parent / f"{base}-{date}.log")

    file_handler.namer = _date_namer

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.getLevelName(settings.log_level),
        handlers=[console_handler, file_handler],
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
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
        provider: Provider name (openai)
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
