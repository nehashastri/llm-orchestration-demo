"""
Targeted unit tests for error handlers, middleware classes, orchestrator edge cases,
health helpers, and logger utilities to boost coverage.
"""

import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import Response

from src.api import errors, health
from src.api.middleware import (
    PerformanceMonitoringMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from src.llm import orchestrator
from src.utils import logger as logger_utils
from src.utils.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(path: str = "/chat") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [(b"x-request-id", b"req-123")],
        "client": ("1.1.1.1", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    req = Request(scope)
    req.state.request_id = "req-123"
    return req


async def dummy_call_next(_request: Request, status_code: int = 200) -> Response:
    return Response(content=b"ok", status_code=status_code)


# ---------------------------------------------------------------------------
# errors.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_exception_handler_includes_status():
    request = make_request()
    exc = HTTPException(status_code=404, detail="not found")
    resp = await errors.http_exception_handler(request, exc)
    assert resp.status_code == 404
    assert b"http_error" in resp.body


@pytest.mark.asyncio
async def test_validation_exception_handler_lists_fields():
    request = make_request()
    exc = RequestValidationError([{"loc": ("body", "field"), "msg": "bad", "type": "value_error"}])
    resp = await errors.validation_exception_handler(request, exc)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert b"validation_error" in resp.body
    assert b"field" in resp.body


@pytest.mark.asyncio
async def test_llm_api_exception_handler_maps_status():
    request = make_request()
    exc = errors.LLMRateLimitError(message="ratelimited", provider="openai")
    resp = await errors.llm_api_exception_handler(request, exc)
    assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert b"llm_api_error" in resp.body
    assert b"ratelimited" in resp.body


@pytest.mark.asyncio
async def test_generic_exception_handler_sanitizes_message():
    request = make_request()
    exc = RuntimeError("boom")
    resp = await errors.generic_exception_handler(request, exc)
    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert b"internal_error" in resp.body


# ---------------------------------------------------------------------------
# middleware classes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_logging_middleware_happy_path():
    middleware = RequestLoggingMiddleware(app=FastAPI())
    request = make_request("/hello")
    resp = await middleware.dispatch(request, lambda req: dummy_call_next(req, 201))
    assert resp.status_code == 201
    assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_request_logging_middleware_error_path():
    middleware = RequestLoggingMiddleware(app=FastAPI())
    request = make_request("/err")

    async def raise_next(_request: Request):
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError):
        await middleware.dispatch(request, raise_next)


@pytest.mark.asyncio
async def test_performance_middleware_warn_branch(monkeypatch):
    middleware = PerformanceMonitoringMiddleware(app=FastAPI())
    monkeypatch.setattr(PerformanceMonitoringMiddleware, "SLOW_REQUEST_THRESHOLD", 0)
    request = make_request("/slow")
    resp = await middleware.dispatch(request, lambda req: dummy_call_next(req, 200))
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_middleware_blocks_second_request(monkeypatch):
    # Mock the Redis rate limit check to simulate rate limit scenarios
    call_count = [0]  # Use list to allow mutation in nested function

    async def mock_check_rate_limit(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            return (True, 0)  # First request allowed
        else:
            return (False, 0)  # Second request blocked

    monkeypatch.setattr("src.api.middleware.check_rate_limit", mock_check_rate_limit)

    middleware = RateLimitMiddleware(app=None, requests_per_window=1, window_seconds=3600)
    request = make_request("/chat")

    resp1 = await middleware.dispatch(request, lambda req: dummy_call_next(req, 200))
    assert resp1.status_code == 200

    resp2 = await middleware.dispatch(request, lambda req: dummy_call_next(req, 200))
    assert resp2.status_code == 429


# ---------------------------------------------------------------------------
# orchestrator branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_orchestration_all_fail(monkeypatch):
    class FailClient:
        async def generate(self, *_, **__):
            raise RuntimeError("fail")

    monkeypatch.setattr(orchestrator, "get_client", lambda _p: FailClient())

    with pytest.raises(RuntimeError):
        await orchestrator.parallel_orchestration(prompt="hi", models=["a", "b"], version=1)


@pytest.mark.asyncio
async def test_fallback_orchestration_all_fail(monkeypatch):
    class FailClient:
        async def generate(self, *_, **__):
            raise RuntimeError("fail")

    monkeypatch.setattr(orchestrator, "get_client", lambda _p: FailClient())

    result = await orchestrator.fallback_orchestration(prompt="hi", primary_model="m1")
    assert result["is_default_message"] is True
    assert result["fallback_triggered"] is True


@pytest.mark.asyncio
async def test_streaming_orchestration_error_branch(monkeypatch):
    class FailStreamClient:
        async def generate_stream(self, *_, **__):
            if False:  # pragma: no cover - ensures async generator type
                yield "never"
            raise RuntimeError("stream fail")

    monkeypatch.setattr(orchestrator, "get_client", lambda _p: FailStreamClient())

    with pytest.raises(RuntimeError):
        async for _token in orchestrator.streaming_orchestration(prompt="hi", provider="openai"):
            pass


# ---------------------------------------------------------------------------
# health helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_openai_unhealthy_when_missing_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await health._check_openai()
    assert result["status"] == "unhealthy"


def test_check_system_resources_unhealthy(monkeypatch):
    memory = SimpleNamespace(percent=95, used=1024 * 1024 * 1024, total=2 * 1024 * 1024 * 1024)
    disk = SimpleNamespace(percent=95, used=1, total=2)

    monkeypatch.setattr(health.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(health.psutil, "disk_usage", lambda _p="/": disk)

    result = health._check_system_resources()
    assert result["status"] == "unhealthy"


def test_get_system_metrics_fields(monkeypatch):
    memory = SimpleNamespace(percent=10, used=1024 * 1024 * 1024, total=2 * 1024 * 1024 * 1024)
    disk = SimpleNamespace(percent=20, used=1, total=2)

    monkeypatch.setattr(health.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(health.psutil, "disk_usage", lambda _p="/": disk)
    monkeypatch.setattr(health.psutil, "cpu_percent", lambda interval=0.1: 5.0)

    metrics = health._get_system_metrics()
    assert "cpu_percent" in metrics and "memory_used_mb" in metrics


# ---------------------------------------------------------------------------
# logger utilities
# ---------------------------------------------------------------------------


def test_setup_logging_production(monkeypatch):
    original_env = settings.environment
    try:
        monkeypatch.setattr(settings, "environment", "production")
        logger_utils.setup_logging()
    finally:
        monkeypatch.setattr(settings, "environment", original_env)


def test_log_llm_call_error_branch():
    logger_utils.log_llm_call(
        model="m",
        provider="openai",
        latency_ms=1.2,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.1,
        success=False,
        error="boom",
    )


def test_log_error_helper():
    logger_utils.log_error("type", "message", extra="x")


# ---------------------------------------------------------------------------
# additional coverage helpers
# ---------------------------------------------------------------------------


def test_register_error_handlers_sets_handlers():
    app = FastAPI()
    errors.register_error_handlers(app)
    assert HTTPException in app.exception_handlers
    assert RequestValidationError in app.exception_handlers
    assert errors.LLMAPIError in app.exception_handlers
    assert Exception in app.exception_handlers


def test_log_api_request_helper():
    logger_utils.log_api_request("GET", "/path", 200, 12.3, "req-1", client_ip="1.1.1.1")


@pytest.mark.asyncio
async def test_health_checks_when_keys_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    openai_status = await health._check_openai()
    readiness = await health._check_if_ready()

    assert openai_status["status"] == "healthy"
    assert readiness is True


@pytest.mark.asyncio
async def test_health_check_direct_call():
    status_obj = await health.health_check()
    assert status_obj.status == "healthy"


@pytest.mark.asyncio
async def test_parallel_orchestration_invalid_version():
    with pytest.raises(ValueError):
        await orchestrator.parallel_orchestration(prompt="hi", version=99)


@pytest.mark.asyncio
async def test_fallback_orchestration_invalid_provider():
    with pytest.raises(ValueError):
        await orchestrator.fallback_orchestration(prompt="hi", primary_provider="anthropic")


@pytest.mark.asyncio
async def test_streaming_orchestration_success_path(monkeypatch):
    class SuccessStreamClient:
        async def generate_stream(self, *_, **__):
            for token in ["a", "b"]:
                yield token

    monkeypatch.setattr(orchestrator, "get_client", lambda _p: SuccessStreamClient())

    tokens = []
    async for tok in orchestrator.streaming_orchestration(prompt="hi", provider="openai"):
        tokens.append(tok)

    assert tokens == ["a", "b"]


@pytest.mark.asyncio
async def test_parallel_orchestration_success(monkeypatch):
    class SuccessClient:
        async def generate(self, *, model, **_kwargs):
            await asyncio.sleep(0)
            return {
                "content": f"resp-{model}",
                "model": model,
                "provider": "openai",
                "metrics": {"cost_usd": 0.01},
                "usage": {},
            }

    monkeypatch.setattr(orchestrator, "get_client", lambda _p: SuccessClient())

    result = await orchestrator.parallel_orchestration(prompt="hi", version=2, models=["m1", "m2"])
    assert result["metrics"]["num_successful"] == 2
    assert result["winner"]["model"] in {"m1", "m2"}


def test_check_system_resources_healthy(monkeypatch):
    memory = SimpleNamespace(percent=10, used=1, total=2)
    disk = SimpleNamespace(percent=10, used=1, total=2)

    monkeypatch.setattr(health.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(health.psutil, "disk_usage", lambda _p="/": disk)

    result = health._check_system_resources()
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check_ready(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ready-key")
    result = await health.readiness_check()
    assert result["status"] == "ready"


def test_logging_date_namer_formats(tmp_path, monkeypatch):
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)

    logger_utils.setup_logging()

    handler = next(
        h for h in logging.getLogger().handlers if isinstance(h, TimedRotatingFileHandler)
    )

    assert handler.namer is not None
    formatted_with_log = handler.namer(str(tmp_path / "app.log.2025-01-01"))
    formatted_plain = handler.namer(str(tmp_path / "app.2025-01-01"))

    assert formatted_with_log.endswith("app-2025-01-01.log")
    assert formatted_plain.endswith("app-2025-01-01.log")
