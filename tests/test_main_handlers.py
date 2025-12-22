"""
Direct tests for exception handlers and middleware helpers in src.api.main to improve coverage.
"""

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.api import main


def _request(path: str = "/chat") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
    }
    req = Request(scope)
    req.state.request_id = "req-test"
    return req


class TestExceptionHandlers:
    @pytest.mark.asyncio
    async def test_validation_handler_returns_json_response(self):
        exc = RequestValidationError(
            [{"loc": ("body", "prompt"), "msg": "bad prompt", "type": "value_error"}]
        )
        response = await main.validation_exception_handler(_request(), exc)
        assert response.status_code == 422
        body = bytes(response.body).decode()
        assert "validation_error" in body
        assert "prompt" in body

    @pytest.mark.asyncio
    async def test_value_error_handler(self):
        exc = ValueError("invalid model")
        response = await main.value_error_handler(_request(), exc)
        assert response.status_code == 400
        assert b"invalid_input" in response.body

    @pytest.mark.asyncio
    async def test_timeout_handler(self):
        exc = TimeoutError("Request timed out")
        response = await main.timeout_error_handler(_request(), exc)
        assert response.status_code == 504
        assert b"timeout_error" in response.body

    @pytest.mark.asyncio
    async def test_general_handler_provider_branch(self):
        exc = Exception("OpenAI API error")
        response = await main.general_exception_handler(_request(), exc)
        assert response.status_code == 502
        assert b"provider_error" in response.body

    @pytest.mark.asyncio
    async def test_general_handler_internal_branch(self):
        exc = Exception("unexpected boom")
        response = await main.general_exception_handler(_request(), exc)
        assert response.status_code == 500
        assert b"internal_error" in response.body


class TestLifespan:
    def test_lifespan_start_and_shutdown_execute(self):
        """Using TestClient as context runs lifespan hooks (startup/shutdown)."""
        with TestClient(main.app) as client:
            res = client.get("/health")
            assert res.status_code == 200
