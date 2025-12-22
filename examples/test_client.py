"""
Comprehensive client to exercise the hosted LLM orchestration server.

Covers endpoints:
- GET /health
- GET /
- POST /chat
- POST /chat/parallel
- POST /chat/fallback
- POST /chat/stream (SSE)
- GET /models
- GET /stats

Usage:
  python examples/test_client.py --base-url http://127.0.0.1:8000 --all
  python examples/test_client.py --base-url https://your-hosted-api.example --chat

Environment variables:
- HOST_URL: overrides --base-url
- API_KEY: optional; sent as X-API-Key if provided

Notes:
- Streaming is Server-Sent Events (SSE) over POST; this client parses SSE lines (`data:`) until [DONE].
- Some success paths depend on provider API keys on the server side. When unavailable, server may return provider/internal errors.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

# Optional convenience: try requests if available for non-streaming endpoints
try:  # noqa: SIM105
    import requests  # type: ignore
except Exception:  # noqa: BLE001
    requests = None  # type: ignore


class SimpleHTTPClient:
    """
    Minimal HTTP client using standard library for portability.

    - Uses requests if available for non-streaming convenience.
    - Falls back to http.client for both standard and streaming requests.
    """

    def __init__(self, base_url: str, timeout: int = 30, api_key: str | None = None) -> None:
        self.base_url = os.environ.get("HOST_URL", base_url).rstrip("/")
        self.parsed = urlparse(self.base_url)
        self.timeout = timeout
        self.default_headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if api_key:
            self.default_headers["X-API-Key"] = api_key

    # ---------------------------------------------
    # Non-streaming GET/POST
    # ---------------------------------------------
    def get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        if requests is not None:
            resp = requests.get(
                self.base_url + path, headers=self.default_headers, timeout=self.timeout
            )  # type: ignore
            return resp.status_code, dict(resp.headers), resp.content
        return self._raw_request("GET", path, None, {**self.default_headers})

    def post_json(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        headers = {**self.default_headers, "Content-Type": "application/json"}
        body = json.dumps(payload).encode("utf-8")
        if requests is not None:
            resp = requests.post(
                self.base_url + path, headers=headers, data=body, timeout=self.timeout
            )  # type: ignore
            return resp.status_code, dict(resp.headers), resp.content
        return self._raw_request("POST", path, body, headers)

    # ---------------------------------------------
    # Streaming SSE over POST
    # ---------------------------------------------
    def post_stream(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, str], Generator[str, None, None]]:
        headers = {
            **self.default_headers,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        body = json.dumps(payload).encode("utf-8")
        status, resp_headers, resp_obj = self._raw_request_stream("POST", path, body, headers)
        return status, resp_headers, self._sse_line_generator(resp_obj)

    # ---------------------------------------------
    # Low-level helpers
    # ---------------------------------------------
    def _connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        host = self.parsed.hostname or "127.0.0.1"
        port = self.parsed.port
        if self.parsed.scheme == "https":
            return http.client.HTTPSConnection(host, port=port, timeout=self.timeout)
        return http.client.HTTPConnection(host, port=port, timeout=self.timeout)

    def _raw_request(
        self, method: str, path: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        conn = self._connection()
        full_path = (self.parsed.path.rstrip("/") + path) if self.parsed.path else path
        conn.request(method, full_path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()  # consume body
        conn.close()
        return resp.status, {k: v for k, v in resp.getheaders()}, data

    def _raw_request_stream(
        self, method: str, path: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], http.client.HTTPResponse]:
        conn = self._connection()
        full_path = (self.parsed.path.rstrip("/") + path) if self.parsed.path else path
        conn.request(method, full_path, body=body, headers=headers)
        resp = conn.getresponse()
        # Do not close connection; caller consumes stream
        return resp.status, {k: v for k, v in resp.getheaders()}, resp

    def _sse_line_generator(self, resp: http.client.HTTPResponse) -> Generator[str, None, None]:
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                yield line.decode("utf-8", errors="replace")
        finally:
            try:
                resp.close()
            except Exception:
                pass


# ============================================================
# Endpoint Test Functions
# ============================================================


def check_headers(name: str, headers: dict[str, str]) -> None:
    rid = headers.get("X-Request-ID")
    limit = headers.get("X-RateLimit-Limit")
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    print(
        f"[{name}] Headers: X-Request-ID={rid} RateLimit(limit={limit}, remaining={remaining}, reset={reset})"
    )


def pretty_json(data: bytes) -> str:
    try:
        return json.dumps(json.loads(data.decode("utf-8")), indent=2)
    except Exception:
        return data.decode("utf-8", errors="replace")


def test_health(client: SimpleHTTPClient) -> bool:
    status, headers, body = client.get("/health")
    ok = status == 200
    print(f"GET /health -> {status}\n{pretty_json(body)}")
    check_headers("health", headers)
    return ok


def test_root(client: SimpleHTTPClient) -> bool:
    status, headers, body = client.get("/")
    ok = status == 200
    print(f"GET / -> {status}\n{pretty_json(body)}")
    check_headers("root", headers)
    return ok


def test_models(client: SimpleHTTPClient) -> bool:
    status, headers, body = client.get("/models")
    ok = status == 200
    print(f"GET /models -> {status}\n{pretty_json(body)}")
    check_headers("models", headers)
    return ok


def test_stats(client: SimpleHTTPClient) -> bool:
    status, headers, body = client.get("/stats")
    ok = status == 200
    print(f"GET /stats -> {status}\n{pretty_json(body)}")
    check_headers("stats", headers)
    return ok


def test_chat_success(client: SimpleHTTPClient, prompt: str = "Write a short haiku.") -> bool:
    payload = {
        "prompt": prompt,
        "model": "gpt-4-turbo",  # default per API schema
        "temperature": 0.7,
        "max_tokens": 64,
        "system_prompt": "You are a helpful assistant",
    }
    status, headers, body = client.post_json("/chat", payload)
    print(f"POST /chat -> {status}\n{pretty_json(body)}")
    check_headers("chat", headers)
    return status == 200


def test_chat_invalid_model(client: SimpleHTTPClient) -> bool:
    payload = {
        "prompt": "Say hello",
        "model": "non-existent-model-xyz",
    }
    status, headers, body = client.post_json("/chat", payload)
    print(f"POST /chat (invalid model) -> {status}\n{pretty_json(body)}")
    check_headers("chat_invalid", headers)
    return status == 400


def test_parallel_v1(client: SimpleHTTPClient, prompt: str = "Brief greeting") -> bool:
    payload = {
        "prompt": prompt,
        "version": 1,
        "temperature": 0.7,
        "max_tokens": 64,
        "system_prompt": "Keep it concise",
    }
    status, headers, body = client.post_json("/chat/parallel", payload)
    print(f"POST /chat/parallel (v1) -> {status}\n{pretty_json(body)}")
    check_headers("parallel_v1", headers)
    return status == 200


def test_parallel_v2(client: SimpleHTTPClient, prompt: str = "Brief greeting") -> bool:
    payload = {
        "prompt": prompt,
        "version": 2,
        "temperature": 0.7,
        "max_tokens": 64,
        "system_prompt": "Keep it concise",
    }
    status, headers, body = client.post_json("/chat/parallel", payload)
    print(f"POST /chat/parallel (v2) -> {status}\n{pretty_json(body)}")
    check_headers("parallel_v2", headers)
    return status == 200


def test_fallback(client: SimpleHTTPClient, prompt: str = "Define AI in one sentence.") -> bool:
    payload = {
        "prompt": prompt,
        "primary_provider": "openai",
        "primary_model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 64,
        "timeout": 30,
        "system_prompt": "You are precise",
    }
    status, headers, body = client.post_json("/chat/fallback", payload)
    print(f"POST /chat/fallback -> {status}\n{pretty_json(body)}")
    check_headers("fallback", headers)
    return status == 200


def test_stream(client: SimpleHTTPClient, prompt: str = "Stream a short sentence.") -> bool:
    payload = {
        "prompt": prompt,
        "provider": "openai",
        "model": None,
        "temperature": 0.7,
        "max_tokens": 32,
        "system_prompt": "Be brief",
    }
    status, headers, stream = client.post_stream("/chat/stream", payload)
    print(f"POST /chat/stream -> {status}")
    # Content-Type should be text/event-stream
    ctype = headers.get("Content-Type")
    print(f"[stream] Headers: Content-Type={ctype} X-Request-ID={headers.get('X-Request-ID')}")
    received_done = False
    token_count = 0
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                received_done = True
                break
            try:
                obj = json.loads(data_str)
                if "token" in obj:
                    token_count += 1
                    # Print token index occasionally
                    if token_count <= 5:
                        print(f"  token[{obj.get('index')}]={obj.get('token')}")
            except Exception:
                # Raw line
                print(f"  raw: {data_str}")
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    except Exception as e:  # noqa: BLE001
        print(f"Streaming error: {e}")
    ok = status == 200 and (ctype or "").startswith("text/event-stream") and received_done
    print(f"Stream tokens received: {token_count}, done: {received_done}")
    return ok


# ============================================================
# CLI
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Client tests for LLM orchestration server")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HOST_URL", "http://127.0.0.1:8000"),
        help="Server base URL",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument(
        "--api-key", default=os.environ.get("API_KEY"), help="Optional X-API-Key header"
    )

    # What to run
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--root", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args(argv)

    client = SimpleHTTPClient(base_url=args.base_url, timeout=args.timeout, api_key=args.api_key)

    targets = []
    if args.all:
        targets = [
            ("health", test_health),
            ("root", test_root),
            ("models", test_models),
            ("chat", test_chat_success),
            ("chat_invalid", test_chat_invalid_model),
            ("parallel_v1", test_parallel_v1),
            ("parallel_v2", test_parallel_v2),
            ("fallback", test_fallback),
            ("stream", test_stream),
            ("stats", test_stats),
        ]
    else:
        if args.health:
            targets.append(("health", test_health))
        if args.root:
            targets.append(("root", test_root))
        if args.models:
            targets.append(("models", test_models))
        if args.chat:
            targets.append(("chat", test_chat_success))
            targets.append(("chat_invalid", test_chat_invalid_model))
        if args.parallel:
            targets.append(("parallel_v1", test_parallel_v1))
            targets.append(("parallel_v2", test_parallel_v2))
        if args.fallback:
            targets.append(("fallback", test_fallback))
        if args.stream:
            targets.append(("stream", test_stream))
        if args.stats:
            targets.append(("stats", test_stats))

    if not targets:
        print("No targets selected. Use --all or a specific flag.")
        return 2

    print(f"Base URL: {client.base_url}")

    failures = []
    start = time.time()
    for name, fn in targets:
        print(f"""
=======================================
Running: {name}
=======================================
""")
        ok = False
        try:
            ok = fn(client)
        except Exception as e:  # noqa: BLE001
            print(f"{name} raised exception: {e}")
        if not ok:
            failures.append(name)

    elapsed = time.time() - start
    print("""
=======================================
Summary
=======================================
""")
    if failures:
        print(f"Failed: {', '.join(failures)}")
    else:
        print("All selected checks passed or responded successfully.")
    print(f"Elapsed: {elapsed:.2f}s")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
