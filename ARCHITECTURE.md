System Architecture

Overview
FastAPI-based LLM orchestration service focused on OpenAI models. Provides single-call chat, parallel fan-out, fallback between model sizes, and streaming responses with structured logging.

Design Principles
- Async-first: asyncio and async HTTP clients for throughput
- Focused scope: OpenAI-only models (4o / 4o-mini / 4-turbo / 3.5)
- Observability: structured logging and lightweight in-memory stats
- Resilience: timeouts, simple fallbacks, default message on total failure
- Testable: dependency injection for clients and orchestrators

System Components
1. API Layer (src/api/)
Responsibility: HTTP interface and request/response handling
- main.py: FastAPI app setup, middleware hooks, exception handlers
- routes.py: Endpoints (/chat, /chat/parallel, /chat/fallback, /chat/stream, /models, /stats, /health)
- models.py: Pydantic schemas for requests and responses
- middleware.py: Request ID, CORS, logging/rate-limit helpers (not all wired by default)
- state.py: In-memory counters for basic metrics

2. LLM Layer (src/llm/)
Responsibility: Provider client and orchestration strategies
- clients.py: OpenAI client wrapper plus factory get_client("openai")
- orchestrator.py: High-level strategies
      - parallel_orchestration(): race multiple OpenAI models and return fastest success
      - fallback_orchestration(): try primary model then progressively smaller models
      - streaming_orchestration(): async token generator for SSE streaming
- models.py: Normalized request/response dataclasses

3. Utilities (src/utils/)
Responsibility: Cross-cutting concerns
- config.py: Pydantic settings (API keys, defaults, timeouts, logging levels)
- logger.py: structlog-based JSON logging with request correlation IDs

Data Flows
Standard Chat
┌──────┐      ┌─────────┐      ┌──────────────┐      ┌──────────┐
│Client│ ───▶ │FastAPI  │ ───▶ │Orchestrator  │ ───▶ │OpenAI    │
└──────┘      └─────────┘      └──────────────┘      └──────────┘
      ▲               │                │                    │
      │               ▼                ▼                    ▼
      │         ┌─────────┐      ┌──────────────┐      ┌──────────┐
      └────────▶│Response │ ◀─── │Aggregate     │ ◀─── │Result    │
                │Model    │      │Metrics/Usage │      │Data      │
                └─────────┘      └──────────────┘      └──────────┘

Parallel Orchestration (OpenAI models)
                    ┌─────────────┐
            ┌──────▶│OpenAI 4o    │─────┐
            │       └─────────────┘     │
┌───────────┴──┐                        ├──▶ asyncio.as_completed()
│Orchestrator  │                        │
└───────────┬──┘                        │
            │       ┌─────────────┐     │
            └──────▶│OpenAI 4o-mini│    │
                    └─────────────┘     │
                                        ▼
                                  ┌──────────┐
                                  │Return    │
                                  │Fastest   │
                                  └──────────┘

Streaming Response
Client ◀─── SSE Stream ◀─── FastAPI ◀─── async for token ◀─── OpenAI

Configuration
Required environment variables (.env)
- OPENAI_API_KEY=sk-...

Optional
- DEFAULT_MODEL=gpt-4o
- DEFAULT_TEMPERATURE=0.7
- DEFAULT_MAX_TOKENS=500
- DEFAULT_TIMEOUT=30
- ENVIRONMENT=development|production|testing
- LOG_LEVEL=INFO
- RATE_LIMIT_REQUESTS=10
- RATE_LIMIT_WINDOW=60

Model Configuration (config.py)
- gpt-4o: max_tokens 128000, streaming supported
- gpt-4o-mini: max_tokens 128000, streaming supported
- gpt-4-turbo: max_tokens 128000, streaming supported
- gpt-3.5-turbo: max_tokens 4096, streaming supported

Caching and consensus orchestration are not implemented; cache flags are present in settings but unused.

Error Handling Strategy
- Validation (400): malformed payloads
- Auth (401): missing or invalid credentials to provider
- Rate limit (429): provider quota exceeded
- Timeout (504): model did not respond in time
- Provider (502): upstream API failure
- Server (500): unexpected exceptions

Retry Logic
- Not enabled; future enhancement to add bounded retries for transient provider errors.

Performance Targets
Metric | Target | Notes
P50 latency | < 2s | time to first token
P99 latency | < 10s | end-to-end response
Throughput | O(100) rps | under parallel fan-out
Error rate | < 1% | excluding timeouts

Future Enhancements
Phase 1 (Current)
- ✅ Basic chat endpoint
- ✅ OpenAI support (4o/4o-mini/4-turbo/3.5)
- ✅ Parallel and fallback orchestration
- ✅ SSE streaming
- ✅ Structured logging and health checks

Phase 2 (Next)
- Streaming polish and backpressure handling
- Parallel tuning and circuit-breaking
- Cost tracking dashboard
- Retry policies with jitter

Phase 3 (Future)
- WebSocket streaming
- Prompt template library
- A/B testing framework
- Additional providers (Cohere, Anthropic, etc.)

Observability
- Metrics: total requests, requests by provider, avg latency, error rate, total cost
- Logging: JSON logs with request_id and timing details

Testing Strategy
- Unit tests: mock OpenAI client, validate orchestration branching and error paths
- Integration tests: exercise FastAPI endpoints with test client
- Coverage: run pixi run test to validate before deploy
