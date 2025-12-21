System Architecture

Overview
FastAPI-based LLM orchestration service that provides intelligent routing, parallel execution, and fallback strategies across multiple LLM providers (OpenAI, Anthropic). Built for low-latency, cost-efficient AI interactions with production-grade monitoring.
Design Principles

Async-First: All I/O operations use asyncio for maximum throughput
Provider-Agnostic: Unified interface across OpenAI, Anthropic, and future providers
Observable: Structured logging for latency, cost, and error tracking
Resilient: Timeouts, retries, and graceful degradation
Testable: Dependency injection and mocked external calls

System Components
1. API Layer (src/api/)
Responsibility: HTTP interface and request/response handling

main.py: FastAPI application factory, middleware registration, exception handlers
routes.py: Endpoint definitions (/chat, /stream, /health)
models.py: Pydantic schemas for request validation and response serialization
middleware.py: Logging, rate limiting, CORS, request ID injection

Key Features:

OpenAPI/Swagger auto-generated docs at /docs
Request validation with Pydantic (type safety + auto-docs)
Structured error responses with proper HTTP status codes

2. LLM Layer (src/llm/)
Responsibility: Provider-specific API clients and response normalization

clients.py:

OpenAIClient: Wraps OpenAI SDK, handles chat completions and streaming
AnthropicClient: Wraps Anthropic SDK, normalizes message format
BaseLLMClient: Abstract interface for future providers


orchestrator.py: High-level orchestration strategies

parallel_call(): Executes multiple providers simultaneously, returns fastest
fallback_call(): Primary → Secondary → Tertiary provider chain
streaming_call(): Token-by-token streaming via Server-Sent Events
consensus_call(): Polls multiple models, returns majority vote


models.py: Provider-agnostic data models

LLMRequest: Normalized request format
LLMResponse: Normalized response with metadata (latency, tokens, cost)



3. Utilities Layer (src/utils/)
Responsibility: Cross-cutting concerns

config.py: Environment-based settings using pydantic-settings

API keys (from .env)
Model configurations (temperature, max_tokens, timeouts)
Feature flags (enable_caching, enable_streaming)


logger.py: Structured logging with structlog

JSON-formatted logs for production parsing
Request correlation IDs
Automatic latency and cost tracking


cache.py: In-memory LRU cache for repeated prompts

TTL-based expiration
Cost savings tracking



Data Flow
Standard Chat Request
┌──────┐      ┌─────────┐      ┌──────────────┐      ┌──────────┐
│Client│─────▶│FastAPI  │─────▶│Orchestrator  │─────▶│LLM Client│
│      │      │(routes) │      │(strategy)    │      │(OpenAI)  │
└──────┘      └─────────┘      └──────────────┘      └──────────┘
   ▲                │                  │                    │
   │                │                  │                    │
   │                ▼                  ▼                    ▼
   │          ┌─────────┐      ┌──────────────┐      ┌──────────┐
   └──────────│Response │◀─────│Aggregate     │◀─────│Response  │
              │Model    │      │Results       │      │Data      │
              └─────────┘      └──────────────┘      └──────────┘
Parallel Orchestration
                    ┌─────────────┐
            ┌──────▶│OpenAI Client│─────┐
            │       └─────────────┘     │
┌───────────┴──┐                        ├──▶ asyncio.gather()
│Orchestrator  │                        │
└───────────┬──┘                        │
            │       ┌──────────────────┐│
            └──────▶│Anthropic Client  ││
                    └──────────────────┘│
                                        ▼
                                  ┌──────────┐
                                  │Return    │
                                  │Fastest   │
                                  └──────────┘
Streaming Response
Client ◀─── SSE Stream ◀─── FastAPI ◀─── async for chunk ◀─── LLM API
  │                           │                                  │
  │ data: {"token": "Hello"}  │       yield {"token": "Hello"}  │
  │ data: {"token": " "}      │       yield {"token": " "}      │
  │ data: {"token": "world"}  │       yield {"token": "world"}  │
  │ data: [DONE]              │       return                    │
Key Patterns
1. Dependency Injection
python# Allows easy mocking in tests
@app.post("/chat")
async def chat(
    request: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    return await orchestrator.execute(request)
2. Circuit Breaker (Future Enhancement)
python# Track failures, open circuit after threshold
if failure_rate > 0.5:
    skip_provider("openai", duration=60)  # 60-second cooldown
3. Cost Tracking
python# Automatic cost calculation per request
cost = (prompt_tokens / 1M * $0.03) + (completion_tokens / 1M * $0.06)
logger.info("llm_call", model="gpt-4", cost=cost, latency=1.23)
Technology Stack
LayerTechnologyJustificationAPIFastAPIAsync-native, auto-docs, modern PythonLLM SDKsOpenAI SDK, Anthropic SDKOfficial clients, streaming supportValidationPydanticType safety, auto-validation, serializationAsyncasyncioNon-blocking I/O for parallel LLM callsLoggingstructlogStructured JSON logs for productionTestingpytest, pytest-asyncioIndustry standard, async supportEnv ManagementPixiReproducible environments, fast installs
Configuration
Environment Variables (.env)
bash# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Defaults
DEFAULT_MODEL=gpt-4-turbo
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=500

# Infrastructure
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
Model Configuration (config.py)
pythonMODELS = {
    "gpt-4-turbo": {
        "provider": "openai",
        "cost_per_1m_prompt": 0.03,
        "cost_per_1m_completion": 0.06,
        "timeout": 30
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "cost_per_1m_prompt": 0.015,
        "cost_per_1m_completion": 0.075,
        "timeout": 30
    }
}
Error Handling Strategy
Error Types

Validation Errors (400): Invalid request format
Authentication Errors (401): Missing/invalid API keys
Rate Limit Errors (429): Provider quota exceeded
Timeout Errors (504): LLM took too long
Provider Errors (502): Upstream API failure
Server Errors (500): Unexpected exceptions

Retry Logic
python@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def call_llm(prompt: str):
    ...
Security Considerations

API Keys: Never logged, stored only in .env (gitignored)
Input Validation: Pydantic prevents injection attacks
Rate Limiting: Prevents abuse (10 requests/minute per IP)
CORS: Configurable allowed origins
Request Size Limits: Max 10KB request body

Performance Targets
MetricTargetMeasurementP50 Latency< 2sTime to first tokenP99 Latency< 10sEnd-to-end responseThroughput100 req/secWith parallel orchestrationCache Hit Rate> 30%For repeated promptsError Rate< 1%Non-timeout errors
Future Enhancements
Phase 1 (Current)

✅ Basic chat endpoint
✅ OpenAI + Anthropic support
✅ Structured logging
✅ Health checks

Phase 2 (Next)

 Streaming responses
 Parallel orchestration
 Response caching
 Cost tracking dashboard

Phase 3 (Future)

 Circuit breaker pattern
 A/B testing framework
 Prompt template library
 WebSocket support
 Additional providers (Cohere, AI21)

Observability
Metrics to Track

Latency: P50, P95, P99 by provider/model
Cost: $ per request, daily spend
Errors: Rate by type (timeout, auth, provider)
Cache: Hit rate, memory usage

Log Format (JSON)
json{
  "timestamp": "2024-01-15T10:30:45Z",
  "level": "info",
  "event": "llm_call",
  "request_id": "req_abc123",
  "model": "gpt-4-turbo",
  "latency_ms": 1234,
  "cost_usd": 0.0045,
  "prompt_tokens": 150,
  "completion_tokens": 300
}
Testing Strategy
Unit Tests

Mock external LLM calls
Test request validation
Test error handling

Integration Tests

Real API calls (small prompts)
End-to-end flow validation
Rate limit testing

Load Tests

Simulate 100 concurrent requests
Measure latency under load
Identify bottlenecks
