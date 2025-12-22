API Specifications

Base URL
- Development: http://localhost:8000

Authentication
- Currently open (no auth). API keys are passed to the OpenAI SDK via environment variable only.

Endpoints
1) Health Check
GET /health
- Purpose: report service and provider status.
- Success 200
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45Z",
  "version": "0.1.0",
  "providers": {
    "openai": {"status": "connected"}
  }
}

2) Chat Completion
POST /chat
- Single OpenAI call.
Request
{
  "prompt": "Explain quantum computing in simple terms",
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 500,
  "system_prompt": "You are a helpful assistant."
}
Response 200
{
  "content": "Quantum computing is like...",
  "model": "gpt-4o",
  "provider": "openai",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "metrics": {
    "latency_ms": 1234,
    "cost_usd": 0.0048
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:45Z"
  }
}

3) Parallel Orchestration
POST /chat/parallel
- Fan out to multiple OpenAI models; return fastest success.
Request
{
  "prompt": "Write a haiku about coding",
  "version": 1,
  "models": ["gpt-4o", "gpt-4o-mini"],
  "temperature": 0.7,
  "max_tokens": 120
}
Response 200
{
  "content": "Code flows like streams...",
  "winner": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "latency_ms": 950
  },
  "all_responses": [
    {
      "content": "Code flows like streams...",
      "model": "gpt-4o-mini",
      "provider": "openai",
      "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
      "metrics": {"latency_ms": 950, "cost_usd": 0.0005},
      "metadata": {"request_id": "req_abc123", "timestamp": "2024-01-15T10:30:45Z"}
    },
    {
      "content": "Async awaits completion...",
      "model": "gpt-4o",
      "provider": "openai",
      "usage": {"prompt_tokens": 12, "completion_tokens": 28, "total_tokens": 40},
      "metrics": {"latency_ms": 1230, "cost_usd": 0.0021},
      "metadata": {"request_id": "req_abc123", "timestamp": "2024-01-15T10:30:45Z"}
    }
  ],
  "errors": null,
  "metrics": {
    "total_latency_ms": 1230,
    "total_cost_usd": 0.0026,
    "num_providers_called": 2,
    "num_successful": 2,
    "num_failed": 0
  }
}

4) Fallback Orchestration
POST /chat/fallback
- Try primary OpenAI model, fall back to smaller OpenAI models on error/timeout.
Request
{
  "prompt": "What is the capital of France?",
  "primary_provider": "openai",
  "primary_model": "gpt-4o",
  "timeout": 15,
  "temperature": 0.3
}
Response 200 (primary success)
{
  "content": "The capital of France is Paris.",
  "provider": "openai",
  "provider_used": "openai",
  "primary_success": true,
  "fallback_triggered": false,
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
  "metrics": {"latency_ms": 876, "cost_usd": 0.0012}
}
Response 200 (fallback)
{
  "content": "The capital of France is Paris.",
  "provider": "openai",
  "provider_used": "openai",
  "primary_success": false,
  "fallback_triggered": true,
  "primary_error": "timeout",
  "fallback_error": null,
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
  "metrics": {"latency_ms": 1230, "cost_usd": 0.0009}
}

5) Streaming Chat
POST /chat/stream
- Server-Sent Events (SSE) stream of tokens.
Request
{
  "prompt": "Tell me a short story",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.9
}
Response (text/event-stream)
data: {"token": "Once", "index": 0}

data: {"token": " upon", "index": 1}

data: {"token": " a", "index": 2}

data: {"token": " time", "index": 3}

data: [DONE]

6) List Models
GET /models
- Return available OpenAI models and pricing metadata.
Response 200
{
  "models": [
    {"id": "gpt-4o", "provider": "openai", "max_tokens": 128000, "cost_per_1m_prompt_tokens": 5.0, "cost_per_1m_completion_tokens": 15.0, "supports_streaming": true, "timeout": 30},
    {"id": "gpt-4o-mini", "provider": "openai", "max_tokens": 128000, "cost_per_1m_prompt_tokens": 0.15, "cost_per_1m_completion_tokens": 0.6, "supports_streaming": true, "timeout": 30},
    {"id": "gpt-4-turbo", "provider": "openai", "max_tokens": 128000, "cost_per_1m_prompt_tokens": 0.03, "cost_per_1m_completion_tokens": 0.06, "supports_streaming": true, "timeout": 30},
    {"id": "gpt-3.5-turbo", "provider": "openai", "max_tokens": 4096, "cost_per_1m_prompt_tokens": 0.001, "cost_per_1m_completion_tokens": 0.002, "supports_streaming": true, "timeout": 30}
  ]
}

7) Request Statistics
GET /stats
- Aggregated in-memory counters.
Response 200
{
  "total_requests": 1523,
  "requests_by_provider": {"openai": 1523},
  "average_latency_ms": 1234,
  "total_cost_usd": 4.567,
  "cache_hit_rate": 0.0,
  "error_rate": 0.008,
  "uptime_seconds": 86400
}

Error Format
All errors follow this structure:
{
  "error": "error_type",
  "message": "Human-readable error description",
  "field": "field_name (if validation error)",
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:45Z"
}

OpenAPI/Swagger
- Interactive docs: http://localhost:8000/docs
- Alternative: http://localhost:8000/redoc

Testing Snippets
bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "model": "gpt-4o",
    "temperature": 0.7
  }'

# Parallel orchestration
curl -X POST http://localhost:8000/chat/parallel \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "version": 1
  }'
