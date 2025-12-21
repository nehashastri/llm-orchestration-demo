API Specifications
Base URL
Development: http://localhost:8000
Production: https://api.your-domain.com
Authentication
Currently open (no auth). Future: API key via X-API-Key header.

Endpoints
1. Health Check
GET /health
Check if the service is running and healthy.
Request
bashcurl http://localhost:8000/health
Response 200 OK
json{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45Z",
  "version": "0.1.0",
  "providers": {
    "openai": "connected",
    "anthropic": "connected"
  }
}
Error Response 503 Service Unavailable
json{
  "status": "unhealthy",
  "timestamp": "2024-01-15T10:30:45Z",
  "providers": {
    "openai": "error",
    "anthropic": "connected"
  },
  "error": "OpenAI API connection failed"
}

2. Chat Completion
POST /chat
Generate a chat completion using a single LLM provider.
Request Body
json{
  "prompt": "Explain quantum computing in simple terms",
  "model": "gpt-4-turbo",
  "temperature": 0.7,
  "max_tokens": 500,
  "system_prompt": "You are a helpful assistant."
}
Field Descriptions
FieldTypeRequiredDefaultDescriptionpromptstringYes-User message/questionmodelstringNogpt-4-turboModel to use (gpt-4-turbo, claude-3-opus, etc.)temperaturefloatNo0.7Randomness (0.0-2.0)max_tokensintegerNo500Max response length (1-4000)system_promptstringNonullSystem instructions
Response 200 OK
json{
  "content": "Quantum computing is like having a super-powered calculator...",
  "model": "gpt-4-turbo",
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
Error Responses
400 Bad Request - Invalid input
json{
  "error": "validation_error",
  "message": "temperature must be between 0.0 and 2.0",
  "field": "temperature"
}
401 Unauthorized - Missing API key
json{
  "error": "authentication_error",
  "message": "OpenAI API key not configured"
}
429 Too Many Requests - Rate limit exceeded
json{
  "error": "rate_limit_error",
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
504 Gateway Timeout - LLM took too long
json{
  "error": "timeout_error",
  "message": "Request timed out after 30 seconds"
}

3. Parallel Orchestration
POST /chat/parallel
Call multiple LLM providers simultaneously and return the fastest response.
Request Body
json{
  "prompt": "Write a haiku about coding",
  "providers": ["openai", "anthropic"],
  "temperature": 0.8,
  "max_tokens": 100
}
Field Descriptions
FieldTypeRequiredDefaultDescriptionpromptstringYes-User message/questionprovidersarrayNo["openai", "anthropic"]List of providers to querytemperaturefloatNo0.7Randomness (0.0-2.0)max_tokensintegerNo500Max response length (1-4000)
Response 200 OK
json{
  "content": "Code flows like streams\nBugs dance in moonlit debug\nMerge conflicts resolved",
  "winner": {
    "provider": "anthropic",
    "model": "claude-3-opus",
    "latency_ms": 987
  },
  "all_responses": [
    {
      "provider": "openai",
      "model": "gpt-4-turbo",
      "latency_ms": 1543,
      "content": "Functions compile\nAsync awaits completion\nCode deploys at dawn"
    },
    {
      "provider": "anthropic",
      "model": "claude-3-opus",
      "latency_ms": 987,
      "content": "Code flows like streams\nBugs dance in moonlit debug\nMerge conflicts resolved"
    }
  ],
  "metrics": {
    "total_latency_ms": 1543,
    "total_cost_usd": 0.0092
  }
}

4. Fallback Orchestration
POST /chat/fallback
Try primary provider, automatically fallback to secondary if it fails.
Request Body
json{
  "prompt": "What is the capital of France?",
  "primary_provider": "openai",
  "fallback_providers": ["anthropic"],
  "timeout": 10,
  "temperature": 0.3
}
Response 200 OK
json{
  "content": "The capital of France is Paris.",
  "provider_used": "openai",
  "primary_success": true,
  "fallback_triggered": false,
  "metrics": {
    "latency_ms": 876,
    "cost_usd": 0.0012
  }
}
Response (with fallback) 200 OK
json{
  "content": "The capital of France is Paris.",
  "provider_used": "anthropic",
  "primary_success": false,
  "fallback_triggered": true,
  "primary_error": "timeout",
  "metrics": {
    "latency_ms": 11234,
    "cost_usd": 0.0015
  }
}

5. Streaming Chat
POST /chat/stream
Stream chat completion token-by-token using Server-Sent Events (SSE).
Request Body
json{
  "prompt": "Tell me a short story",
  "model": "gpt-4-turbo",
  "temperature": 0.9
}
Response 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"token": "Once", "index": 0}

data: {"token": " upon", "index": 1}

data: {"token": " a", "index": 2}

data: {"token": " time", "index": 3}

data: [DONE]
Client Example (JavaScript)
javascriptconst eventSource = new EventSource('/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ prompt: "Tell me a joke" })
});

eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const data = JSON.parse(event.data);
  console.log(data.token);
};

6. List Models
GET /models
Get available LLM models and their configurations.
Response 200 OK
json{
  "models": [
    {
      "id": "gpt-4-turbo",
      "provider": "openai",
      "max_tokens": 4096,
      "cost_per_1m_prompt_tokens": 0.03,
      "cost_per_1m_completion_tokens": 0.06,
      "supports_streaming": true
    },
    {
      "id": "claude-3-opus",
      "provider": "anthropic",
      "max_tokens": 4096,
      "cost_per_1m_prompt_tokens": 0.015,
      "cost_per_1m_completion_tokens": 0.075,
      "supports_streaming": true
    }
  ]
}

7. Request Statistics
GET /stats
Get API usage statistics (for monitoring/debugging).
Response 200 OK
json{
  "total_requests": 1523,
  "requests_by_provider": {
    "openai": 892,
    "anthropic": 631
  },
  "average_latency_ms": 1234,
  "total_cost_usd": 45.67,
  "cache_hit_rate": 0.32,
  "error_rate": 0.008,
  "uptime_seconds": 86400
}

Error Response Format
All errors follow this structure:
json{
  "error": "error_type",
  "message": "Human-readable error description",
  "field": "field_name (if validation error)",
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:45Z"
}
Rate Limiting

Limit: 10 requests per minute per IP address
Headers:

X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1705318800



Request IDs
Every request gets a unique ID for tracing:

Response header: X-Request-ID: req_abc123
Included in all error responses
Use for support/debugging

OpenAPI/Swagger
Interactive API documentation available at:

Development: http://localhost:8000/docs
Alternative: http://localhost:8000/redoc

Testing the API
Using curl
bash# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "model": "gpt-4-turbo",
    "temperature": 0.7
  }'

# Parallel orchestration
curl -X POST http://localhost:8000/chat/parallel \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "providers": ["openai", "anthropic"]
  }'
Using Python
pythonimport requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "prompt": "Explain async programming",
        "model": "gpt-4-turbo",
        "temperature": 0.5
    }
)

print(response.json()["content"])
Using JavaScript/Node.js
javascriptconst response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Write a function to reverse a string',
    model: 'gpt-4-turbo'
  })
});

const data = await response.json();
console.log(data.content);
Versioning
Current API version: v1
Future versions will use URL prefix:

v1: http://localhost:8000/v1/chat
v2: http://localhost:8000/v2/chat
