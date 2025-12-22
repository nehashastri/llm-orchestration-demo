Production-ready FastAPI service for orchestrating OpenAI models with parallel execution and fallback strategies.

✨ Features

🔄 OpenAI Integration: GPT-4o/4o-mini/3.5-turbo models
⚡ Async-First Architecture: Non-blocking I/O for maximum throughput
🎯 Intelligent Fallback: Automatic fallback within the OpenAI model family with default message safety net
📊 Built-in Observability: Structured logging, cost tracking, latency monitoring
🛡️ Production-Ready: Error handling, rate limiting, request validation
📚 Auto-Generated Docs: Interactive Swagger UI at /docs


🎬 Quick Start
Prerequisites

Python 3.11+
Pixi
OpenAI API key

1. Clone the Repository
```bash
git clone https://github.com/yourusername/llm-orchestration-demo.git
cd llm-orchestration-demo
```

2. Install Dependencies
```bash
pixi install
```

3. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. Run the Server
```bash
pixi run dev
```
Server starts at: http://localhost:8000

5. View Interactive Docs
```bash
start http://localhost:8000/docs
```

Project Overview

**Key Directories:**
- `src/api/` — FastAPI application (routes, models, middleware)
- `src/llm/` — LLM orchestration (clients, orchestrator strategies)
- `src/utils/` — Configuration, logging, utilities
- `tests/` — Pytest test suite
- `examples/` — Working code examples (basic, parallel, streaming, fallback)
- `docs/` — Architecture and API specifications

🎯 Usage Examples
Basic Chat
pythonimport requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "prompt": "Explain async programming in Python",
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 500
    }
)

print(response.json()["content"])
## Parallel Orchestration
Runs multiple OpenAI models in parallel (e.g., gpt-4o vs gpt-4o-mini) and returns the fastest success.
Fallback Strategy
python# Automatic fallback: gpt-4-turbo → gpt-3.5-turbo → default message
response = requests.post(
    "http://localhost:8000/chat/fallback",
    json={
        "prompt": "What is quantum computing?",
        "primary_provider": "openai",
        "primary_model": "gpt-4-turbo",
        "timeout": 10
    }
)

result = response.json()
print(f"Provider used: {result['provider_used']}")
print(f"Fallback triggered: {result['fallback_triggered']}")
print(f"Is default message: {result['is_default_message']}")
Streaming Response
pythonimport sseclient
import requests

response = requests.post(
    "http://localhost:8000/chat/stream",
    json={"prompt": "Tell me a story"},
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.data == '[DONE]':
        break
    print(event.data, end='', flush=True)

🧪 Testing
```bash
pixi run test                    # Run full suite
pixi run test-cov                # With HTML coverage report
pixi run -e dev pytest tests/test_api.py  # Single test file
```

🔧 Development
```bash
pixi run format                  # Format code + markdown
pixi run lint                    # Lint and auto-fix
pixi run typecheck               # Type checking
pixi run check                   # All checks: format, lint, typecheck
```

For comprehensive development guidance, see [COPILOT.md](COPILOT.md) — covers patterns, testing strategy, error handling, and documentation standards.

📊 Monitoring & Observability
Structured Logging
All requests are logged in JSON format:
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
Log Files
The service writes logs to the `logs` directory with daily rotation:
- Active file: `logs/app.log`
- Rotated files: `logs/app-YYYY-MM-DD.log` (e.g., `app-2025-12-21.log`)
At midnight, the current `app.log` rolls over to the dated file, so yesterday's logs are always available as a separate file.
View Statistics
bashcurl http://localhost:8000/stats
Returns:

Total requests
Average latency
Total cost (OpenAI only)
Error rate


🚀 Deployment
Environment Variables
bash# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (with defaults)
ENVIRONMENT=production
LOG_LEVEL=INFO
DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.7
Docker (Coming Soon)
bashdocker build -t llm-orchestration .
Production Checklist

 Set ENVIRONMENT=production in .env
 Configure API key rotation
Set up monitoring (Prometheus, Grafana)
Enable HTTPS
Configure rate limiting per user/API key
Set up log aggregation (ELK, Datadog)
Add authentication/authorization
Configure CORS allowed origins


📚 Documentation

- **Development Guide:** [COPILOT.md](COPILOT.md) — Patterns, code style, testing, how-to guides
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design and data flows
- **API Specs:** [docs/api_specs.md](docs/api_specs.md) — Endpoint reference
- **Interactive Docs:** http://localhost:8000/docs (when running)


🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Make your changes
Run tests (pixi run test)
Format code (pixi run format)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request

Development Setup
bash# Install pre-commit hooks
pixi run pre-commit install

# Run all checks before committing
pixi run pre-commit run --all-files

📝 License
This project is licensed under the MIT License - see LICENSE file for details.

🙏 Acknowledgments

FastAPI - Modern web framework
OpenAI - GPT models
