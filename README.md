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
Pixi (recommended) or pip
OpenAI API key

1. Clone the Repository
```bash
git clone https://github.com/yourusername/llm-orchestration-demo.git
cd llm-orchestration-demo
```
2. Set Up Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=sk-...
```
3. Install Dependencies (Pixi recommended)
```bash
pixi install
# or
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```
4. Run the Server
```bash
pixi run dev
# or
uvicorn src.api.main:app --reload
```
Server will start at: http://localhost:8000

5. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, world!", "model": "gpt-4o"}'
```

6. View Interactive Docs
Open in your browser: http://localhost:8000/docs

🏗️ Project Structure
llm-orchestration-demo/
├── src/
│   ├── api/                  # FastAPI application
│   │   ├── main.py           # App wiring, middleware, handlers
│   │   ├── routes.py         # Endpoint definitions
│   │   ├── models.py         # Pydantic schemas
│   │   ├── middleware.py     # Optional middleware utilities
│   │   └── health.py         # Health/metrics helpers
│   ├── llm/                  # LLM orchestration
│   │   ├── clients.py        # OpenAI client wrapper
│   │   └── orchestrator.py   # Parallel/fallback/streaming orchestration
│   └── utils/                # Utilities
│       ├── config.py         # Settings & environment
│       └── logger.py         # Structured logging
├── tests/                    # Pytest test suite
│   ├── test_api.py          # API endpoint tests
│   ├── test_llm.py          # LLM orchestration tests
│   └── conftest.py          # Shared fixtures
├── examples/                 # Working code examples
│   ├── basic_call.py        # Simple chat example
│   ├── fallback_patterns.py # Fallback usage patterns
│   ├── parallel_calls.py    # Parallel orchestration
│   ├── streaming.py         # Streaming responses
│   └── test_client.py       # Minimal client usage
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md      # System design
│   └── api_specs.md         # API specifications
├── .vscode/                  # VS Code configuration
│   ├── settings.json        # Editor settings
│   ├── tasks.json           # Build tasks
│   └── keybindings.json     # Custom shortcuts
├── pixi.toml                # Pixi dependencies
├── pyproject.toml           # Python project config
├── .env.example             # Environment template
├── COPILOT.md               # AI assistant instructions
└── README.md                # This file

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
Run All Tests (Pixi)
```bash
pixi run test
```
Run With Coverage (terminal summary)
```bash
pixi run test --cov=src --cov-report=term
```
Run Specific Tests
```bash
pytest tests/test_api.py -v
pytest tests/test_llm.py -v
```
Watch Mode (Auto-rerun on file changes)
```bash
pixi run test-watch
```

🔧 Development
VS Code Shortcuts
ShortcutActionCtrl+Shift+RStart development serverCtrl+Shift+TRun testsCtrl+Shift+FFormat & lint codeCtrl+Shift+DOpen API docs in browser
Code Formatting
bash# Format code
pixi run format

# Lint code
pixi run lint

# Type check
pixi run type-check
Adding Dependencies
bash# Add a new package
pixi add <package-name>

# Add a development dependency
pixi add --feature dev <package-name>

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

Architecture: docs/ARCHITECTURE.md
API Specs: docs/api_specs.md
Interactive Docs: http://localhost:8000/docs (when running)
Copilot Instructions: COPILOT.md


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
