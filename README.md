Production-ready FastAPI service for orchestrating multiple LLM providers with intelligent routing, parallel execution, and fallback strategies.

✨ Features

🔄 OpenAI Integration: GPT-4 and GPT-3.5 Turbo models
⚡ Async-First Architecture: Non-blocking I/O for maximum throughput
🎯 Intelligent Fallback: Automatic fallback to cheaper models with default message safety net
📊 Built-in Observability: Structured logging, cost tracking, latency monitoring
🛡️ Production-Ready: Error handling, rate limiting, request validation
📚 Auto-Generated Docs: Interactive Swagger UI at /docs


🎬 Quick Start
Prerequisites

Python 3.11+
Pixi (recommended) or pip
OpenAI API key

1. Clone the Repository
bashgit clone https://github.com/yourusername/llm-orchestration-demo.git
cd llm-orchestration-demo
2. Set Up Environment
bash# Copy example env file
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=sk-...
3. Install Dependencies
Option A: Using Pixi (Recommended)
bashpixi install
Option B: Using pip
bashpython -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
4. Run the Server
bash# Using Pixi
pixi run dev

# Using uvicorn directly
uvicorn src.api.main:app --reload
Server will start at: http://localhost:8000
5. Test the API
bash# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model": "gpt-4-turbo"}'
6. View Interactive Docs
Open in your browser: http://localhost:8000/docs

🏗️ Project Structure
llm-orchestration-demo/
├── src/
│   ├── api/                  # FastAPI application
│   │   ├── main.py          # App factory, middleware
│   │   ├── routes.py        # Endpoint definitions
│   │   ├── models.py        # Pydantic schemas
│   │   └── middleware.py    # Logging, rate limiting
│   ├── llm/                  # LLM orchestration
│   │   ├── clients.py       # Provider-specific clients
│   │   ├── orchestrator.py  # Orchestration strategies
│   │   └── models.py        # LLM request/response models
│   └── utils/                # Utilities
│       ├── config.py        # Settings & environment
│       ├── logger.py        # Structured logging
│       └── cache.py         # Response caching
├── tests/                    # Pytest test suite
│   ├── test_api.py          # API endpoint tests
│   ├── test_llm.py          # LLM orchestration tests
│   └── conftest.py          # Shared fixtures
├── examples/                 # Working code examples
│   ├── basic_chat.py        # Simple chat example
│   ├── parallel_calls.py    # Parallel orchestration
│   └── streaming.py         # Streaming responses
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
## Parallel Testing (Disabled)
**Note**: Parallel orchestration endpoint has been disabled in OpenAI-only mode.
Use fallback orchestration for reliability instead.
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
Run All Tests
bash# Using Pixi
pixi run test

# Using pytest directly
pytest tests/ -v
Run Specific Tests
bash# Test API endpoints only
pytest tests/test_api.py -v

# Test LLM orchestration only
pytest tests/test_llm.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
Watch Mode (Auto-rerun on file changes)
bashpixi run test-watch

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
View Statistics
bashcurl http://localhost:8000/stats
Returns:

Total requests
Requests by provider
Average latency
Total cost
Cache hit rate
Error rate


🚀 Deployment
Environment Variables
bash# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (with defaults)
ENVIRONMENT=production
LOG_LEVEL=INFO
DEFAULT_MODEL=gpt-4-turbo
DEFAULT_TEMPERATURE=0.7
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
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
Anthropic - Claude models
