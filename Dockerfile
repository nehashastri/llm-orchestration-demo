# Multi-stage Dockerfile for LLM Orchestration API
# Base image: Python 3.11 slim (Debian-based, smaller than full Python)
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for any Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
# Note: We extract dependencies from pyproject.toml and install via pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.32.0 \
    pydantic>=2.10.0 \
    pydantic-settings>=2.6.0 \
    openai>=1.55.0 \
    anthropic>=0.39.0 \
    httpx>=0.27.0 \
    python-dotenv>=1.0.0 \
    python-multipart>=0.0.12 \
    structlog>=25.5.0 \
    redis>=5.0.0

# Copy application source code
COPY src/ ./src/

# Create logs directory
RUN mkdir -p logs

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn server
# Single worker for simplicity (can be scaled horizontally with multiple containers)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
