# Docker Deployment Guide

This guide explains how to deploy the LLM Orchestration API using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- OpenAI API Key

## Quick Start

1. **Set your OpenAI API key** (create `.env` file in project root):
```bash
OPENAI_API_KEY=your-api-key-here
```

2. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

3. **Access the API**:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Architecture

The deployment consists of two services:

### 1. **API Service** (`api`)
- FastAPI application running on uvicorn
- Single worker (scale horizontally with multiple containers)
- Port: 8000
- Based on `python:3.11-slim`

### 2. **Redis Service** (`redis`)
- Redis 7 Alpine
- Used for distributed rate limiting
- Port: 6379
- **Ephemeral storage** (data lost on restart)

## Environment Variables

### Required
- `OPENAI_API_KEY` - Your OpenAI API key

### Optional (with defaults)
- `ENVIRONMENT` - development/production/testing (default: production)
- `LOG_LEVEL` - DEBUG/INFO/WARNING/ERROR (default: INFO)
- `REDIS_HOST` - Redis hostname (default: redis)
- `REDIS_PORT` - Redis port (default: 6379)
- `REDIS_DB` - Redis database number (default: 0)
- `DEFAULT_MODEL` - Default LLM model (default: gpt-4o)
- `DEFAULT_TEMPERATURE` - Temperature 0.0-2.0 (default: 0.7)
- `DEFAULT_MAX_TOKENS` - Max tokens (default: 500)
- `DEFAULT_TIMEOUT` - Request timeout seconds (default: 30)
- `RATE_LIMIT_REQUESTS` - Requests per window (default: 100)
- `RATE_LIMIT_WINDOW` - Window in seconds (default: 3600)

## Docker Commands

### Build the image
```bash
docker-compose build
```

### Start services (detached)
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f api
docker-compose logs -f redis
```

### Stop services
```bash
docker-compose down
```

### Stop and remove volumes
```bash
docker-compose down -v
```

### Rebuild from scratch
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Manual Docker Commands (without Compose)

### Build image
```bash
docker build -t llm-orchestration-api:latest .
```

### Run Redis
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Run API
```bash
docker run -d \
  --name llm-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e REDIS_HOST=redis \
  --link redis:redis \
  llm-orchestration-api:latest
```

## Health Checks

The API includes three health check endpoints:

1. **Basic Health**: `GET /health`
2. **Liveness**: `GET /health/live` (for Kubernetes liveness probe)
3. **Readiness**: `GET /health/ready` (for Kubernetes readiness probe)

Docker health checks run automatically every 30 seconds.

## Scaling

### Horizontal Scaling with Docker Compose

Scale to 3 API instances:
```bash
docker-compose up -d --scale api=3
```

**Note**: You'll need a load balancer (nginx, traefik, etc.) to distribute traffic.

### Kubernetes Deployment

For production Kubernetes deployment:
1. Push image to container registry
2. Create Kubernetes manifests (Deployment, Service, Ingress)
3. Use Redis Cluster or managed Redis (ElastiCache, Cloud Memorystore)
4. Configure HPA (Horizontal Pod Autoscaler)

## Production Considerations

### 1. **Rate Limiting**
- Currently using Redis with ephemeral storage
- For production, consider:
  - Redis persistence (RDB or AOF)
  - Managed Redis service (AWS ElastiCache, GCP Memorystore)
  - Redis Sentinel for high availability
  - Redis Cluster for scalability

### 2. **Security**
- Set `REDIS_PASSWORD` for Redis authentication
- Use secrets management (Docker secrets, Kubernetes secrets, Vault)
- Enable TLS for Redis connection
- Run containers as non-root user
- Scan images for vulnerabilities

### 3. **Monitoring**
- Logs are output to stdout/stderr (captured by Docker)
- Integrate with logging aggregators (ELK, Splunk, Datadog)
- Add Prometheus metrics endpoint
- Configure APM (Application Performance Monitoring)

### 4. **Resource Limits**

Add to `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

### 5. **Multi-Worker Uvicorn**

For higher throughput on a single container:
```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Note**: Workers = (2 × CPU cores) + 1

### 6. **Persistent Redis**

Modify `docker-compose.yml` to add volume:
```yaml
redis:
  volumes:
    - redis-data:/data
  command: redis-server --save 60 1 --loglevel warning

volumes:
  redis-data:
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs api
```

### Redis connection failed
```bash
docker-compose exec redis redis-cli ping
# Should return: PONG
```

### Check API health
```bash
curl http://localhost:8000/health
```

### Access container shell
```bash
docker-compose exec api /bin/bash
```

### Check Redis keys
```bash
docker-compose exec redis redis-cli
> KEYS rate_limit:*
> ZRANGE rate_limit:127.0.0.1 0 -1 WITHSCORES
```

## Development vs Production

### Development (with Pixi)
```bash
pixi run dev
```

### Production (with Docker)
```bash
docker-compose up
```

This separation allows developers to use Pixi for fast iteration while deploying with Docker for consistency across environments.
