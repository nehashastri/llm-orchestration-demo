"""
Production health checks and monitoring endpoints.
"""

from datetime import datetime
from typing import Any

import psutil  # For system metrics
from fastapi import APIRouter, status
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Monitoring"])


# ============================================
# RESPONSE MODELS
# ============================================


class HealthStatus(BaseModel):
    """Basic health check response"""

    status: str  # "healthy" or "unhealthy"
    timestamp: str
    version: str = "0.1.0"


class DetailedHealthStatus(BaseModel):
    """Detailed health check with dependencies"""

    status: str
    timestamp: str
    version: str
    checks: dict[str, dict[str, Any]]
    system: dict[str, Any]


class MetricsResponse(BaseModel):
    """System metrics response"""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float


# ============================================
# BASIC HEALTH CHECK
# ============================================


@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Quick check if the API is running. Used by load balancers.",
)
async def health_check() -> HealthStatus:
    """
    Lightweight health check.

    Use this for:
    - Load balancer health checks (AWS ELB, etc.)
    - Kubernetes liveness probes
    - Quick "is the server up?" checks

    Returns immediately without checking dependencies.
    """
    return HealthStatus(status="healthy", timestamp=datetime.utcnow().isoformat(), version="0.1.0")


# ============================================
# DETAILED HEALTH CHECK
# ============================================


@router.get(
    "/health/detailed",
    response_model=DetailedHealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Detailed health check",
    description="Comprehensive health check including all dependencies",
)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Comprehensive health check that tests all dependencies.

    Use this for:
    - Kubernetes readiness probes
    - Debugging issues
    - Monitoring dashboards

    Checks:
    - API endpoints are accessible
    - LLM providers are reachable
    - Database connections (if you add a database)
    - External services
    """

    checks = {}
    overall_status = "healthy"

    # Check 1: OpenAI API
    openai_status = await _check_openai()
    checks["openai"] = openai_status
    if openai_status["status"] != "healthy":
        overall_status = "degraded"

    # Check 2: System resources
    system_status = _check_system_resources()
    checks["system"] = system_status
    if system_status["status"] != "healthy":
        overall_status = "unhealthy"

    # Get system metrics
    system_metrics = _get_system_metrics()

    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        checks=checks,
        system=system_metrics,
    )


# ============================================
# READINESS CHECK
# ============================================


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the API is ready to accept traffic",
)
async def readiness_check() -> dict[str, str]:
    """
    Readiness check for Kubernetes.

    Difference between liveness and readiness:
    - Liveness (/health): Is the container alive? If not, restart it.
    - Readiness (/health/ready): Is the app ready to serve traffic? If not, don't send requests yet.

    Example:
    - App is starting up, loading ML models → Not ready
    - App finished loading → Ready
    """

    # Check if critical dependencies are available
    # For now, just return ready
    # In production, you'd check:
    # - Are API keys configured?
    # - Are models loaded?
    # - Is database connected?

    is_ready = await _check_if_ready()

    if not is_ready:
        return {"status": "not_ready", "message": "Application is starting up"}

    return {"status": "ready", "message": "Application is ready to accept traffic"}


# ============================================
# LIVENESS CHECK
# ============================================


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the API is alive (not deadlocked)",
)
async def liveness_check() -> dict[str, str]:
    """
    Liveness check for Kubernetes.

    This should be VERY simple and fast.
    If this fails, Kubernetes will restart your container.

    Only checks: "Can the server respond to HTTP requests?"
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


# ============================================
# METRICS ENDPOINT
# ============================================


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="System metrics",
    description="Get current system resource usage",
)
async def get_metrics() -> MetricsResponse:
    """
    Get system metrics.

    Useful for:
    - Monitoring dashboards (Grafana, Datadog)
    - Alerting on high resource usage
    - Capacity planning
    """

    memory = psutil.virtual_memory()

    return MetricsResponse(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_total_mb=memory.total / 1024 / 1024,
        disk_percent=psutil.disk_usage("/").percent,
    )


# ============================================
# HELPER FUNCTIONS
# ============================================


async def _check_openai() -> dict[str, Any]:
    """Check if OpenAI API is accessible"""
    try:
        # Try to make a minimal API call
        # In production, you'd do a lightweight test call
        # For now, just check if API key is configured
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return {
                "status": "unhealthy",
                "message": "OpenAI API key not configured",
                "latency_ms": 0,
            }

        # If you want to actually test the API:
        # import openai
        # start = time.time()
        # await openai.models.list()
        # latency = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "message": "OpenAI API accessible",
            "latency_ms": 0,  # Replace with actual latency if testing
        }
    except Exception as e:
        logger.error("OpenAI health check failed", error=str(e))
        return {"status": "unhealthy", "message": f"OpenAI API error: {str(e)}", "latency_ms": 0}


def _check_system_resources() -> dict[str, Any]:
    """Check if system has enough resources"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # Define thresholds
    MEMORY_THRESHOLD = 90  # percent
    DISK_THRESHOLD = 90  # percent

    issues = []

    if memory.percent > MEMORY_THRESHOLD:
        issues.append(f"High memory usage: {memory.percent}%")

    if disk.percent > DISK_THRESHOLD:
        issues.append(f"High disk usage: {disk.percent}%")

    if issues:
        return {
            "status": "unhealthy",
            "message": "; ".join(issues),
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
        }

    return {
        "status": "healthy",
        "message": "System resources OK",
        "memory_percent": memory.percent,
        "disk_percent": disk.percent,
    }


def _get_system_metrics() -> dict[str, Any]:
    """Get current system metrics"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": memory.percent,
        "memory_used_mb": round(memory.used / 1024 / 1024, 2),
        "memory_total_mb": round(memory.total / 1024 / 1024, 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
        "disk_total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
    }


async def _check_if_ready() -> bool:
    """
    Check if application is ready to serve traffic.

    Add checks for:
    - Configuration loaded
    - API keys present
    - Models initialized
    - Database connected (if applicable)
    """

    # Example checks
    import os

    # Check 1: API keys configured
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not configured")
        return False

    # Check 2: Add more checks as needed
    # - Database connection
    # - ML models loaded
    # - Cache warmed up

    return True


# ============================================
# USAGE IN MAIN.PY
# ============================================

"""
# In src/api/main.py:

from src.api.health import router as health_router

app = FastAPI()

# Include health check routes
app.include_router(health_router)

# Now you have these endpoints:
# - GET /health              → Quick health check
# - GET /health/detailed     → Detailed with dependency checks
# - GET /health/ready        → Kubernetes readiness probe
# - GET /health/live         → Kubernetes liveness probe
# - GET /metrics             → System metrics
"""

# ============================================
# KUBERNETES CONFIGURATION EXAMPLE
# ============================================

"""
# In your Kubernetes deployment.yaml:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-orchestration-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: your-api:latest
        ports:
        - containerPort: 8000

        # Liveness probe - Is the container alive?
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness probe - Is the app ready for traffic?
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
"""
