"""
Shared API state and helpers.

Provides a central place for runtime stats and utility functions
that are imported by both the FastAPI app and route modules.
"""

import time

# Track app start time for uptime calculation
app_start_time = time.time()

# Statistics tracking (in production, use Redis or database)
stats: dict = {
    "total_requests": 0,
    "requests_by_provider": {},
    "total_cost_usd": 0.0,
    "total_latency_ms": 0.0,
    "errors": 0,
}


def get_uptime_seconds() -> int:
    """Get application uptime in seconds."""
    return int(time.time() - app_start_time)


def update_provider_stats(provider: str, cost: float):
    """Update statistics for a provider."""
    if provider not in stats["requests_by_provider"]:
        stats["requests_by_provider"][provider] = 0

    stats["requests_by_provider"][provider] += 1
    stats["total_cost_usd"] += cost
