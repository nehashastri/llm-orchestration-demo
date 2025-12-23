"""
Redis connection management and rate limiting utilities.

This module provides:
- Async Redis connection pool
- Lifecycle management (startup/shutdown)
- Rate limiting using Redis sorted sets (sliding window algorithm)
"""

import time

import redis.asyncio as redis

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global Redis connection pool
_redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """
    Get the Redis client instance.

    Returns:
        Redis client

    Raises:
        RuntimeError: If Redis not initialized
    """
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client


async def init_redis() -> None:
    """
    Initialize Redis connection pool.

    Called during application startup.
    """
    global _redis_client

    try:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,  # Return strings instead of bytes
            socket_connect_timeout=5,
            socket_keepalive=True,
            max_connections=50,
        )

        # Test connection
        _redis_client.ping()

        logger.info(
            "redis_connected",
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        raise


async def close_redis() -> None:
    """
    Close Redis connection pool.

    Called during application shutdown.
    """
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("redis_disconnected")


async def check_rate_limit(client_id: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Check if client has exceeded rate limit using sliding window algorithm.

    This uses Redis sorted sets where:
    - Key: "rate_limit:{client_id}"
    - Score: timestamp
    - Member: unique request ID (timestamp + random)

    Algorithm:
    1. Remove old entries outside the time window
    2. Count remaining entries
    3. If under limit, add new entry and allow
    4. If over limit, deny

    Args:
        client_id: Unique identifier for client (e.g., IP address)
        limit: Maximum requests allowed in window
        window: Time window in seconds

    Returns:
        Tuple of (is_allowed, remaining_requests)
        - is_allowed: True if request should be allowed
        - remaining_requests: Number of requests remaining in window
    """
    redis_client = await get_redis()
    current_time = time.time()
    window_start = current_time - window
    key = f"rate_limit:{client_id}"

    # Remove entries older than the window
    await redis_client.zremrangebyscore(key, 0, window_start)

    # Count current requests in window
    request_count = await redis_client.zcard(key)

    if request_count < limit:
        # Add this request to the set
        await redis_client.zadd(key, {f"{current_time}": current_time})
        # Set expiration on the key (cleanup)
        await redis_client.expire(key, window)

        remaining = limit - request_count - 1
        return True, remaining
    else:
        remaining = 0
        return False, remaining


async def reset_rate_limit(client_id: str) -> None:
    """
    Reset rate limit for a specific client.

    Useful for testing or manual intervention.

    Args:
        client_id: Client identifier to reset
    """
    redis_client = await get_redis()
    key = f"rate_limit:{client_id}"
    await redis_client.delete(key)
    logger.info("rate_limit_reset", client_id=client_id)


async def get_rate_limit_info(client_id: str, window: int) -> dict:
    """
    Get rate limit information for a client.

    Args:
        client_id: Client identifier
        window: Time window in seconds

    Returns:
        Dictionary with rate limit stats
    """
    redis_client = await get_redis()
    current_time = time.time()
    window_start = current_time - window
    key = f"rate_limit:{client_id}"

    # Clean old entries
    await redis_client.zremrangebyscore(key, 0, window_start)

    # Get current count
    request_count = await redis_client.zcard(key)

    # Get oldest entry timestamp
    oldest = await redis_client.zrange(key, 0, 0, withscores=True)
    oldest_timestamp = oldest[0][1] if oldest else None

    return {
        "client_id": client_id,
        "request_count": request_count,
        "window_seconds": window,
        "oldest_request_timestamp": oldest_timestamp,
        "current_time": current_time,
    }
