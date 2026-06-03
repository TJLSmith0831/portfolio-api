"""
Redis client factory for synchronous access.

This module centralizes Redis configuration so all components
(API endpoints, job stores, workers) share the same connection
behavior.
"""

import os
import redis

REDIS_DB = 15

def get_redis_client() -> redis.Redis:
    """
    Create and return a Redis client instance.

    Prefers REDIS_URL (provided by Railway's Redis plugin) when set.
    Falls back to REDIS_HOST / REDIS_PORT for docker-compose local dev.

    Environment variables:
    - REDIS_URL  (e.g. redis://host:6379) — takes precedence
    - REDIS_HOST (default: localhost)
    - REDIS_PORT (default: 6379)
    - REDIS_DB   (default: REDIS_DB constant)
    """
    url = os.getenv("REDIS_URL")
    if url:
        return redis.from_url(url, db=int(os.getenv("REDIS_DB", REDIS_DB)), decode_responses=True)

    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", REDIS_DB)),
        decode_responses=True,
    )
