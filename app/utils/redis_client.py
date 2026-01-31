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

    This client is intended for synchronous usage and is safe
    to share across request handlers.

    Environment variables:
    - REDIS_HOST (default: localhost)
    - REDIS_PORT (default: 6379)
    - REDIS_DB   (default: 0)

    :return: Configured redis.Redis client
    """
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", REDIS_DB)),
        decode_responses=True,
    )
