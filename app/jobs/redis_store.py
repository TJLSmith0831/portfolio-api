"""
Redis-backed job store for managing asynchronous background jobs.

This module defines:
- A JobStatus enum representing the lifecycle of a job
- A RedisJobStore class responsible for creating, updating, and retrieving jobs

Jobs are stored in Redis as JSON blobs under deterministic keys and are intended
to be consumed by background workers or task processors.
"""

import hashlib
import json
import uuid
from enum import Enum


class JobStatus(str, Enum):
    """
    Enumeration of valid job lifecycle states.

    Jobs transition through these states in a strict order:
    - QUEUED: Job has been created and is awaiting processing
    - RUNNING: Job is currently being processed by a worker
    - COMPLETED: Job finished successfully and produced a result
    - FAILED: Job encountered an unrecoverable error
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Redis list key for the dedicated worker to poll (BLPOP)
PLAYER_FIT_QUEUE_KEY = "queue:player_fit"

# Cache key prefix for completed player-fit results (player_name + team_name)
PLAYER_FIT_CACHE_PREFIX = "player_fit_cache:"
PLAYER_FIT_CACHE_TTL_SECONDS = 86400  # 24 hours


class RedisJobStore:
    """
    Redis-backed persistence layer for background jobs.

    This class is responsible for:
    - Creating new job records
    - Updating job status and results
    - Fetching job state for polling APIs

    Jobs are stored as JSON-serialized dictionaries in Redis using
    the key pattern: `job:<job_id>`.
    """

    def __init__(self, redis_client):
        """
        Initialize the job store.

        :param redis_client: An instance of redis.Redis configured for
                             synchronous access
        """
        self.redis = redis_client

    def _key(self, job_id: str) -> str:
        """
        Construct the Redis key for a given job ID.

        This method centralizes key naming to ensure consistency
        across all job operations.

        :param job_id: Unique job identifier
        :return: Redis key string for the job
        """
        return f"job:{job_id}"

    def create_job(self, payload: dict) -> str:
        """
        Create a new job with an initial QUEUED status.

        The job is persisted immediately to Redis and is ready
        to be picked up by a background worker.

        :param payload: Arbitrary job input data required for processing
        :return: Generated job ID
        """
        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "payload": payload,
            "result": None,
            "error": None,
        }

        self.redis.set(self._key(job_id), json.dumps(job))
        return job_id

    def enqueue_for_processing(self, job_id: str, queue_key: str = PLAYER_FIT_QUEUE_KEY) -> None:
        """
        Push a job ID onto the given queue for a dedicated worker to process.

        :param job_id: Job ID to enqueue.
        :param queue_key: Redis list key (default: PLAYER_FIT_QUEUE_KEY).
        """
        self.redis.lpush(queue_key, job_id)

    def _player_fit_cache_key(self, player_name: str, team_name: str) -> str:
        raw = f"{player_name.strip()}|{team_name.strip()}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return f"{PLAYER_FIT_CACHE_PREFIX}{h}"

    def get_cached_player_fit(
        self,
        player_name: str,
        requested_team_name: str,
    ) -> dict | None:
        """
        Return a cached completed result for (player_name, requested_team_name), or None.
        """
        key = self._player_fit_cache_key(player_name, requested_team_name)
        raw = self.redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set_cached_player_fit(
        self,
        player_name: str,
        requested_team_name: str,
        result: dict,
        ttl_seconds: int = PLAYER_FIT_CACHE_TTL_SECONDS,
    ) -> None:
        """
        Store a completed player-fit result for (player_name, requested_team_name).
        """
        key = self._player_fit_cache_key(player_name, requested_team_name)
        self.redis.setex(key, ttl_seconds, json.dumps(result))

    def get_job(self, job_id: str) -> dict:
        """
        Retrieve a job record from Redis.

        :param job_id: Unique job identifier
        :return: Deserialized job dictionary
        :raises KeyError: If the job does not exist
        """
        raw = self.redis.get(self._key(job_id))
        if raw is None:
            raise KeyError(f"Job {job_id} not found")

        return json.loads(raw)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update the status of an existing job.

        This method does not modify result or error fields.

        :param job_id: Unique job identifier
        :param status: New job status
        """
        job = self.get_job(job_id)
        job["status"] = status
        self.redis.set(self._key(job_id), json.dumps(job))

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Mark a job as successfully completed.

        Sets the status to COMPLETED, stores the result payload,
        and clears any existing error.

        :param job_id: Unique job identifier
        :param result: Final job output
        """
        job = self.get_job(job_id)
        job["status"] = JobStatus.COMPLETED
        job["result"] = result
        job["error"] = None

        self.redis.set(self._key(job_id), json.dumps(job))

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Mark a job as failed.

        Sets the status to FAILED and records an error message.
        The result field is left unchanged.

        :param job_id: Unique job identifier
        :param error: Human-readable error message
        """
        job = self.get_job(job_id)
        job["status"] = JobStatus.FAILED
        job["error"] = error

        self.redis.set(self._key(job_id), json.dumps(job))
