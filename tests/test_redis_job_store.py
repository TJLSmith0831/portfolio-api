"""
Tests for the Redis-backed job store used by the job-based Player Fit Summarizer.

These tests validate the core job lifecycle:
- job creation
- status updates
- successful completion
- failure handling

All tests run against a local Redis instance and use an isolated test database.
"""

import uuid
import pytest
import redis

from app.jobs.redis_store import RedisJobStore, JobStatus


@pytest.fixture(scope="function")
def redis_client():
    """
    Create a Redis client connected to a dedicated test database.

    The database is flushed before and after each test to ensure isolation
    and prevent state leakage between tests.

    :return: redis.Redis client instance
    """
    client = redis.Redis(
        host="localhost",
        port=6379,
        db=15,  # dedicated test DB
        decode_responses=True,
    )
    client.flushdb()
    yield client
    client.flushdb()


@pytest.fixture
def job_store(redis_client):
    """
    Initialize a RedisJobStore using the test Redis client.

    :param redis_client: Redis client fixture
    :return: RedisJobStore instance
    """
    return RedisJobStore(redis_client)


def test_create_job_sets_initial_state(job_store):
    """
    Verify that creating a job initializes all required fields correctly.

    Expectations:
    - status is QUEUED
    - payload is stored verbatim
    - result is None
    - error is None
    """
    payload = {"player_name": "Darian Mensah", "team": "Texas"}

    job_id = job_store.create_job(payload)
    job = job_store.get_job(job_id)

    assert job["status"] == JobStatus.QUEUED
    assert job["payload"] == payload
    assert job["result"] is None
    assert job["error"] is None


def test_update_job_status(job_store):
    """
    Verify that a job's status can be updated independently
    without mutating other fields.
    """
    job_id = job_store.create_job({"foo": "bar"})

    job_store.update_status(job_id, JobStatus.RUNNING)
    job = job_store.get_job(job_id)

    assert job["status"] == JobStatus.RUNNING


def test_store_job_result(job_store):
    """
    Verify that completing a job:
    - sets status to COMPLETED
    - stores the result payload
    - clears any existing error
    """
    job_id = job_store.create_job({"foo": "bar"})
    result = {"summary": "Player fits well"}

    job_store.complete_job(job_id, result)
    job = job_store.get_job(job_id)

    assert job["status"] == JobStatus.COMPLETED
    assert job["result"] == result
    assert job["error"] is None


def test_fail_job(job_store):
    """
    Verify that failing a job:
    - sets status to FAILED
    - records the error message
    """
    job_id = job_store.create_job({"foo": "bar"})
    error_message = "Scraping failed"

    job_store.fail_job(job_id, error_message)
    job = job_store.get_job(job_id)

    assert job["status"] == JobStatus.FAILED
    assert job["error"] == error_message
