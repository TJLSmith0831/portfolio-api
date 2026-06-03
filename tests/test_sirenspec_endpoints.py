"""
Endpoint tests for the SirenSpec demo routes.

`test_endpoints_return_job_id` runs with no network (the background task
is patched). `test_*_integration` tests are gated behind real API keys
and the `integration` pytest marker.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from app.api.sirenspec import (
    COMPRESSION_GAUNTLET_ENDPOINT,
    GRADING_FACTORY_ENDPOINT,
    POKEMON_BATTLE_ENDPOINT,
)
from app.jobs.redis_store import JobStatus, RedisJobStore
from app.main import app
from app.utils.redis_client import get_redis_client


HAS_LIVE_KEYS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.fixture
def client():
    return TestClient(app)


def _poll_job(job_id: str, timeout_s: float = 120.0) -> dict:
    store = RedisJobStore(get_redis_client())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = store.get_job(job_id)
        if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")


def test_compression_gauntlet_enqueues_job(client):
    resp = client.post(
        COMPRESSION_GAUNTLET_ENDPOINT,
        json={"text": "Some passage here."},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"]


def test_grading_factory_enqueues_job(client):
    resp = client.post(
        GRADING_FACTORY_ENDPOINT,
        json={"papers": ["paper one", "paper two"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"]


def test_pokemon_battle_enqueues_job(client):
    body = {
        "pokemon1": {
            "name": "pikachu",
            "displayName": "Pikachu",
            "hp": 35,
            "attack": 55,
            "defense": 40,
            "speed": 90,
            "types": ["electric"],
        },
        "pokemon2": {
            "name": "charizard",
            "displayName": "Charizard",
            "hp": 78,
            "attack": 84,
            "defense": 78,
            "speed": 100,
            "types": ["fire", "flying"],
        },
        "max_turns": 6,
    }
    resp = client.post(POKEMON_BATTLE_ENDPOINT, json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"]


# =========================
# Live integration tests
# =========================


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires OPENAI_API_KEY and ANTHROPIC_API_KEY")
def test_compression_gauntlet_live(client):
    passage = (
        "The invention of the printing press by Johannes Gutenberg around 1440 is "
        "widely considered one of the most transformative events in human history. "
        "Before its introduction, books were painstakingly copied by hand, a process "
        "that made them extraordinarily expensive and confined literacy primarily to "
        "the clergy and nobility. Gutenberg's movable-type system allowed pages to "
        "be composed quickly and reproduced in large quantities at a fraction of "
        "the previous cost."
    )
    resp = client.post(COMPRESSION_GAUNTLET_ENDPOINT, json={"text": passage})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(job_id, timeout_s=180.0)
    assert job["status"] == JobStatus.COMPLETED, job.get("error")

    result = job["result"]
    assert len(result["rounds"]) == 4
    assert all(r["output"] for r in result["rounds"]), result["rounds"]
    # Each round should compress (allow slack — model isn't perfect)
    assert (
        result["rounds"][3]["outputWords"] <= result["rounds"][0]["inputWords"]
    )


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires OPENAI_API_KEY and ANTHROPIC_API_KEY")
def test_grading_factory_live(client):
    papers = [
        "The mitochondria is the powerhouse of the cell. It produces ATP through "
        "oxidative phosphorylation, converting nutrients into usable energy.",
        "Climate change is a big problem. Lots of scientists agree. We should do "
        "something about it before it gets worse.",
    ]
    resp = client.post(GRADING_FACTORY_ENDPOINT, json={"papers": papers})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(job_id, timeout_s=240.0)
    assert job["status"] == JobStatus.COMPLETED, job.get("error")

    result = job["result"]
    assert len(result["papers"]) == 2
    for p in result["papers"]:
        assert p["grade"] in {"A", "B", "C", "D", "F", "?"}
        assert p["integrity"] in {"LIKELY_HUMAN", "UNCERTAIN", "LIKELY_AI"}
        ids = [a["id"] for a in p["agents"]]
        assert ids == ["editor", "ai_detector", "grader"]


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires OPENAI_API_KEY and ANTHROPIC_API_KEY")
def test_pokemon_battle_live(client):
    body = {
        "pokemon1": {
            "name": "pikachu",
            "displayName": "Pikachu",
            "hp": 35,
            "attack": 55,
            "defense": 40,
            "speed": 90,
            "types": ["electric"],
        },
        "pokemon2": {
            "name": "caterpie",
            "displayName": "Caterpie",
            "hp": 30,
            "attack": 30,
            "defense": 35,
            "speed": 45,
            "types": ["bug"],
        },
        "max_turns": 8,
    }
    resp = client.post(POKEMON_BATTLE_ENDPOINT, json=body)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(job_id, timeout_s=240.0)
    assert job["status"] == JobStatus.COMPLETED, job.get("error")

    result = job["result"]
    assert result["winner"] in (1, 2)
    assert 1 <= len(result["turns"]) <= 8
    for turn in result["turns"]:
        assert turn["attacker"] in (1, 2)
        assert turn["damage"] >= 1
        assert turn["move"]
