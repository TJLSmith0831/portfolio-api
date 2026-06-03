"""
SirenSpec-backed demo endpoints.

Each endpoint enqueues a job, kicks off a FastAPI BackgroundTask that runs
the workflow with the SirenSpec async SDK, persists the shaped result via
RedisJobStore, and returns a job_id the frontend polls at /jobs/{id}.

The job storage matches the pattern used by /summarize_player_fit so the
existing /jobs/{id} endpoint works unchanged.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import APIRouter, BackgroundTasks

from app.jobs.redis_store import RedisJobStore
from app.models.redis_models import JobEnqueueResponse
from app.models.sirenspec_models import (
    CompressionGauntletRequest,
    GradingFactoryRequest,
    PokemonBattleRequest,
)
from app.services.pokemon_battle import run_pokemon_battle
from app.services.sirenspec_runner import run_workflow
from app.services.sirenspec_shapers import (
    shape_compression_gauntlet,
    shape_grading_factory,
)
from app.utils.redis_client import get_redis_client

log = logging.getLogger(__name__)


COMPRESSION_GAUNTLET_ENDPOINT = "/sirenspec/compression_gauntlet"
GRADING_FACTORY_ENDPOINT = "/sirenspec/grading_factory"
POKEMON_BATTLE_ENDPOINT = "/sirenspec/pokemon_battle"


router = APIRouter(tags=["SirenSpec"])


async def _run_and_store(job_id: str, runner: Callable[[], Awaitable[dict]]) -> None:
    """Execute *runner*, persist its result (or error) under *job_id*."""
    job_store = RedisJobStore(get_redis_client())
    try:
        result = await runner()
        job_store.complete_job(job_id, result)
    except Exception as exc:  # noqa: BLE001
        log.exception("sirenspec job %s failed", job_id)
        job_store.fail_job(job_id, str(exc))


# =========================
# Compression Gauntlet
# =========================


def _require_success(trace: dict) -> None:
    """Raise RuntimeError if the SirenSpec workflow did not complete successfully."""
    status = trace.get("summary", {}).get("status", "failed")
    if status != "success":
        error = next(
            (n.get("error") for n in trace.get("nodes", []) if n.get("error")),
            "Workflow failed without a specific error message",
        )
        raise RuntimeError(error)


async def _execute_compression_gauntlet(text: str) -> dict:
    trace = await run_workflow("compression_gauntlet", text)
    _require_success(trace)
    return shape_compression_gauntlet(trace, original_input=text).model_dump()


@router.post(COMPRESSION_GAUNTLET_ENDPOINT, response_model=JobEnqueueResponse)
def compression_gauntlet(
    request: CompressionGauntletRequest, background_tasks: BackgroundTasks
) -> JobEnqueueResponse:
    """
    Run a 4-round compression gauntlet over the provided passage.
    Returns a job_id the client should poll at /jobs/{id}.
    """
    job_store = RedisJobStore(get_redis_client())
    job_id = job_store.create_job({"workflow": "compression_gauntlet"})
    background_tasks.add_task(
        _run_and_store, job_id, lambda: _execute_compression_gauntlet(request.text)
    )
    return JobEnqueueResponse(job_id=job_id)


# =========================
# Grading Factory
# =========================


async def _execute_grading_factory(papers: list[str]) -> dict:
    trace = await run_workflow("grading_factory", papers)
    _require_success(trace)
    return shape_grading_factory(trace, original_papers=papers).model_dump()


@router.post(GRADING_FACTORY_ENDPOINT, response_model=JobEnqueueResponse)
def grading_factory(
    request: GradingFactoryRequest, background_tasks: BackgroundTasks
) -> JobEnqueueResponse:
    """
    Run the grading factory over a list of student papers.
    Returns a job_id the client should poll at /jobs/{id}.
    """
    job_store = RedisJobStore(get_redis_client())
    job_id = job_store.create_job({"workflow": "grading_factory"})
    background_tasks.add_task(
        _run_and_store,
        job_id,
        lambda: _execute_grading_factory(request.papers),
    )
    return JobEnqueueResponse(job_id=job_id)


# =========================
# Pokemon Battle
# =========================


async def _execute_pokemon_battle(req: PokemonBattleRequest) -> dict:
    result = await run_pokemon_battle(req)
    return result.model_dump()


@router.post(POKEMON_BATTLE_ENDPOINT, response_model=JobEnqueueResponse)
def pokemon_battle(
    request: PokemonBattleRequest, background_tasks: BackgroundTasks
) -> JobEnqueueResponse:
    """
    Run a multi-turn Pokémon battle driven by a Python loop that invokes
    a single-turn SirenSpec workflow once per round.
    """
    job_store = RedisJobStore(get_redis_client())
    job_id = job_store.create_job({"workflow": "pokemon_battle"})
    background_tasks.add_task(
        _run_and_store, job_id, lambda: _execute_pokemon_battle(request)
    )
    return JobEnqueueResponse(job_id=job_id)
