"""
The Ollama connection to the player-fit summarizer demo.
This module now uses a Playwright-based browser implementation
while preserving all original behavior and prompts.

Jobs are enqueued to Redis and processed by a dedicated worker process,
so Ollama does not run in the API process.
"""

import logging

from fastapi import APIRouter

from app.jobs.redis_store import RedisJobStore
from app.models.player_fit_models import PlayerFitSummaryRequest
from app.models.redis_models import JobEnqueueResponse
from app.utils.redis_client import get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

SUMMARIZER_ENDPOINT = "/summarize_player_fit"

router = APIRouter(tags=["Summarize Player Fit"])


# =========================
# API entry point
# =========================


@router.post("/summarize_player_fit", response_model=JobEnqueueResponse)
def summarize_player_fit(request: PlayerFitSummaryRequest) -> JobEnqueueResponse:
    """
    API endpoint to enqueue a player fit summarization job.

    If a completed result for (player_name, requested_team_name) exists in cache,
    a job is created and immediately marked completed so polling returns the same JSON.
    Otherwise the job is enqueued for the dedicated worker.

    :param request: PlayerFitSummaryRequest payload containing player
    information.
    :return: JobEnqueueResponse with the enqueued job_id.
    """
    job_store = RedisJobStore(get_redis_client())

    cached = job_store.get_cached_player_fit(
        request.player_name,
        request.requested_team_name,
    )
    payload = {
        "player_name": request.player_name,
        "requested_team_name": request.requested_team_name,
    }
    job_id = job_store.create_job(payload)

    if cached is not None:
        job_store.complete_job(job_id, cached)
    else:
        job_store.enqueue_for_processing(job_id)

    return JobEnqueueResponse(job_id=job_id)
