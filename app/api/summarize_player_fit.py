"""
The Ollama connection to the player-fit summarizer demo.
This module now uses a Playwright-based browser implementation
while preserving all original behavior and prompts.
"""
import sys
import logging

from fastapi import APIRouter

from app.jobs.redis_store import RedisJobStore
from app.models.player_fit_models import (
    PlayerFitSummaryRequest,
)
from fastapi import BackgroundTasks
from app.jobs.player_fit_worker import process_player_fit_job

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
def summarize_player_fit(request: PlayerFitSummaryRequest,
                         background_tasks: BackgroundTasks) -> JobEnqueueResponse:
    """
    Enqueue a player fit summarization job and schedule background execution.

    :param request: Player fit summarization request payload
    :param background_tasks: FastAPI background task manager
    :return: JobEnqueueResponse containing the job ID
    """
    job_store = RedisJobStore(get_redis_client())

    payload = {
        "player_name": request.player_name,
        "requested_team_name": request.requested_team_name,
    }

    job_id = job_store.create_job(payload)

    background_tasks.add_task(process_player_fit_job, job_id)

    return JobEnqueueResponse(job_id=job_id)

