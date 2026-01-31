from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.jobs.redis_store import RedisJobStore
from app.models.redis_models import JobStatusResponse
from app.utils.redis_client import get_redis_client


from app.api.chat import router as chat_router
from app.api.summarize_player_fit import router as player_fit_router
from app.utils.scrapers.driver_singleton import get_driver

VERSION = '0.1.0'

app = FastAPI(
    title="Portfolio API",
    description="FastAPI app exposing chat and player‑fit summarization endpoints",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://tjlsmith0831.dev",
        "https://www.tjlsmith0831.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(player_fit_router)

@app.get("/")
def read_root():
    return {"message": "Hello from Tristan Smith! Welcome to my portfolio API hosted on DigitalOcean!"}

@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.get("/version")
def read_version():
    return {"version": VERSION}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("app/static/favicon.ico")

# =========================
# Job status endpoint
# =========================

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Retrieve the current status and result of a background job.

    This endpoint is intended to be polled by the UI until the job
    reaches a terminal state (COMPLETED or FAILED).

    :param job_id: Redis job identifier
    :return: JobStatusResponse containing status, result, or error
    """
    job_store = RedisJobStore(get_redis_client())

    try:
        job = job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found",
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )
