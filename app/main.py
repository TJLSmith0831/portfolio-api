from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.player_fit_summarizer import router as player_fit_router
from app.utils.scrapers.driver_singleton import get_driver

VERSION = '0.1.0'

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: eagerly initialize the browser
    get_driver()
    yield
    # Shutdown: let OS reclaim resources (optional explicit cleanup)

app = FastAPI(
    title="Portfolio API",
    description="FastAPI app exposing chat and player‑fit summarization endpoints",
    version=VERSION)

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
