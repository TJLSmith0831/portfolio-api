"""
Dedicated worker process for player-fit summarization jobs.

Polls Redis (BLPOP) for job IDs and runs process_player_fit_job in this process,
so Ollama runs in the worker only and does not contend with the API process.
Concurrency is 1 (one job at a time).
"""

import logging
import sys

from app.jobs.player_fit_worker import process_player_fit_job
from app.jobs.redis_store import PLAYER_FIT_QUEUE_KEY
from app.utils.redis_client import get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

BLPOP_TIMEOUT = 5


def main() -> None:
    redis_client = get_redis_client()
    log.info("Player-fit worker started, listening on %s", PLAYER_FIT_QUEUE_KEY)

    while True:
        try:
            result = redis_client.blpop(
                [PLAYER_FIT_QUEUE_KEY], timeout=BLPOP_TIMEOUT
            )
            if result is None:
                continue
            _, job_id_bytes = result
            if isinstance(job_id_bytes, bytes):
                job_id = job_id_bytes.decode()
            else:
                job_id = job_id_bytes
            log.info("Processing job_id=%s", job_id)
            process_player_fit_job(job_id)
            log.info("Completed job_id=%s", job_id)
        except KeyboardInterrupt:
            log.info("Worker shutting down")
            sys.exit(0)
        except Exception as exc:
            log.exception("Worker error: %s", exc)


if __name__ == "__main__":
    main()
