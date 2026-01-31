"""
Background worker for processing Player Fit Summarization jobs.

This module contains the logic that was previously executed inline
in the API request and is now run asynchronously.
"""

from app.jobs.redis_store import RedisJobStore, JobStatus
from app.utils.redis_client import get_redis_client
from app.utils.scrapers.driver_singleton import get_driver, driver_lock
from app.utils.scrapers.sports247_scraper import Sports247Scraper, PlaywrightDriver
from app.llm_client import get_llm_client
from app.services.player_fit_summarizer import PlayerFitSummarizer


def process_player_fit_job(job_id: str) -> None:
    """
    Execute a Player Fit Summarization job.

    This function:
    - Updates job status
    - Scrapes the player profile
    - Calls the LLM
    - Stores the final result or error

    :param job_id: Redis job identifier
    """
    redis_client = get_redis_client()
    job_store = RedisJobStore(redis_client)

    job_store.update_status(job_id, JobStatus.RUNNING)

    try:
        job = job_store.get_job(job_id)
        payload = job["payload"]

        browser = get_driver()

        with driver_lock:
            context = browser.new_context()
            page = context.new_page()
            driver = PlaywrightDriver(page)

            try:
                scraper = Sports247Scraper(driver)
                summarizer = PlayerFitSummarizer(client=get_llm_client())

                search_result = scraper.search_player_profile(
                    payload["player_name"]
                )

                if not search_result or not search_result.found:
                    raise RuntimeError(
                        f"No player profile found for '{payload['player_name']}'"
                    )

                relevant_info = summarizer.select_relevant_information(
                    driver,
                    str(search_result.profile_url),
                )

                summary = summarizer.summarizer_player_fit(
                    player_name=payload["player_name"],
                    team_name=payload["requested_team_name"],
                    player_profile=relevant_info,
                )

                job_store.complete_job(job_id, summary.model_dump())

            finally:
                context.close()

    except Exception as exc:
        job_store.fail_job(job_id, str(exc))
