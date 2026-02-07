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

FINAL_PRODUCTION_OVERRIDES = {
    "program_guardrail": (
        "Evaluate the player strictly through the lens of how they would fit within "
        "the requested program’s system, roster composition, and development environment. "
        "Avoid generic program praise or national narratives."
    ),
    "recruiter_voice": (
        "Write in the tone of an internal college recruiting evaluation. "
        "Be measured, realistic, and specific. Avoid promotional or marketing language."
    ),
    "score_calibration": (
        "Use the full fit_score range conservatively. "
        "Only assign scores above 85 for exceptional, low-risk fits. "
        "Developmental or depth candidates should typically fall between 60–75."
    ),
    "anti_extraction": (
        "Do not quote rankings, menu text, page headers, or scraped artifacts. "
        "Base analysis only on inferred traits and context."
    ),
}

def process_player_fit_job(job_id: str) -> None:
    """
    Process a player fit summarization job using the given job ID.

    This function handles the entire lifecycle of processing a player fit summarization:
    - Updates the job status in Redis (queued, running, completed, failed)
    - Uses a web scraper to obtain player profile data
    - Uses the LLM client to generate summarization output
    - Saves the results or error back to Redis

    :param job_id: The unique identifier of the job in Redis
    :return: None
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
            llm_client = get_llm_client()
            model = llm_client.model_enum

            try:
                scraper = Sports247Scraper(driver)
                summarizer = PlayerFitSummarizer(client=llm_client)

                search_result = scraper.search_player_profile(
                    payload["player_name"]
                )

                if not search_result or not search_result.found:
                    raise RuntimeError(
                        f"No player profile found for '{payload['player_name']}'"
                    )

                relevant_info = summarizer.select_relevant_information(
                    driver=driver,
                    profile_url=str(search_result.profile_url),
                    model=model
                )

                summary = summarizer.summarize_player_fit(
                    player_name=payload["player_name"],
                    team_name=payload["requested_team_name"],
                    player_profile=relevant_info,
                    model=model,
                    prompt_overrides=FINAL_PRODUCTION_OVERRIDES
                )

                job_store.complete_job(job_id, summary.model_dump())
                job_store.set_cached_player_fit(
                    payload["player_name"],
                    payload["requested_team_name"],
                    summary.model_dump(),
                )

            finally:
                context.close()

    except Exception as exc:
        job_store.fail_job(job_id, str(exc))
