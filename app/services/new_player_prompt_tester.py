import time
import json
from typing import List

from app.llm_client import get_llm_client
from app.services.player_fit_summarizer import PlayerFitSummarizer
from app.utils.scrapers.sports247_scraper import Sports247Scraper
from app.utils.scrapers.playwright_helpers import PlaywrightDriver
from app.utils.scrapers.driver_singleton import get_driver, driver_lock

# ----------------------------
# Test Configuration
# ----------------------------

TEST_CASES = [
    {
        # Control: whatever the current prompt produces
        "name": "baseline",
        "player_name": "Darian Mensah",
        "team_name": "LSU",
        "prompt_overrides": {},
    },
    {
        "name": "production_recruiter_voice",
        "player_name": "Darian Mensah",
        "team_name": "LSU",
        "prompt_overrides": {
            "program_framing": (
                "Evaluate this player strictly through the lens of how they would fit within "
                "the requested program’s on-field system, roster construction, competitive environment, "
                "and development philosophy. Write as if advising the program’s recruiting staff."
            ),
            "grounding_guardrail": (
                "Base all conclusions on the provided excerpt and structured profile only. "
                "Do not reference rankings tables, site navigation text, recruiting service metadata, "
                "or unrelated programs."
            ),
            "scoring_calibration": (
                "Use the full fit_score range conservatively. "
                "Scores above 85 should be rare and reserved for near-ideal fits with minimal projection risk. "
                "A solid, realistic fit typically falls between 72–82."
            ),
            "analysis_tone": (
                "Write with the voice of a college football recruiting analyst: balanced, specific, "
                "and evaluative rather than promotional. Emphasize projection, development curve, "
                "and roster context over hype."
            ),
            "risk_discipline": (
                "Always identify at least one concrete risk or uncertainty (e.g., development timeline, "
                "competition level, physical traits, roster congestion). Avoid generic or placeholder risks."
            ),
        },
    }
]

# ----------------------------
# Test Runner
# ----------------------------

def run_test_case(case: dict) -> dict:
    llm_client = get_llm_client()
    summarizer = PlayerFitSummarizer(client=llm_client)

    browser = get_driver()

    with driver_lock:
        context = browser.new_context()
        page = context.new_page()
        driver = PlaywrightDriver(page)

        try:
            scraper = Sports247Scraper(driver)

            search_result = scraper.search_player_profile(case["player_name"])
            if not search_result or not search_result.found:
                raise RuntimeError("Player profile not found")

            start = time.perf_counter()
            relevant_info = summarizer.select_relevant_information(
                driver=driver,
                profile_url=str(search_result.profile_url),
                model=llm_client.model_enum,
            )

            summary = summarizer.summarize_player_fit(
                player_name=case["player_name"],
                team_name=case["team_name"],
                player_profile=relevant_info,
                prompt_overrides=case["prompt_overrides"],
            )

        finally:
            context.close()

    elapsed = round(time.perf_counter() - start, 2)

    return {
        "test_case": case["name"],
        "latency_seconds": elapsed,
        "fit_score": summary.fit_score,
        "scheme_fit_excerpt": summary.scheme_fit[:250],
        "depth_chart_impact_excerpt": summary.depth_chart_impact[:250],
        "development_outlook_excerpt": summary.development_outlook[:250],
        "risk_factors": summary.risk_factors,
    }


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("\n=== Player Fit Prompt Test Suite (Production Flow) ===\n")

    results: List[dict] = []

    for case in TEST_CASES:
        print(f"Running test case: {case['name']}...")
        result = run_test_case(case)
        results.append(result)

        print(
            f"  → latency: {result['latency_seconds']}s | "
            f"fit_score: {result['fit_score']}"
        )

    print("\n=== Results ===\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
