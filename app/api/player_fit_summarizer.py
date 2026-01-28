"""
The Ollama connection to the player-fit summarizer demo.
This module now uses an object‑oriented design while preserving
all original behaviour and prompts.
"""
import json
import sys
import re
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from app.models.player_models import PlayerFitSummary, PlayerFitRequest, RelevantInfoResponse
from app.scrapers.generic_helpers import fetch_website_contents
from app.scrapers.sports247_scraper import Sports247Scraper
from app.ollama_client import OllamaModels, get_ollama_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

# =========================
# Helper functions
# =========================

def _extract_json(raw_text: str) -> dict:
    """Extract JSON object from a string response."""
    clean_string = re.sub(r"[ \t]+$", "", raw_text)  # trim trailing whitespace
    try:
        return json.loads(clean_string)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse JSON. Raw text received was {raw_text}. Error details:\n{exc}"
        )

def _generate_with_retries(
    client,
    model: str,
    system: str,
    prompt: str,
    max_retries: int = 2,
) -> dict:
    last_error = None

    for attempt in range(1, max_retries + 2):
        log.info("Ollama attempt %d/%d", attempt, max_retries + 1)

        kwargs = {"model": model, "prompt": prompt}
        if system is not None:
            kwargs["system"] = system

        try:
            response = client.generate(**kwargs)
        except TypeError:
            full_prompt = prompt if not system else f"{system}\n\n{prompt}"
            response = client.generate(model=model, prompt=full_prompt)

        raw_text = getattr(response, "response", None)
        if raw_text is not None:
            raw_text = raw_text.strip()
        else:
            raw_text = response.choices[0].text.strip()
        try:
            return _extract_json(raw_text)
        except Exception as exc:
            last_error = exc
            log.warning("JSON parse failed on attempt %d. Retrying.", attempt)
            # Repair-style retry prompt
            prompt = (
                "The previous response was not valid JSON.\n\n"
                "Return ONLY a complete, valid JSON object matching the schema.\n"
                "Do not include explanations, comments, or markdown.\n\n"
                f"Original content:\n{raw_text}"
            )

    raise RuntimeError(
        f"Failed after {max_retries + 1} attempts.\nLast error: {last_error}"
    )

# =========================
# System Prompts
# =========================

REL_INFO_SYSTEM_PROMPT = """
    You are given the raw textual contents of a college football player profile webpage.

    Your task is to identify and extract the MOST IMPORTANT football-relevant information
    needed to accurately understand and evaluate the player.

    Focus on factual, high-signal information that should be normalized into a structured player record.
    Ignore navigation elements, marketing copy, duplicated sections, and site boilerplate.

    Prioritize identifying the following categories when present:

    - Player identity (name, position, height, weight)
    - Current and former schools
    - Transfer portal or transfer prediction context (e.g., Crystal Ball, destinations, confidence)
    - Recruiting or transfer rankings (overall and positional)
    - Most recent season statistics (passing, rushing, receiving as applicable)
    - Experience year / class
    - High school and hometown
    - Notable recent headlines or narrative signals (titles only, not article bodies)

    Explicitly IGNORE:
    - Login prompts, subscription offers, ads
    - Site navigation, footers, copyright notices
    - Repeated or duplicated tables or labels
    - UI labels such as “Timeline”, “Embed”, “Join”, “Watch”, etc.

    Respond ONLY with valid JSON using the following structure:

    {
        "identity": {
            "name": "...",
            "position": "...",
            "height": "...",
            "weight": "..."
        },
        "school_context": {
            "current_school": "...",
            "transfer_interest": {
            "destination": "...",
            "confidence": "..."
            }
        },
        "rankings": {
            "transfer_overall": "...",
            "transfer_position": "...'
        },
        "latest_season_stats": {
            "year": "...",
            "summary": "concise statistical summary"
        },
        "background": {
            "high_school": "...",
            "hometown": "...",
            "experience_year": "..."
        },
        "notable_headlines": [
            {"title": "...", "date": "..."}
        ]
    }

    Only include fields when the information is clearly present in the text.
    Do NOT infer or guess missing values.
    Return ONLY valid JSON.
    Do not include commentary, markdown, or trailing text.
    Ensure the JSON object is complete and properly closed.
    Use "N/A" for missing values.
"""

PLAYER_FIT_SYSTEM_PROMPT = """
    You are a college football recruiting analyst. You will evaluate the most recent season
    stats and utilize that heavily in your evaluation and include them explicitly in the overall summary.

    IMPORTANT:
        - The teams are all colleges. No professional teams.
        - Position should be limited to: QB, RB, WR, TE, OL, DL, LB, CB, EDGE, S, K, P, ATH
            - EDGE is a defensive position. Previously, referred to as defensive ends.
        - Factor in scheme_fit and development outlook into fit_score
        - Do NOT mention legal or off-field issues in risk factors.
        - Reminder: the year is 2026, transfers don't have to redshirt

    Your job is to evaluate how well a college / high school player fits a specific team.
    Be concise, analytical, and realistic.
    Avoid hype. Use football terminology.
    Return structured JSON only.
"""

# =========================
# PlayerFitSummarizer
# =========================

class PlayerFitSummarizer:
    """Object‑oriented wrapper around the original procedural logic."""
    def __init__(self, client=None):
        """
        :param client: Optional Ollama client; if None, a default client is fetched.
        """
        self.client = client or get_ollama_client()
        self.logger = log

    # -------------- private helpers -----------------
    def _extract_json(self, raw_text: str) -> dict:
        return _extract_json(raw_text)

    def _generate_with_retries(
        self,
        model: str,
        system: str,
        prompt: str,
        max_retries: int = 2,
    ) -> dict:
        return _generate_with_retries(
            client=self.client,
            model=model,
            system=system,
            prompt=prompt,
            max_retries=max_retries,
        )

    def _build_player_fit_prompt(self, request: PlayerFitRequest) -> str:
        """
        Build the user prompt for player fit analysis.
        """
        return f"""
            **Output valid JSON only**.

            Player: {request.player_name}
            Team: {request.team_name}

            Player profile information:
            {request.player_profile.model_dump_json(indent=2)}

            Analyze the player fit and respond in the following JSON format:

            {{
            "player": "{request.player_name}",
            "team": "{request.team_name}",
            "position": "...",
            "fit_score": 0-100,
            "scheme_fit": "...",
            "depth_chart_impact": "...",
            "development_outlook": "...",
            "risk_factors": ["...", "..."],
            "overall_summary": "..."
            }}

            IMPORTANT:
                - overall_summary should be at least four sentences long and include player stats for the
                    last season
                - The teams are all colleges. No professional teams.
                - Position should be limited to: QB, RB, WR, TE, OL, DL, LB, CB, EDGE, S, K, P, ATH
                    - EDGE is a defensive position. Previously, referred to as defensive ends.
                - Factor in scheme_fit and development outlook into fit_score
                - Do NOT mention legal or off-field issues in risk factors.
                - Reminder: the year is 2026, transfers don't have to redshirt

            Never make things up!
            Do NOT infer or guess missing values.
            Return ONLY valid JSON.
            Do not include commentary, markdown, or trailing text.
            Ensure the JSON object is complete and properly closed.
            Use "N/A" for missing values.
        """

    # -------------- public API -----------------
    def select_relevant_information(self, driver, profile_url: str) -> RelevantInfoResponse:
        """
        Normalize the player profile webpage into structured football-relevant information.
        """
        page_text = fetch_website_contents(driver, profile_url)
        parsed_json = self._generate_with_retries(
            model=OllamaModels.LLAMA.value,
            system=REL_INFO_SYSTEM_PROMPT,
            prompt=page_text,
            max_retries=2,
        )
        return RelevantInfoResponse(**parsed_json)

    def summarize_transfer_fit(
        self, player_name: str, team_name: str, player_profile: RelevantInfoResponse
    ) -> PlayerFitSummary:
        """
        Generate a structured player fit summary using llama3.2.
        """
        self.logger.info(
            "Generating player fit summary | player='%s' | team='%s'",
            player_name,
            team_name,
        )
        request = PlayerFitRequest(
            player_name=player_name,
            team_name=team_name,
            player_profile=player_profile,
        )
        parsed_json = self._generate_with_retries(
            model=OllamaModels.LLAMA.value,
            system=PLAYER_FIT_SYSTEM_PROMPT,
            prompt=self._build_player_fit_prompt(request),
            max_retries=2,
        )
        return PlayerFitSummary(**parsed_json)

# =========================
# CLI entry point
# =========================

if __name__ == "__main__":
    """
    Usage: python player_fit_summarizer.py "Player Name" "Team Name"
    """

    if len(sys.argv) < 3:
        raise SystemExit(
            "Please provide a player name and team name.\n"
            "Example:\n"
            "  python player_fit_summarizer.py \"John Doe\" \"Texas\""
        )

    player_name = sys.argv[1]
    team_name = sys.argv[2]

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)

    summarizer = PlayerFitSummarizer()

    try:
        scraper = Sports247Scraper(driver)
        search_result = scraper.search_player_profile(player_name)

        if not search_result or not search_result.found:
            raise RuntimeError(f"No player profile found for '{player_name}'")

        relevant_info = summarizer.select_relevant_information(
            driver, str(search_result.profile_url)
        )

        summary = summarizer.summarize_transfer_fit(
            player_name=player_name,
            team_name=team_name,
            player_profile=relevant_info,
        )

        print(summary)

    finally:
        driver.quit()
