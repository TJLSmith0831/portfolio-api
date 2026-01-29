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
from selenium.webdriver.chrome.service import Service

from fastapi import APIRouter

from app.models.player_fit_models import PlayerFitSummary, PlayerFitRequest, PlayerFitSummaryRequest, PlayerFitSummaryResponse, RelevantInfoResponse
from app.scrapers.generic_helpers import fetch_website_contents
from app.scrapers.sports247_scraper import Sports247Scraper
from app.ollama_client import OllamaModels, get_llm_client, LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

SUMMARIZER_ENDPOINT = "/summarize_player_fit"

router = APIRouter(tags=["Summarize Player Fit"])

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

def _parse_json_lenient(text: str) -> dict:
    """
    Parse JSON from LLM output.
    Repairs common LLM failure modes:
      - missing closing braces
      - trailing whitespace
    """
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ---- Repair: balance braces ----
    open_braces = text.count("{")
    close_braces = text.count("}")

    if open_braces > close_braces:
        text = text + ("}" * (open_braces - close_braces))

    return json.loads(text)


def _chat_with_retries(
    client,
    model: str,
    system: str | None,
    messages: list[dict],
    max_retries: int = 2,
) -> dict:
    last_error = None

    for attempt in range(1, max_retries + 2):
        log.info("Ollama attempt %d/%d", attempt, max_retries + 1)

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        try:
            response = client.chat(
                model=model,
                messages=chat_messages,
            )
        except Exception as exc:
            last_error = exc
            log.warning("Chat call failed on attempt %d: %s", attempt, exc)
            continue

        # ---- Extract raw text (Ollama-safe) ----
        raw_text = response.choices[0].message.content.strip()
        log.debug("Raw LLM output:\n%s", raw_text)

        try:
            return _parse_json_lenient(raw_text)
        except Exception as exc:
            last_error = exc
            log.warning("JSON parse failed on attempt %d: %s", attempt, exc)

            # ---- Repair prompt only if retrying ----
            messages = [
                {
                    "role": "user",
                    "content": (
                        "The previous response was not valid JSON.\n\n"
                        "Return ONLY a complete, valid JSON object.\n"
                        "Do not include markdown, comments, or explanations.\n"
                        "Use null for unknown values.\n\n"
                        f"Previous response:\n{raw_text}"
                    ),
                }
            ]

    raise RuntimeError(
        f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
    )

def _normalize_player_fit_json(data: dict) -> dict:
    """
    Ensure all required PlayerFitSummary fields exist with safe defaults.
    This guarantees Pydantic validation will not fail due to missing keys.
    """
    return {
        "player": data.get("player"),
        "team": data.get("team"),
        "position": data.get("position"),
        "fit_score": data.get("fit_score", 0),
        "scheme_fit": data.get("scheme_fit") or "No scheme fit was provided.",
        "depth_chart_impact": data.get("depth_chart_impact") or "No depth chart impact was provided.",
        "development_outlook": data.get("development_outlook") or "No development outlook was provided.",
        "risk_factors": data.get("risk_factors") or [],
        "overall_summary": data.get("overall_summary")
            or "No detailed summary was generated based on the available information.",
    }

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
    Output valid JSON only.
    Do not use N/A, undefined, or comments.
    Use null for unknown values.
    Do not wrap the response in Markdown or code fences.
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
    Output valid JSON only.
    Do not use N/A, undefined, or comments.
    Use null for unknown values.
    Do not wrap the response in Markdown or code fences.
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
        self.client = client or get_llm_client()
        self.logger = log

    # -------------- private helpers -----------------
    def _extract_json(self, raw_text: str) -> dict:
        return _extract_json(raw_text)

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
            Use null for missing values.
            All fields MUST be present in the JSON object.
        """

    # -------------- public API -----------------
    def select_relevant_information(self, driver, profile_url: str) -> RelevantInfoResponse:
        """
        Normalize the player profile webpage into structured football-relevant information.
        """
        page_text = fetch_website_contents(driver, profile_url)
        messages = [
            {"role": "user", "content": page_text}
        ]

        parsed_json = _chat_with_retries(
            client=self.client,
            model=OllamaModels.LLAMA.value,
            system=REL_INFO_SYSTEM_PROMPT,
            messages=messages,
            max_retries=2,
        )
        return RelevantInfoResponse(**parsed_json)

    def summarize_transfer_fit(self, player_name: str, team_name: str,
                               player_profile: RelevantInfoResponse) -> PlayerFitSummary:
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
        messages = [
            {
                "role": "user",
                "content": self._build_player_fit_prompt(request),
            }
        ]
        parsed_json = _chat_with_retries(
            client=self.client,
            model=OllamaModels.LLAMA.value,
            system=PLAYER_FIT_SYSTEM_PROMPT,
            messages=messages,
            max_retries=2,
        )
        normalized = _normalize_player_fit_json(parsed_json)
        return PlayerFitSummary(**normalized)


# =========================
# API entry point
# =========================

@router.post("/summarize_player_fit", response_model=PlayerFitSummaryResponse)
def summarize_player_fit(request: PlayerFitSummaryRequest,
                         model: OllamaModels = OllamaModels.LLAMA) -> PlayerFitSummaryResponse:
    """
    Generate a structured player fit summary using llama3.2.

    :param request: The player fit summary request containing the player's name and team name.
    :param model: The Ollama model to use for generating the summary.
    :return: The player fit summary response containing the structured summary.
    """
    # Use the incoming request fields instead of sys.argv
    player_name = request.player_name
    team_name = request.requested_team_name

    options = Options()
    options.binary_location = "/usr/bin/chromium"
    
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    service = Service("/usr/bin/chromedriver")
    
    driver = webdriver.Chrome(service=service, options=options)

    client = LLMClient(model=model)

    summarizer = PlayerFitSummarizer(client=client)

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

        return PlayerFitSummaryResponse(summary=summary)

    finally:
        driver.quit()


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
