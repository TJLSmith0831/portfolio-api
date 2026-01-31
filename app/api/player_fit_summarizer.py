"""
The Ollama connection to the player-fit summarizer demo.
This module now uses a Playwright-based browser implementation
while preserving all original behavior and prompts.
"""
import json
import sys
import re
import logging

from fastapi import APIRouter
from playwright.sync_api import sync_playwright

from app.models.player_fit_models import (
    PlayerFitSummary,
    PlayerFitRequest,
    PlayerFitSummaryRequest,
    PlayerFitSummaryResponse,
    RelevantInfoResponse,
)
from app.utils.decorators import timed
from app.utils.scrapers.driver_singleton import get_driver, driver_lock
from app.utils.scrapers.playwright_helpers import fetch_website_contents
from app.utils.scrapers.sports247_scraper import Sports247Scraper, PlaywrightDriver
from app.llm_client import OllamaModels, get_llm_client, LLMClient

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
    clean_string = re.sub(r"[ \t]+$", "", raw_text)
    try:
        return json.loads(clean_string)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse JSON. Raw text received was {raw_text}. Error details:\n{exc}"
        )


def _parse_json_lenient(text: str) -> dict:
    """
    Parse JSON from LLM output with basic repair logic.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    open_braces = text.count("{")
    close_braces = text.count("}")

    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)

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


def _normalize_player_fit_json(
    data: dict,
    *,
    player_name: str,
    team_name: str,
) -> dict:
    """
    Normalize and harden LLM output so it always satisfies PlayerFitSummary.
    """

    raw_risks = data.get("risk_factors") or []
    normalized_risks: list[str] = []

    for risk in raw_risks:
        if isinstance(risk, str):
            normalized_risks.append(risk)
        elif isinstance(risk, dict):
            area = risk.get("area")
            severity = risk.get("severity")
            if area and severity:
                normalized_risks.append(f"{area} ({severity})")
            else:
                normalized_risks.append(json.dumps(risk))
        else:
            normalized_risks.append(str(risk))

    return {
        # AUTHORITATIVE FIELDS
        "player": player_name,
        "team": team_name,
        "position": data.get("position") or "Unknown Position",

        # SCORING
        "fit_score": int(data.get("fit_score", 0)),

        # ANALYSIS
        "scheme_fit": data.get("scheme_fit")
        or "No scheme fit was provided.",

        "depth_chart_impact": data.get("depth_chart_impact")
        or "No depth chart impact was provided.",

        "development_outlook": data.get("development_outlook")
        or "No development outlook was provided.",

        "risk_factors": normalized_risks,

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
    - If Transfer Player, Transfer portal or transfer prediction context (e.g., Crystal Ball, destinations, confidence)
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
            "transfer_ranking": "...",
            "prospect_ranking": "...'
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
You are a college football recruiting analyst.

Based on the provided player profile and the requested team,
evaluate the player's projected fit.

Return ONLY valid JSON using EXACTLY this structure:

{
  "position": "string",
  "fit_score": number (0-100),
  "scheme_fit": "string",
  "depth_chart_impact": "string",
  "development_outlook": "string",
  "risk_factors": ["string"],
  "overall_summary": "string"
}

Rules:
- Only reference the requested college team. No NFL teams.
- Do NOT repeat raw biographical data
- Do NOT invent facts not supported by the profile
- Double check you're using the right position. Use the two character abbreviation.
- If information is missing, explain uncertainty in text fields
- risk_factors MUST be an array (empty if none)
- Return JSON only
"""


# =========================
# PlayerFitSummarizer
# =========================

class PlayerFitSummarizer:
    """Object-oriented wrapper around the player fit summarization flow."""

    def __init__(self, client=None):
        """
        :param client: Optional Ollama client
        """
        self.client = client or get_llm_client()
        self.logger = log

    def _build_player_fit_prompt(self, request: PlayerFitRequest) -> str:
        """Build the LLM prompt for player fit analysis."""
        return f"""
        **Output valid JSON only**.

        Player: {request.player_name}
        Team: {request.team_name}

        Player profile information:
        {request.player_profile.model_dump_json(indent=2)}

        Evaluate scheme fit, roster impact, development trajectory, and risks.
        """

    @timed()
    def select_relevant_information(
        self,
        driver: PlaywrightDriver,
        profile_url: str,
        model: OllamaModels = OllamaModels.LLAMA_LATEST
    ) -> RelevantInfoResponse:
        """
        Extract and normalize relevant football information from a player profile.
        """
        page_text = fetch_website_contents(driver, profile_url)

        parsed_json = _chat_with_retries(
            client=self.client,
            model=model.value,
            system=REL_INFO_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": page_text}],
            max_retries=2,
        )

        return RelevantInfoResponse(**parsed_json)

    @timed()
    def summarizer_player_fit(
        self,
        player_name: str,
        team_name: str,
        player_profile: RelevantInfoResponse,
        model: OllamaModels = OllamaModels.LLAMA_LATEST
    ) -> PlayerFitSummary:
        """
        Generate a structured player fit summary.
        """
        request = PlayerFitRequest(
            player_name=player_name,
            team_name=team_name,
            player_profile=player_profile,
        )

        # Evaluate scheme fit, roster impact, development trajectory, and risks.
        parsed_json = _chat_with_retries(
            client=self.client,
            model=model.value,
            system=PLAYER_FIT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": self._build_player_fit_prompt(request),
                }
            ],
            max_retries=2,
        )

        # Normalize JSON to ensure consistent structure
        normalized = _normalize_player_fit_json(
            parsed_json,
            player_name=player_name,
            team_name=team_name,
        )
        return PlayerFitSummary(**normalized)

# =========================
# API entry point
# =========================

@router.post("/summarize_player_fit", response_model=PlayerFitSummaryResponse)
def summarize_player_fit(
    request: PlayerFitSummaryRequest,
    model: OllamaModels = OllamaModels.LLAMA_LATEST,
) -> PlayerFitSummaryResponse:
    """
    Generate a structured player fit summary using Playwright.
    """
    
    browser = get_driver()
    
    with driver_lock:
        context = browser.new_context()
        page = context.new_page()
        driver = PlaywrightDriver(page)

        client = LLMClient(model=model)
        summarizer = PlayerFitSummarizer(client=client)

        try:
            scraper = Sports247Scraper(driver)
            search_result = scraper.search_player_profile(request.player_name)

            if not search_result or not search_result.found:
                raise RuntimeError(
                    f"No player profile found for '{request.player_name}'"
                )

            relevant_info = summarizer.select_relevant_information(
                driver,
                str(search_result.profile_url),
            )

            summary = summarizer.summarizer_player_fit(
                player_name=request.player_name,
                team_name=request.requested_team_name,
                player_profile=relevant_info,
            )

            return PlayerFitSummaryResponse(summary=summary)

        finally:
            context.close()

# =========================
# CLI entry point
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python player_fit_summarizer.py \"Player Name\" \"Team Name\""
        )

    player_name = sys.argv[1]
    team_name = sys.argv[2]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        page = browser.new_page()
        driver = PlaywrightDriver(page)

        summarizer = PlayerFitSummarizer()

        try:
            scraper = Sports247Scraper(driver)
            search_result = scraper.search_player_profile(player_name)

            if not search_result or not search_result.found:
                raise RuntimeError(f"No player profile found for '{player_name}'")

            relevant_info = summarizer.select_relevant_information(
                driver=driver,
                profile_url=str(search_result.profile_url),
                model=OllamaModels.LLAMA_SMALL
            )

            summary = summarizer.summarizer_player_fit(
                player_name=player_name,
                team_name=team_name,
                player_profile=relevant_info,
                model=OllamaModels.LLAMA_SMALL
            )

            print(summary)

        finally:
            browser.close()
