import json
import sys
import logging

from playwright.sync_api import sync_playwright

from app.utils.decorators import timed
from app.utils.scrapers.playwright_helpers import fetch_website_contents, PlaywrightDriver
from app.utils.scrapers.sports247_scraper import Sports247Scraper

from app.llm_client import get_llm_client, OllamaModels
from app.models.player_fit_models import (
    PlayerFitRequest,
    PlayerFitSummary,
    RelevantInfoResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

# =========================
# Helper functions
# =========================

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
        log.info("LLM attempt %d/%d", attempt, max_retries + 1)

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        try:
            response = client.chat(
                model=model,
                messages=chat_messages,
                temperature=0,
                max_tokens=500,
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

    Your task is to extract ONLY information that is explicitly and unambiguously stated
    verbatim in the provided text. You are performing factual information extraction,
    not analysis or inference.

    CRITICAL RULE:
    Player position is a factual field.
    You may ONLY populate "identity.position" if the exact position label
    (e.g., "QB", "Quarterback", "Linebacker") appears explicitly in the text.
    Do NOT infer position from statistics, context, archetypes, or football knowledge.
    If no explicit position is present, set the value to null.

    Focus on high-signal football information suitable for normalization into
    a structured player record.
    Ignore navigation elements, marketing copy, duplicated sections, and site boilerplate.

    Prioritize extracting the following categories when explicitly present:

    - Player identity (name, position, height, weight)
    - Current and former schools
    - Transfer portal or transfer prediction context when explicitly mentioned
    - Recruiting or transfer rankings
    - Most recent season statistics
    - Experience year / class
    - High school and hometown
    - Notable recent headlines (titles only)

    Explicitly IGNORE:
    - Login prompts, subscription offers, ads
    - Navigation, footers, UI labels
    - Duplicated tables or repeated labels

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

    Rules:
    - Only include fields when the information is explicitly present in the text
    - Every value must be directly supported by the text
    - When information is missing or ambiguous, use null
    - Do NOT infer or guess
    - Do NOT use N/A, undefined, or comments
    - Output valid JSON only
    - Do not wrap the response in Markdown
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
        model: OllamaModels = OllamaModels.LLAMA_3B
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
        model: OllamaModels = OllamaModels.LLAMA_3B
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
                model=OllamaModels.LLAMA_1B
            )

            summary = summarizer.summarizer_player_fit(
                player_name=player_name,
                team_name=team_name,
                player_profile=relevant_info,
                model=OllamaModels.LLAMA_1B
            )

            print(summary)

        finally:
            browser.close()
