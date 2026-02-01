import json
import logging
import sys
from dataclasses import dataclass
from typing import Iterable, Tuple

from playwright.sync_api import sync_playwright

from app.llm_client import OllamaModels, get_llm_client
from app.models.player_fit_models import (
    PlayerFitRequest,
    PlayerFitSummary,
    RelevantInfoResponse,
)
from app.utils.decorators import timed
from app.utils.scrapers.playwright_helpers import PlaywrightDriver, fetch_website_contents
from app.utils.scrapers.sports247_scraper import Sports247Scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

PLAYER_FIT_JSON_TEMPLATE = """{
  "position": "string",
  "fit_score": number (0-100),
  "scheme_fit": "string",
  "depth_chart_impact": "string",
  "development_outlook": "string",
  "risk_factors": ["string"],
  "overall_summary": "string"
}"""

REQUIRED_TEXT_FIELDS = [
    "scheme_fit",
    "depth_chart_impact",
    "development_outlook",
    "overall_summary",
]

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON from LLM output with extremely small repair logic."""
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
    """
    Execute a chat completion with light retry / JSON repair logic.
    """
    last_error = None

    for attempt in range(1, max_retries + 2):
        log.info("LLM attempt %d/%d", attempt, max_retries + 1)

        chat_messages: list[dict] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        try:
            response = client.chat(
                model=model,
                messages=chat_messages,
                temperature=0,
                max_tokens=400,
            )
        except Exception as exc:  # pragma: no cover - network failures
            last_error = exc
            log.warning("Chat call failed on attempt %d: %s", attempt, exc)
            continue

        raw_text = response.choices[0].message.content.strip()
        log.debug("Raw LLM output:\n%s", raw_text)

        try:
            return _parse_json_lenient(raw_text)
        except Exception as exc:
            last_error = exc
            log.warning("JSON parse failed on attempt %d: %s", attempt, exc)

            repair_prompt = (
                "The previous response was not valid JSON.\n\n"
                "Return ONLY a complete JSON object that matches this template:\n"
                f"{PLAYER_FIT_JSON_TEMPLATE}\n\n"
                "Do not include markdown, comments, or explanations.\n"
                "Use null only when information is truly absent.\n\n"
                f"Previous response:\n{raw_text}"
            )

            messages = [{"role": "user", "content": repair_prompt}]

    raise RuntimeError(
        f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
    )


def _normalize_player_fit_json(
    data: dict,
    *,
    player_name: str,
    team_name: str,
) -> dict:
    """Normalize the LLM response into a PlayerFitSummary-compatible dict."""
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

    fit_score_raw = data.get("fit_score", 0)
    try:
        fit_score_value = int(fit_score_raw)
    except (TypeError, ValueError):
        fit_score_value = 0

    return {
        "player": player_name,
        "team": team_name,
        "position": data.get("position") or "Unknown Position",
        "fit_score": fit_score_value,
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


def _extract_identity_and_profile(extracted_text: str) -> Tuple[dict[str, str], str]:
    """
    Extract the structured identity snippet and cleaned main text from the page payload.
    """
    marker_identity = "=== PLAYER IDENTITY ==="
    marker_profile = "=== PROFILE CONTENT ==="

    if marker_identity not in extracted_text:
        return {}, extracted_text.strip()

    _, after_identity = extracted_text.split(marker_identity, 1)

    if marker_profile in after_identity:
        identity_block, profile_block = after_identity.split(marker_profile, 1)
    else:
        identity_block = after_identity
        profile_block = ""

    identity_map: dict[str, str] = {}

    for raw_line in identity_block.strip().splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key:
            identity_map[key] = value

    profile_text = profile_block.strip() or extracted_text.strip()
    return identity_map, profile_text


def _build_relevant_info_response(
    identity_map: dict[str, str],
    profile_text: str,
) -> RelevantInfoResponse:
    """Convert the scraped artifacts into a RelevantInfoResponse."""
    identity_section = {
        "name": identity_map.get("Name"),
        "position": identity_map.get("Position"),
        "height": identity_map.get("Height"),
        "weight": identity_map.get("Weight"),
    }
    identity_section = {k: v for k, v in identity_section.items() if v}

    school_context = {}
    affiliation = identity_map.get("Affiliation")
    if affiliation:
        school_context["current_school"] = affiliation

    profile_excerpt = profile_text.strip()

    background = {}
    if profile_excerpt:
        background["profile_text"] = profile_excerpt
    if identity_map:
        background["identity_labels"] = identity_map

    return RelevantInfoResponse(
        identity=identity_section or None,
        school_context=school_context or None,
        rankings=None,
        latest_season_stats=None,
        background=background or None,
        raw_text=profile_excerpt or None,
        notable_headlines=[],
    )


def _validate_player_fit_payload(data: dict) -> tuple[bool, list[str]]:
    """
    Verify the LLM output satisfies the required schema and semantics.
    Returns (is_valid, list_of_issues)
    """
    issues: list[str] = []

    fit_score = data.get("fit_score")

    if isinstance(fit_score, str):
        try:
            fit_score_int = int(float(fit_score))
            if fit_score_int < 0 or fit_score_int > 100:
                issues.append("fit_score must be between 0 and 100")
        except (ValueError, TypeError):
            issues.append("fit_score must be a valid integer between 0 and 100")

    for field in REQUIRED_TEXT_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{field} must be a non-empty string")

    risks = data.get("risk_factors")
    if not isinstance(risks, list):
        issues.append("risk_factors must be an array (can be empty)")

    return (len(issues) == 0), issues


def _build_repair_prompt(
    *,
    missing_reasons: Iterable[str],
    structured_json: str,
    raw_excerpt: str,
    player_name: str,
    team_name: str,
) -> str:
    """Construct the follow-up prompt when fields are missing."""
    missing_bullets = "\n".join(f"- {reason}" for reason in missing_reasons)

    return (
        "Your previous JSON omitted required information. Rewrite the JSON so every "
        "required field is populated with substantive analysis grounded in the excerpt.\n\n"
        "Missing / invalid items:\n"
        f"{missing_bullets}\n\n"
        "Return ONLY JSON that matches this template exactly:\n"
        f"{PLAYER_FIT_JSON_TEMPLATE}\n\n"
        f"Player: {player_name}\n"
        f"Team: {team_name}\n\n"
        "Structured profile data (for reference):\n"
        f"{structured_json}\n\n"
        "Primary source excerpt (verbatim):\n"
        f"{raw_excerpt}\n"
    )


# --------------------------------------------------------------------------- #
# PlayerFitSummarizer                                                         #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ScrapedProfile:
    """Container for the scraped content used by the summarizer."""

    relevant_info: RelevantInfoResponse
    structured_json: str
    raw_excerpt: str


class PlayerFitSummarizer:
    """Single-pass player fit summarization pipeline."""

    def __init__(self, client=None) -> None:
        self.client = client or get_llm_client()
        self.logger = log

    # ------------------------------------------------------------------ #
    # Scraping helpers                                                   #
    # ------------------------------------------------------------------ #

    def _scrape_profile(
        self,
        driver: PlaywrightDriver,
        profile_url: str,
    ) -> ScrapedProfile:
        page_payload = fetch_website_contents(driver, profile_url)
        identity_map, profile_text = _extract_identity_and_profile(page_payload)

        self.logger.info(
            "scrape_profile extracted identity keys: %s", list(identity_map.keys())
        )
        self.logger.info(
            "scrape_profile profile excerpt length: %d", len(profile_text),
        )

        relevant_info = _build_relevant_info_response(identity_map, profile_text)

        structured_json = relevant_info.model_dump_json(
            indent=2,
            exclude_none=True,
        )
        raw_excerpt = (relevant_info.raw_text or "")[:1800]

        background = dict(relevant_info.background or {})
        if raw_excerpt and not background.get("profile_text"):
            background["profile_text"] = raw_excerpt
        background["structured_json"] = structured_json
        relevant_info.background = background
        relevant_info.raw_text = raw_excerpt

        return ScrapedProfile(
            relevant_info=relevant_info,
            structured_json=structured_json,
            raw_excerpt=raw_excerpt,
        )

    # ------------------------------------------------------------------ #
    # Prompting                                                          #
    # ------------------------------------------------------------------ #

    def _build_player_fit_prompt(
        self,
        request: PlayerFitRequest,
        *,
        structured_json: str,
        raw_excerpt: str,
    ) -> str:
        return (
            "**Output valid JSON only.**\n\n"
            f"Player: {request.player_name}\n"
            f"Team: {request.team_name}\n\n"
            "Structured profile data (scraped context summarized below):\n"
            f"{structured_json}\n\n"
            "Primary source excerpt (verbatim, truncated to 1800 chars):\n"
            f"{raw_excerpt}\n\n"
            "Instructions:\n"
            "- Populate every field in the template below with grounded analysis.\n"
            "- Do not repeat raw biographical facts unless they support the evaluation.\n"
            "- Always justify the fit_score using evidence from the excerpt.\n"
            "- risk_factors must be an array. Use [] if none are identified.\n"
            "- Write scheme_fit, depth_chart_impact, and development_outlook as distinct paragraphs.\n"
            "- overall_summary must be 2-3 sentences synthesizing the recommendation.\n"
            "- Output JSON only—no markdown, commentary, or extra keys.\n\n"
            "Template:\n"
            f"{PLAYER_FIT_JSON_TEMPLATE}\n"
        )

    def _invoke_player_fit_model(
        self,
        *,
        player_name: str,
        team_name: str,
        relevant_info: RelevantInfoResponse,
        structured_json: str,
        raw_excerpt: str,
    ) -> dict:
        request = PlayerFitRequest(
            player_name=player_name,
            team_name=team_name,
            player_profile=relevant_info,
        )
        base_prompt = self._build_player_fit_prompt(
            request,
            structured_json=structured_json,
            raw_excerpt=raw_excerpt,
        )

        parsed_json = _chat_with_retries(
            client=self.client,
            model=self.client.model_enum.value,
            system="You are an elite college football recruiting analyst.",
            messages=[{"role": "user", "content": base_prompt}],
            max_retries=1,
        )

        valid, issues = _validate_player_fit_payload(parsed_json)

        if valid:
            return parsed_json

        repair_prompt = _build_repair_prompt(
            missing_reasons=issues,
            structured_json=structured_json,
            raw_excerpt=raw_excerpt,
            player_name=player_name,
            team_name=team_name,
        )

        self.logger.info(
            "First LLM response missing required fields. Triggering repair: %s", issues
        )

        repaired_json = _chat_with_retries(
            client=self.client,
            model=self.client.model_enum.value,
            system="You are an elite college football recruiting analyst.",
            messages=[{"role": "user", "content": repair_prompt}],
            max_retries=1,
        )

        return repaired_json

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @timed()
    def select_relevant_information(
        self,
        driver: PlaywrightDriver,
        profile_url: str,
        model: OllamaModels = OllamaModels.LLAMA_1B,  # retained for compatibility
    ) -> RelevantInfoResponse:
        """
        Scrape the player profile and return structured information plus raw excerpt.
        """
        scraped = self._scrape_profile(driver, profile_url)
        return scraped.relevant_info

    @timed()
    def summarizer_player_fit(
        self,
        player_name: str,
        team_name: str,
        player_profile: RelevantInfoResponse | None,
        model: OllamaModels = OllamaModels.LLAMA_1B,
    ) -> PlayerFitSummary:
        """
        Generate the full player-fit summary in a single LLM call with validation.
        """
        if not player_profile:
            raise ValueError("player_profile is required for summarizer_player_fit")

        background = player_profile.background or {}
        raw_excerpt = player_profile.raw_text or background.get("profile_text", "")

        structured_json = background.get("structured_json") or player_profile.model_dump_json(
            indent=2,
            exclude_none=True,
        )

        parsed_json = self._invoke_player_fit_model(
            player_name=player_name,
            team_name=team_name,
            relevant_info=player_profile,
            structured_json=structured_json,
            raw_excerpt=raw_excerpt[:1800],
        )

        normalized = _normalize_player_fit_json(
            parsed_json,
            player_name=player_name,
            team_name=team_name,
        )

        return PlayerFitSummary(**normalized)


# --------------------------------------------------------------------------- #
# CLI entry point                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            'Usage: python player_fit_summarizer.py "Player Name" "Team Name"'
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
            )

            summary = summarizer.summarizer_player_fit(
                player_name=player_name,
                team_name=team_name,
                player_profile=relevant_info,
            )

            print(summary)

        finally:
            browser.close()
