import json
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from typing_extensions import Optional

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

# Cap excerpt length sent to the LLM (reduces tokens / memory on small droplets)
EXCERPT_MAX_CHARS = 1200

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
    """
    Attempt to parse a JSON object from LLM output with minimal repair logic.

    The function first attempts strict JSON parsing. If parsing fails due to
    unbalanced braces, it appends missing closing braces and retries.

    :param text: Raw text returned by the language model.
    :type text: str
    :return: Parsed JSON object.
    :rtype: dict
    :raises json.JSONDecodeError: If parsing fails after repair attempts.
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
    """
    Execute a chat completion with retries and JSON repair handling.

    This function submits a chat request to the LLM, attempts to parse the
    response as JSON, and retries with a repair prompt if parsing fails.

    :param client: LLM client instance.
    :param model: Model identifier to use.
    :param system: Optional system prompt.
    :param messages: User/assistant message history.
    :param max_retries: Maximum number of retry attempts.
    :type max_retries: int
    :return: Parsed JSON response from the model.
    :rtype: dict
    :raises RuntimeError: If all retry attempts fail.
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
                "Return ONLY a complete JSON object that matches this template. "
                "No markdown, no code fences, no comments, no extra keys.\n\n"
                f"{PLAYER_FIT_JSON_TEMPLATE}\n\n"
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
    """
    Normalize raw LLM output into a PlayerFitSummary-compatible payload.

    This includes coercing types, normalizing risk factors, clamping the
    fit score, and injecting player/team identifiers.

    :param data: Raw JSON returned by the LLM.
    :param player_name: Player name for attribution.
    :param team_name: Team name for attribution.
    :return: Normalized dictionary suitable for PlayerFitSummary.
    :rtype: dict
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
    Extract structured identity metadata and profile text from scraped content.

    The function detects identity and profile markers and splits the payload
    accordingly, returning a key-value identity map and cleaned profile text.

    :param extracted_text: Full scraped page text.
    :type extracted_text: str
    :return: Tuple of (identity_map, profile_text).
    :rtype: tuple[dict[str, str], str]
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


def _trim_structured_for_prompt(
    relevant_info: RelevantInfoResponse,
    raw_excerpt: str,
    *,
    profile_snippet_chars: int = 600,
) -> str:
    """
    Build a minimal JSON string for the LLM prompt (identity, school, short excerpt).
    Reduces token count while keeping behavior consistent.
    """
    trimmed: Dict[str, object] = {}
    if relevant_info.identity:
        trimmed["identity"] = relevant_info.identity
    if relevant_info.school_context:
        trimmed["school_context"] = relevant_info.school_context
    if raw_excerpt:
        trimmed["profile_excerpt"] = raw_excerpt[:profile_snippet_chars]
    return json.dumps(trimmed, indent=2)


def _build_relevant_info_response(
    identity_map: dict[str, str],
    profile_text: str,
) -> RelevantInfoResponse:
    """
    Build a RelevantInfoResponse from scraped identity and profile text.

    :param identity_map: Parsed identity metadata.
    :param profile_text: Main profile text content.
    :return: Structured RelevantInfoResponse object.
    :rtype: RelevantInfoResponse
    """
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
    Validate the structure and semantic correctness of a player-fit payload.

    :param data: Parsed JSON output from the LLM.
    :return: Tuple of (is_valid, list_of_issues).
    :rtype: tuple[bool, list[str]]
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
    """
    Build a repair prompt to request missing or invalid fields from the LLM.

    :param missing_reasons: List of validation failures.
    :param structured_json: Structured profile context.
    :param raw_excerpt: Raw source excerpt.
    :param player_name: Player name.
    :param team_name: Team name.
    :return: Repair prompt text.
    :rtype: str
    """
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
        """
        Initialize the PlayerFitSummarizer with an LLM client.

        :param client: LLM client instance.
        """
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
        """
        Scrape the profile page and extract relevant information.

        :param driver: PlaywrightDriver instance.
        :param profile_url: URL of the profile page.
        :return: ScrapedProfile instance.
        """
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
        raw_excerpt = (relevant_info.raw_text or "")[:EXCERPT_MAX_CHARS]

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
        prompt_overrides: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build the prompt for player fit analysis.

        :param request: PlayerFitRequest instance.
        :param structured_json: Structured JSON data.
        :param raw_excerpt: Raw excerpt from the profile page.
        :param prompt_overrides: Optional prompt overrides.
        :return: Prompt string.
        """
        overrides = prompt_overrides or {}
        
        guardrails = "\n".join(
            f"- {text}" for text in overrides.values()
        )
            
        return (
            "Output valid JSON only. No markdown, no code fences, no extra keys, no commentary.\n\n"
            f"Player: {request.player_name}\n"
            f"Team: {request.team_name}\n\n"
            "Structured profile data (scraped context summarized below):\n"
            f"{structured_json}\n\n"
            "Primary source excerpt (verbatim, truncated to 1200 chars):\n"
            f"{raw_excerpt}\n\n"
            f"{guardrails}\n\n"
            "Instructions:\n"
            "- Populate every field in the template below with grounded analysis.\n"
            "- Do not repeat raw biographical facts unless they support the evaluation.\n"
            "- Always justify the fit_score using evidence from the excerpt. Use a conservative scale (average ~72; 90+ is rare).\n"
            "- risk_factors must be an array. Use [] if none are identified.\n"
            "- Write scheme_fit, depth_chart_impact, and development_outlook as distinct paragraphs, "
            f"explicitly tied to {request.team_name}'s program identity, roster context, and development philosophy.\n"
            f"- overall_summary must be 2-3 sentences synthesizing the recommendation relative to {request.team_name}'s program.\n"
            "- Return only the JSON object. No markdown, no ```json, no explanation, no extra keys beyond the template.\n\n"
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
        prompt_overrides: dict | None = None,
    ) -> dict:
        """
        Invoke the player fit model to generate a JSON response.

        :param player_name: Player's name.
        :param team_name: Team's name.
        :param relevant_info: Relevant information about the player.
        :param structured_json: Structured JSON data.
        :param raw_excerpt: Raw excerpt from the profile page.
        :param prompt_overrides: Optional prompt overrides.
        :return: JSON response.
        """
        prompt_overrides = prompt_overrides or {}
        trimmed_json = _trim_structured_for_prompt(relevant_info, raw_excerpt)

        request = PlayerFitRequest(
            player_name=player_name,
            team_name=team_name,
            player_profile=relevant_info,
        )
        base_prompt = self._build_player_fit_prompt(
            request,
            structured_json=trimmed_json,
            raw_excerpt=raw_excerpt,
            prompt_overrides=prompt_overrides
        )

        parsed_json = _chat_with_retries(
            client=self.client,
            model=self.client.model_enum.value,
            system="You are an elite college football recruiting analyst. Output only valid JSON.",
            messages=[{"role": "user", "content": base_prompt}],
            max_retries=0,
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
            system="You are an elite college football recruiting analyst. Output only valid JSON.",
            messages=[{"role": "user", "content": repair_prompt}],
            max_retries=0,
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
        model: OllamaModels = OllamaModels.GEMMA4_31B,  # retained for compatibility
    ) -> RelevantInfoResponse:
        """
        Scrape the player profile and return structured information plus raw excerpt.
        
        :param driver: Playwright driver.
        :param profile_url: Player's profile URL.
        :param model: Ollama model.
        :return: Structured information and raw excerpt.
        """
        scraped = self._scrape_profile(driver, profile_url)
        return scraped.relevant_info

    @timed()
    def summarize_player_fit(
        self,
        player_name: str,
        team_name: str,
        player_profile: RelevantInfoResponse | None,
        model: OllamaModels = OllamaModels.GEMMA4_31B,
        prompt_overrides: dict | None = None,
    ) -> PlayerFitSummary:
        """
        Generate the full player-fit summary in a single LLM call with validation.
        
        :param player_name: Player's name.
        :param team_name: Team's name.
        :param player_profile: Player's profile information.
        :param model: Ollama model.
        :param prompt_overrides: Prompt overrides.
        :return: Player-fit summary.
        """
        if not player_profile:
            raise ValueError("player_profile is required for summarizer_player_fit")
            
        prompt_overrides = prompt_overrides or {}

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
            raw_excerpt=raw_excerpt[:EXCERPT_MAX_CHARS],
            prompt_overrides=prompt_overrides,
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

            summary = summarizer.summarize_player_fit(
                player_name=player_name,
                team_name=team_name,
                player_profile=relevant_info,
            )

            print(summary)

        finally:
            browser.close()
