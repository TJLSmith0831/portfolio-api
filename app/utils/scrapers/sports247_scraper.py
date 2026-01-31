import logging
from typing import cast

from playwright.sync_api import sync_playwright
from pydantic import HttpUrl

from app.models.player_fit_models import PlayerSearchResult
from app.utils.decorators import timed
from app.utils.scrapers.playwright_helpers import PlaywrightDriver, PlaywrightElement, fetch_website_contents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

# Base URL for the 247Sports player search experience.
PLAYER_SEARCH_URL = "https://247sports.com/player"

# ---------------------------
# Scraper
# ---------------------------

class Sports247Scraper:
    """
    Playwright-based 247Sports scraper.

    Responsibilities:
    - Execute player search via the 247Sports UI
    - Resolve player profile URLs deterministically
    - Return a structured PlayerSearchResult

    Non-responsibilities:
    - Player evaluation or inference
    - Data normalization beyond extraction
    - URL validation beyond presence
    """

    def __init__(self, driver: PlaywrightDriver, logger=None):
        """
        :param driver: PlaywrightDriver instance
        :param logger: Optional logger override
        """
        self.driver = driver
        self.logger = logger or log


    @timed()
    def search_player_profile(self, player_name: str) -> PlayerSearchResult:
        """
        Search for a player profile on 247Sports and return the first matching result.

        The function submits a name-based search via the React-controlled input,
        waits for results to fully hydrate, parses the rendered list, and navigates
        to the selected player profile if found.

        :param player_name: Full name of the player to search
        :return: PlayerSearchResult describing the outcome of the search
        """
        entry_url = PLAYER_SEARCH_URL
        self.driver.get(entry_url)

        # Locate the React-controlled search input
        input_handle = self.driver.page.wait_for_selector(
            "input#FullName",
            state="attached",
            timeout=10_000,
        )

        if input_handle is None:
            raise RuntimeError("Player search input did not render")

        search_box = PlaywrightElement(input_handle)

        # Submit the search query
        search_box.clear()
        search_box.send_keys(player_name)
        search_box.send_keys("\n")

        # Wait for at least one real player result to be rendered
        self.driver.page.wait_for_selector(
            "ul.content-list li a[href*='/player/']",
            timeout=10_000,
        )

        current_url = self.driver.current_url
        self.logger.info("Post-search URL: %s", current_url)

        # Presence of rendered results determines success, not URL alone.
        if current_url.rstrip("/") == PLAYER_SEARCH_URL:
            self.logger.info("Search stayed on player index; parsing results")

        results = self.driver.find_elements(
            "css selector",
            "ul.content-list > li",
        )

        self.logger.info("Found %d raw results", len(results))

        # Filter out non-player rows (e.g., summary/header items)
        player_results = [
            li
            for li in results
            if li.element.query_selector("a[href*='/player/']") is not None
        ]

        if not player_results:
            return PlayerSearchResult(
                query=player_name,
                found=False,
            )

        first_result = player_results[0]

        link = first_result.find_element(
            "css selector",
            "a[href*='/player/']",
        )

        profile_url: str | None = link.get_attribute("href")
        displayed_name: str = link.text.strip()

        if not profile_url:
            return PlayerSearchResult(
                query=player_name,
                found=False,
            )

        self.logger.info(
            "Player search result selected: displayed_name='%s', profile_url='%s'",
            displayed_name,
            profile_url,
        )

        # Navigate to the player profile page
        self.driver.get(profile_url)

        return PlayerSearchResult(
            query=player_name,
            found=True,
            profile_url=cast(HttpUrl, profile_url),
            displayed_name=displayed_name,
        )


# ---------------------------
# Manual runner
# ---------------------------

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
            ],
        )
        page = browser.new_page()

        driver = PlaywrightDriver(page)
        scraper = Sports247Scraper(driver)

        results = scraper.search_player_profile("Darian Mensah")
        log.info(
            "Final result: displayed_name='%s', profile_url='%s'",
            results.displayed_name,
            results.profile_url,
        )

        if results.found and results.profile_url:
            profile = fetch_website_contents(
                scraper.driver,
                str(results.profile_url),
            )
            log.info("Fetched profile content (%d chars)", len(profile))
            log.info("Profile Content", profile)

        browser.close()
