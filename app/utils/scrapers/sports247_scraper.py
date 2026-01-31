import logging
from typing import List, cast, Optional

from playwright.sync_api import Page, sync_playwright
from pydantic import HttpUrl

from app.models.player_fit_models import PlayerSearchResult
from app.utils.decorators import timed
from app.utils.scrapers.playwright_helpers import fetch_website_contents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

# Base URL for the 247Sports player search experience.
PLAYER_SEARCH_URL = "https://247sports.com/player"


# ---------------------------
# Test compatibility shim
# ---------------------------

class WebDriverWait:
    """
    Minimal wait shim to preserve unit test behavior.

    IMPORTANT:
    - This is NOT a real waiting mechanism.
    - Playwright already handles synchronization internally.
    - This shim exists purely to preserve existing test expectations.

    :param driver: PlaywrightDriver instance under test
    :param timeout: Ignored; preserved for interface compatibility
    """

    def __init__(self, driver, timeout: int):
        self.driver = driver
        self.timeout = timeout

    def until(self, condition):
        """
        Execute the provided condition callable immediately.

        :param condition: Callable accepting the driver and returning a value
        :return: The result of the condition callable
        """
        if callable(condition):
            result = condition(self.driver)

            # Synchronize current_url after condition execution
            self.driver.current_url = self.driver.page.url
            return result

        return condition


# ---------------------------
# Playwright-backed driver
# ---------------------------

class PlaywrightDriver:
    """
    Thin wrapper around a Playwright Page.

    This class intentionally exposes a small, stable surface area
    so that scraping logic remains readable and unit tests can
    mock browser behavior predictably.

    This is an adapter, not a general-purpose browser abstraction.
    """

    def __init__(self, page: Page):
        """
        :param page: Playwright Page instance
        """
        self.page = page
        self.current_url = ""

    def get(self, url: str) -> None:
        """
        Navigate to the given URL.

        :param url: Absolute URL to navigate to
        :return: None
        """
        self.page.goto(url, wait_until="domcontentloaded")
        self.current_url = self.page.url

    def find_elements(
        self,
        by: str,
        value: Optional[str] = None,
    ) -> List["PlaywrightElement"]:
        """
        Locate multiple elements using a CSS selector.

        Only CSS selectors are supported to keep the interface minimal
        and explicit.

        :param by: Locator strategy (must be "css selector")
        :param value: CSS selector string
        :return: List of PlaywrightElement wrappers
        """
        if by != "css selector":
            raise NotImplementedError("Only CSS selector is supported")

        elements = self.page.query_selector_all(value or "")
        return [PlaywrightElement(el) for el in elements]


class PlaywrightElement:
    """
    Lightweight wrapper around a Playwright element handle.

    Exposes a limited set of helper methods required by the scraper
    and test suite while keeping direct DOM access explicit.
    """

    def __init__(self, element):
        """
        :param element: Playwright element handle
        """
        self.element = element

    def clear(self) -> None:
        """
        Clear the contents of an input element.

        :return: None
        """
        self.element.fill("")

    def send_keys(self, keys: str) -> None:
        """
        Send keystrokes to the element.

        NOTE:
        - Newline ('\\n') is treated as Enter
        - All other input is typed verbatim

        :param keys: Text or newline to send
        :return: None
        """
        if keys == "\n":
            self.element.press("Enter")
        else:
            self.element.type(keys)

    def get_attribute(self, name: str) -> Optional[str]:
        """
        Retrieve an element attribute.

        :param name: Attribute name
        :return: Attribute value or None if missing
        """
        return self.element.get_attribute(name)

    def find_element(
        self,
        by: str,
        value: Optional[str] = None,
    ) -> "PlaywrightElement":
        """
        Locate a single descendant element using a CSS selector.

        :param by: Locator strategy (must be "css selector")
        :param value: CSS selector
        :return: Wrapped PlaywrightElement
        :raises ValueError: If the element cannot be found
        """
        if by != "css selector":
            raise NotImplementedError

        el = self.element.query_selector(value)
        if el is None:
            raise ValueError("Element not found")

        return PlaywrightElement(el)

    @property
    def text(self) -> str:
        """
        Return visible inner text of the element.

        :return: Stripped text content
        """
        return self.element.inner_text().strip()


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
