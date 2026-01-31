from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
import json
import logging
from typing import List, Optional

from playwright.sync_api import Page

log = logging.getLogger(__name__)

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



def fetch_website_contents(driver, url: str) -> str:
    """
    Extract high-signal, football-relevant content from a 247Sports player page.

    Strategy:
    1. Navigate to page (DOM ready only)
    2. Wait for main semantic container
    3. Extract canonical structured data
    4. Extract cleaned main text content
    5. Return compact, LLM-ready payload

    :param driver: PlaywrightDriver-compatible wrapper
    :param url: Fully qualified player profile URL
    :return: Normalized textual content for LLM ingestion
    """
    log.info("Fetching website contents: %s", url)

    page = driver.page

    # Cold-start-safe navigation
    page.goto(url, wait_until="domcontentloaded")

    # Retry selector wait once to absorb cold start
    for attempt in range(2):
        try:
            log.info("Waiting for main content container (attempt %d)", attempt + 1)
            page.wait_for_selector(
                "section.main-content.full",
                timeout=20_000,
            )
            break
        except PlaywrightTimeoutError:
            if attempt == 1:
                raise PlaywrightTimeoutError(
                    "Timed out waiting for main content container"
                )
            log.warning("Main container not ready, retrying once")

    parts: list[str] = []

    # --------------------------------------------------
    # 1. Canonical structured data (ld+json)
    # --------------------------------------------------
    try:
        log.info("Extracting structured ld+json data")
        ld_json_nodes = page.locator('script[type="application/ld+json"]')

        for i in range(ld_json_nodes.count()):
            raw = ld_json_nodes.nth(i).text_content()
            if not raw:
                continue

            data = json.loads(raw)

            if isinstance(data, dict) and data.get("@type") == "Person":
                log.info("Found Person structured data")

                parts.append("=== PLAYER IDENTITY ===")
                parts.append(f"Name: {data.get('name')}")

                affiliation = data.get("affiliation", {})
                if isinstance(affiliation, dict):
                    parts.append(f"Affiliation: {affiliation.get('name')}")

                height = (
                    data.get("height", [{}])[0].get("value")
                    if isinstance(data.get("height"), list)
                    else None
                )
                weight = (
                    data.get("weight", [{}])[0].get("value")
                    if isinstance(data.get("weight"), list)
                    else None
                )

                if height:
                    parts.append(f"Height: {height}")
                if weight:
                    parts.append(f"Weight: {weight}")

                break
    except Exception as exc:
        raise ValueError(f"Failed extracting structured data: {exc}")

    # --------------------------------------------------
    # 2. Main semantic content only
    # --------------------------------------------------
    try:
        log.info("Cleaning noisy DOM elements inside main content")
        page.evaluate(
            """
            () => {
                const container = document.querySelector('section.main-content.full');
                if (!container) return;
                ['script', 'style', 'img', 'svg', 'iframe', 'input', 'button']
                  .forEach(tag => {
                    container.querySelectorAll(tag).forEach(el => el.remove());
                  });
            }
            """
        )

        log.info("Extracting textContent from main content container")
        main_text = page.evaluate(
            """
            () => {
                const el = document.querySelector('section.main-content.full');
                return el ? el.textContent : '';
            }
            """
        ).strip()

        if not main_text:
            raise ValueError("Main content container found but empty")

        parts.append("\n=== PROFILE CONTENT ===")
        parts.append(main_text)
        log.info("Main content extracted (%d chars)", len(main_text))

    except Exception as exc:
        raise ValueError(f"Failed extracting main content text: {exc}")

    combined = "\n\n".join(parts)
    log.info("Final extracted payload size: %d chars", len(combined))

    return combined[:1800]


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    TEST_URL = "https://247sports.com/player/darian-mensah-46116055/"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-gpu"],
        )
        page = browser.new_page()
        driver = PlaywrightDriver(page)

        try:
            contents = fetch_website_contents(driver, TEST_URL)

            print("==== EXTRACTED PAGE CONTENT ====")
            print(contents)
            print("\n==== CHARACTER COUNT ====")
            print(len(contents))

        finally:
            browser.close()
