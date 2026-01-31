from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
import json
import logging

log = logging.getLogger(__name__)


def fetch_website_contents(driver, url: str) -> str:
    """
    Extract high-signal, football-relevant content from a 247Sports player page.

    Strategy:
    1. Load page and wait for hydration
    2. Parse application/ld+json for canonical identity data
    3. Extract text from section.main-content.full only
    4. Return a compact, LLM-ready text payload

    Expects a PlaywrightDriver-compatible interface.
    """
    log.info("Fetching website contents: %s", url)

    driver.get(url)
    page = driver.page

    try:
        log.info("Waiting for page network idle")
        page.wait_for_load_state("networkidle", timeout=15_000)

        log.info("Waiting for main content container")
        page.wait_for_selector("section.main-content.full", timeout=15_000)
    except PlaywrightTimeoutError:
        log.warning("Timed out waiting for main content container")
        return "No usable content found."

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
        log.warning("Failed extracting structured data: %s", exc)

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

        if main_text:
            parts.append("\n=== PROFILE CONTENT ===")
            parts.append(main_text)
            log.info("Main content extracted (%d chars)", len(main_text))
        else:
            log.warning("Main content container found but empty")

    except Exception as exc:
        log.warning("Failed extracting main content text: %s", exc)

    combined = "\n\n".join(parts)
    log.info("Final extracted payload size: %d chars", len(combined))

    return combined[:2_500]


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    from app.utils.scrapers.sports247_scraper import PlaywrightDriver

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
