from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


def fetch_website_contents(driver, url: str) -> str:
    """
    Return the title and visible text contents of the website at the given URL.
    Truncate to 2,000 characters.

    Expects a PlaywrightDriver-compatible interface.
    """
    driver.get(url)

    page = driver.page  # Access underlying Playwright Page

    try:
        page.wait_for_selector("body", timeout=10_000)
    except PlaywrightTimeoutError:
        return "No title found\n\n"

    title = page.title() or "No title found"

    # Remove irrelevant elements
    page.evaluate(
        """
        () => {
            ['script', 'style', 'img', 'input'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
        }
        """
    )

    text = page.inner_text("body").strip()

    combined = f"{title}\n\n{text}"

    return combined[:2_000]
