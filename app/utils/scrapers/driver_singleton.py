# app/scrapers/driver_singleton.py

from playwright.sync_api import sync_playwright
import threading

_driver = None
_playwright = None

driver_lock = threading.Lock()

def get_driver():
    global _driver, _playwright

    if _driver is None:
        _playwright = sync_playwright().start()
        _driver = _playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

    return _driver
