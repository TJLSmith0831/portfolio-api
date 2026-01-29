
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pydantic import HttpUrl

from app.models.player_fit_models import PlayerSearchResult
from app.scrapers.generic_helpers import fetch_website_contents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

class Sports247Scraper:
    """
    Responsible for:
    - Resolving player identity via 247Sports
    - Extracting authoritative player profile content
    - Producing grounded inputs for downstream analysis

    This class does NOT:
    - Perform football analysis
    - Infer missing facts
    - Call LLMs directly
    """

    def __init__(self, driver, logger=None):
        self.driver = driver
        self.logger = logger

    def search_player_profile(self, player_name: str) -> PlayerSearchResult:
        """
        Searches for a player profile on 247Sports.com.

        :param player_name: The name of the player to search for.
        :return: PlayerSearchResult containing search outcome and profile URL if found.
        """
        log.info("Searching 247Sports for player profile: %s", player_name)
        entry_url = "https://247sports.com/player"
        self.driver.get(entry_url)
        log.info("Loaded 247Sports Player Search page")

        wait = WebDriverWait(self.driver, 10)

        # Wait for React player search input
        search_box = wait.until(
            EC.presence_of_element_located((By.ID, "FullName"))
        )
        log.info("Player search input located")

        search_box.clear()
        search_box.send_keys(player_name)
        search_box.send_keys(Keys.RETURN)
        log.info("Submitted player search for: %s", player_name)

        # Wait for either redirect or results render
        wait.until(lambda d: d.current_url != entry_url or d.find_elements(By.CSS_SELECTOR, ".content-list"))

        current_url = self.driver.current_url
        log.info("Post-search URL: %s", current_url)

        # Case 1: Direct redirect to player profile
        if "/player/" in current_url and "FullName=" not in current_url:
            log.info("Direct profile match detected")

            return PlayerSearchResult(
                query=player_name,
                found=True,
                profile_url=current_url,
                displayed_name=player_name,
            )

        # Case 2: Results list
        results = self.driver.find_elements(By.CSS_SELECTOR, ".content-list > li")

        # Filter out the header / results summary row
        player_results = [
            li for li in results
            if "results_itm" not in (li.get_attribute("class") or "")
        ]

        if len(player_results) == 0:
            log.info("No player results found for query: %s", player_name)
            return PlayerSearchResult(
                query=player_name,
                found=False,
            )

        # Deterministically take the first actual player result
        first_result = player_results[0]

        link = first_result.find_element(By.CSS_SELECTOR, "a.name")
        profile_url = link.get_attribute("href")
        displayed_name = link.text.strip()

        log.info(
            "Player search result selected: displayed_name='%s', profile_url='%s'",
            displayed_name,
            profile_url,
        )

        # Navigate to profile page
        self.driver.get(profile_url)
        log.info("Navigated to player profile page")

        return PlayerSearchResult(
            query=player_name,
            found=True,
            profile_url=profile_url,
            displayed_name=displayed_name,
        )

if __name__ == "__main__":

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    scraper = Sports247Scraper(driver)
    results = scraper.search_player_profile("Darian Mensah")
    print(results)
    profile = fetch_website_contents(scraper.driver, str(results.profile_url))
    print(profile)
