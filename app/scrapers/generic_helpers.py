from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def fetch_website_contents(driver, url: str) -> str:
    """
    Return the title and visible text contents of the website at the given URL.
    Truncate to 2,000 characters.
    """
    driver.get(url)

    # Wait until the page body is present
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    title = driver.title or "No title found"

    # Remove irrelevant elements (script/style/img/input)
    driver.execute_script("""
        ['script', 'style', 'img', 'input'].forEach(tag => {
            document.querySelectorAll(tag).forEach(el => el.remove());
        });
    """)

    body = driver.find_element(By.TAG_NAME, "body")
    text = body.text.strip() if body else ""

    combined = f"{title}\n\n{text}"

    return combined[:2_000]


def fetch_website_links(driver, url: str) -> list[str]:
    """
    Return the links on the website at the given URL using Selenium.

    :param url: The URL to fetch links from.
    :return: A list of non-empty href links found on the page.
    """

    try:
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
        )

        links = [
            element.get_attribute("href")
            for element in driver.find_elements(By.TAG_NAME, "a")
        ]

        return [link for link in links if link]

    finally:
        driver.quit()


if __name__ == "__main__":

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)

    print(fetch_website_links(driver, "https://247sports.com/season/2026-football/transferportal/"))
    print(fetch_website_contents(driver, "https://247sports.com/player/darian-mensah-46116055/"))
