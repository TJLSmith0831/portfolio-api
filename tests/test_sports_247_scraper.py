import pytest
from unittest.mock import MagicMock, patch

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from app.scrapers.sports247_scraper import Sports247Scraper
from app.models.player_fit_models import PlayerSearchResult


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    driver.current_url = "https://247sports.com/player"
    driver.find_elements.return_value = []
    return driver


@pytest.fixture
def scraper(mock_driver):
    return Sports247Scraper(driver=mock_driver)


@patch("app.scrapers.sports247_scraper.WebDriverWait")
def test_direct_profile_redirect(mock_wait_cls, scraper, mock_driver):
    mock_wait = MagicMock()
    mock_wait_cls.return_value = mock_wait

    search_input = MagicMock()

    # First wait: search box
    # Second wait: redirect condition satisfied
    mock_wait.until.side_effect = [
        search_input,
        True,
    ]

    # Simulate redirect BEFORE scraper checks current_url
    mock_driver.current_url = "https://247sports.com/player/darian-mensah-46116055/"

    result = scraper.search_player_profile("Darian Mensah")

    assert result.found is True
    assert str(result.profile_url) == mock_driver.current_url
    assert result.displayed_name == "Darian Mensah"

    search_input.clear.assert_called_once()
    search_input.send_keys.assert_any_call("Darian Mensah")
    search_input.send_keys.assert_any_call(Keys.RETURN)


@patch("app.scrapers.sports247_scraper.WebDriverWait")
def test_no_search_results(mock_wait_cls, scraper, mock_driver):
    """
    Case 2:
    Search results page loads but no player rows exist
    """
    mock_wait = MagicMock()
    mock_wait_cls.return_value = mock_wait

    search_input = MagicMock()
    mock_wait.until.side_effect = [
        search_input,  # search box
        True,          # wait for results render
    ]

    mock_driver.find_elements.return_value = []

    result = scraper.search_player_profile("Nonexistent Player")

    assert result.found is False
    assert result.profile_url is None
    assert result.displayed_name is None


@patch("app.scrapers.sports247_scraper.WebDriverWait")
def test_results_list_selects_first_player(mock_wait_cls, scraper, mock_driver):
    mock_wait = MagicMock()
    mock_wait_cls.return_value = mock_wait

    search_input = MagicMock()
    mock_wait.until.side_effect = [
        search_input,
        True,
    ]

    header_li = MagicMock()
    header_li.get_attribute.return_value = "results_itm"

    player_li = MagicMock()
    player_li.get_attribute.return_value = ""

    link = MagicMock()
    link.get_attribute.return_value = "https://247sports.com/player/john-doe-999/"
    link.text = "John Doe"

    player_li.find_element.return_value = link
    mock_driver.find_elements.return_value = [header_li, player_li]

    result = scraper.search_player_profile("John Doe")

    assert result.found is True
    assert str(result.profile_url) == "https://247sports.com/player/john-doe-999/"
    assert result.displayed_name == "John Doe"

    mock_driver.get.assert_any_call("https://247sports.com/player/john-doe-999/")


@patch("app.scrapers.sports247_scraper.WebDriverWait")
def test_results_list_ignores_header_rows(mock_wait_cls, scraper, mock_driver):
    """
    Ensures rows with 'results_itm' class are ignored
    """
    mock_wait = MagicMock()
    mock_wait_cls.return_value = mock_wait

    search_input = MagicMock()
    mock_wait.until.side_effect = [
        search_input,
        True,
    ]

    header_only = MagicMock()
    header_only.get_attribute.return_value = "results_itm"

    mock_driver.find_elements.return_value = [header_only]

    result = scraper.search_player_profile("Some Player")

    assert result.found is False
