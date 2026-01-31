import pytest
from unittest.mock import MagicMock

from app.utils.scrapers.sports247_scraper import Sports247Scraper, PlaywrightElement
from app.models.player_fit_models import PlayerSearchResult


@pytest.fixture
def mock_page():
    """
    Mock Playwright Page object.
    """
    page = MagicMock()
    page.url = "https://247sports.com/player"
    return page


@pytest.fixture
def mock_driver(mock_page):
    """
    Mock PlaywrightDriver with a mocked Page.
    """
    driver = MagicMock()
    driver.page = mock_page
    driver.current_url = "https://247sports.com/player"
    return driver


@pytest.fixture
def scraper(mock_driver):
    """
    Sports247Scraper instance using mocked PlaywrightDriver.
    """
    return Sports247Scraper(driver=mock_driver)


def test_no_search_results(scraper, mock_driver, mock_page):
    """
    Case:
    - Search input exists
    - Results container renders
    - No player rows are found
    """

    input_handle = MagicMock()
    mock_page.wait_for_selector.side_effect = [
        input_handle,  # input#FullName
        None,          # results selector
    ]

    mock_driver.find_elements.return_value = []

    result = scraper.search_player_profile("Nonexistent Player")

    assert isinstance(result, PlayerSearchResult)
    assert result.found is False
    assert result.profile_url is None
    assert result.displayed_name is None


def test_results_list_selects_first_player(scraper, mock_driver, mock_page):
    """
    Case:
    - Multiple <li> rows
    - Only one contains a player link
    - First valid player is selected
    """

    # Mock search input
    input_handle = MagicMock()
    mock_page.wait_for_selector.side_effect = [
        input_handle,  # input#FullName
        MagicMock(),   # results selector
    ]

    # Header row (no player link)
    header_el = MagicMock()
    header_el.query_selector.return_value = None

    header_li = PlaywrightElement(header_el)

    # Player row
    player_el = MagicMock()
    player_el.query_selector.return_value = MagicMock()

    link_el = MagicMock()
    link_el.get_attribute.return_value = "https://247sports.com/player/john-doe-999/"
    link_el.inner_text.return_value = "John Doe"

    player_el.query_selector.return_value = link_el

    player_li = PlaywrightElement(player_el)

    mock_driver.find_elements.return_value = [
        header_li,
        player_li,
    ]

    result = scraper.search_player_profile("John Doe")

    assert result.found is True
    assert str(result.profile_url) == "https://247sports.com/player/john-doe-999/"
    assert result.displayed_name == "John Doe"

    mock_driver.get.assert_any_call("https://247sports.com/player/john-doe-999/")


def test_results_list_ignores_non_player_rows(scraper, mock_driver, mock_page):
    """
    Case:
    - Results render
    - Rows exist but none contain player links
    """

    input_handle = MagicMock()
    mock_page.wait_for_selector.side_effect = [
        input_handle,  # input#FullName
        MagicMock(),   # results selector
    ]

    non_player_el = MagicMock()
    non_player_el.query_selector.return_value = None

    non_player_li = PlaywrightElement(non_player_el)

    mock_driver.find_elements.return_value = [non_player_li]

    result = scraper.search_player_profile("Some Player")

    assert result.found is False
    assert result.profile_url is None
    assert result.displayed_name is None
