from __future__ import annotations

"""
Browser interaction tools for Saplings.

This module provides tools for browser interaction, including navigation,
clicking, scrolling, and taking screenshots.
"""


import logging
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any

import PIL.Image

# Type checking imports
if TYPE_CHECKING:
    from selenium import webdriver  # type: ignore[import-not-found]
    from selenium.common.exceptions import (  # type: ignore[import-not-found]
        ElementNotInteractableException,
        NoSuchElementException,
        TimeoutException,
    )
    from selenium.webdriver.chrome.options import (
        Options as ChromeOptions,  # type: ignore[import-not-found]
    )
    from selenium.webdriver.common.by import By  # type: ignore[import-not-found]
    from selenium.webdriver.common.keys import Keys  # type: ignore[import-not-found]
    from selenium.webdriver.support import (
        expected_conditions as EC,  # type: ignore[import-not-found]
    )
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import-not-found]

# Runtime imports
try:
    from selenium import webdriver  # type: ignore[import]
    from selenium.common.exceptions import (  # type: ignore[import]
        ElementNotInteractableException,
        NoSuchElementException,
        TimeoutException,
    )
    from selenium.webdriver.chrome.options import Options as ChromeOptions  # type: ignore[import]
    from selenium.webdriver.common.by import By  # type: ignore[import]
    from selenium.webdriver.common.keys import Keys  # type: ignore[import]
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import]
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import]

    SELENIUM_AVAILABLE = True
except ImportError:
    webdriver = None
    ElementNotInteractableException = None
    NoSuchElementException = None
    TimeoutException = None
    ChromeOptions = None
    By = None
    Keys = None
    EC = None
    WebDriverWait = None
    SELENIUM_AVAILABLE = False
    logging.warning(
        "Selenium not installed. Browser tools will not be available. "
        "Install selenium with: pip install saplings[browser]"
    )

from saplings.tools import Tool, register_tool


# Define a helper function to require selenium
def _require_selenium():
    """
    Ensure selenium is available and return the webdriver module.

    Raises
    ------
        ImportError: If selenium is not installed

    Returns
    -------
        The selenium webdriver module

    """
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium is optional – install with `pip install saplings[browser]`.")
    return webdriver


# Define ActionStep and Agent types for type checking
if TYPE_CHECKING:

    class ActionStep:
        """Type stub for ActionStep."""

        step_number: int
        observations: str | None = None
        observations_images: list | None = None

    class Agent:
        """Type stub for Agent."""

        memory: Any = None


class BrowserManager:
    """
    Manages browser instances for agent interactions.

    This class handles browser initialization, cleanup, and provides
    access to the browser driver.
    """

    _driver = None

    def __init__(self, headless: bool = False, window_size: tuple[int, int] = (1000, 1350)) -> None:
        """
        Initialize the browser manager.

        Args:
        ----
            headless: Whether to run the browser in headless mode
            window_size: The size of the browser window (width, height)

        """
        self.headless = headless
        self.window_size = window_size

        if not SELENIUM_AVAILABLE:
            logging.warning(
                "BrowserManager initialized but selenium is not installed. "
                "Browser tools will not be available."
            )

    def initialize_driver(self):
        """
        Initialize and return a WebDriver instance.

        Returns
        -------
            The WebDriver instance or None if selenium is not available

        """
        try:
            # Ensure selenium is available
            driver_module = _require_selenium()
        except ImportError as e:
            logging.warning(
                f"Cannot initialize browser driver: {e}. "
                "Install selenium with: pip install saplings[browser]"
            )
            return None

        if self._driver is not None:
            return self._driver

        # Get the required classes from selenium
        chrome_options_class = ChromeOptions
        if chrome_options_class is None:
            logging.warning("ChromeOptions is not available")
            return None

        chrome_options = chrome_options_class()
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")
        chrome_options.add_argument("--disable-pdf-viewer")
        chrome_options.add_argument("--window-position=0,0")

        if self.headless:
            chrome_options.add_argument("--headless")

        # Check if Chrome is available
        chrome_class = getattr(driver_module, "Chrome", None)
        if chrome_class is None:
            logging.warning("Chrome class is not available in the webdriver module")
            return None

        self._driver = chrome_class(options=chrome_options)
        return self._driver

    def get_driver(self):
        """
        Get the current WebDriver instance.

        Returns
        -------
            The WebDriver instance or None if not initialized or selenium is not available

        """
        if not SELENIUM_AVAILABLE:
            return None
        return self._driver

    def close(self):
        """Close the browser and clean up resources."""
        if not SELENIUM_AVAILABLE:
            return

        if self._driver is not None:
            self._driver.quit()
            self._driver = None


def get_browser_manager():
    """
    Get the browser manager.

    This function is maintained for backward compatibility.
    New code should use constructor injection via the DI container.

    Returns
    -------
        BrowserManager: The browser manager from the DI container or None if not available

    """
    if not SELENIUM_AVAILABLE:
        logging.warning("Cannot get browser manager because selenium is not installed.")
        return None

    try:
        from saplings.di import container

        return container.resolve(BrowserManager)
    except Exception as e:
        logging.warning(f"Error resolving BrowserManager from container: {e}")
        return None


def save_screenshot(memory_step: ActionStep, agent: Agent) -> None:
    """
    Save a screenshot of the current browser state to the memory step.

    Args:
    ----
        memory_step: The memory step to save the screenshot to
        agent: The agent instance

    """
    if not SELENIUM_AVAILABLE:
        logging.warning("Cannot save screenshot because selenium is not installed.")
        return

    # Let JavaScript animations happen before taking the screenshot
    time.sleep(1.0)

    browser_manager = get_browser_manager()
    if browser_manager is None:
        return

    driver = browser_manager.get_driver()
    if driver is None:
        return

    current_step = memory_step.step_number

    # Remove previous screenshots from logs for lean processing
    if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
        for previous_memory_step in agent.memory.steps:
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 2
            ):
                previous_memory_step.observations_images = None

    # Take screenshot
    png_bytes = driver.get_screenshot_as_png()
    image = PIL.Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )


class BrowserTool(Tool):
    """Base class for browser interaction tools."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._browser_manager = get_browser_manager()
        if not SELENIUM_AVAILABLE:
            logging.warning("BrowserTool initialized but selenium is not installed.")

    @property
    def driver(self):
        """Get the WebDriver instance, initializing if necessary."""
        if not SELENIUM_AVAILABLE:
            logging.warning("Cannot access browser driver because selenium is not installed.")
            return None
        if self._browser_manager is None:
            return None
        return self._browser_manager.initialize_driver()


@register_tool(name="go_to")
class GoToTool(BrowserTool):
    """Tool for navigating to a URL."""

    name = "go_to"
    description = "Navigate to a URL"
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = {
            "url": {"type": "string", "description": "The URL to navigate to", "required": True}
        }

    def forward(self, url: str) -> str:
        """
        Navigate to the specified URL.

        Args:
        ----
            url: The URL to navigate to

        Returns:
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot navigate: browser driver is not available"

        # Add http:// prefix if not present
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        self.driver.get(url)
        return f"Navigated to {url}"


@register_tool(name="click")
class ClickTool(BrowserTool):
    """Tool for clicking elements on a page."""

    name = "click"
    description = "Click an element on the page"
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = {
            "selector": {
                "type": "string",
                "description": "The text or CSS selector of the element to click",
                "required": True,
            },
            "by_text": {
                "type": "boolean",
                "description": "Whether to find the element by text (True) or CSS selector (False)",
                "required": False,
            },
            "wait_time": {
                "type": "integer",
                "description": "Time to wait for the element to be clickable (in seconds)",
                "required": False,
            },
        }

    def forward(self, selector: str, by_text: bool = True, wait_time: int = 10) -> str:
        """
        Click an element on the page.

        Args:
        ----
            selector: The text or CSS selector of the element to click
            by_text: Whether to find the element by text (True) or CSS selector (False)
            wait_time: Time to wait for the element to be clickable (in seconds)

        Returns:
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot click: browser driver is not available"

        try:
            # Ensure selenium is available
            _require_selenium()

            # Check if required classes are available
            if WebDriverWait is None or EC is None or By is None:
                return "Cannot click: selenium components are not available"

            if by_text:
                # Try to find element by text
                xpath = f"//*[contains(text(), '{selector}')]"
                element = WebDriverWait(self.driver, wait_time).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
            else:
                # Find element by CSS selector
                element = WebDriverWait(self.driver, wait_time).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )

            # Scroll element into view
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(0.5)  # Give time for scroll to complete

            # Click the element
            element.click()
            return f"Clicked element: {selector}"

        except ImportError:
            return "Cannot click: selenium is not installed"
        except Exception as e:
            # Handle specific exceptions if they're available
            exception_name = type(e).__name__
            if exception_name == "TimeoutException":
                return f"Timeout waiting for element: {selector}"
            if exception_name == "NoSuchElementException":
                return f"Element not found: {selector}"
            if exception_name == "ElementNotInteractableException":
                return f"Element not interactable: {selector}"
            return f"Error clicking element: {e}"


@register_tool(name="scroll")
class ScrollTool(BrowserTool):
    """Tool for scrolling the page."""

    name = "scroll"
    description = "Scroll the page up or down"
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = {
            "direction": {
                "type": "string",
                "description": "The direction to scroll ('up' or 'down')",
                "required": True,
            },
            "pixels": {
                "type": "integer",
                "description": "The number of pixels to scroll",
                "required": False,
            },
        }

    def forward(self, direction: str, pixels: int = 500) -> str:
        """
        Scroll the page up or down.

        Args:
        ----
            direction: The direction to scroll ('up' or 'down')
            pixels: The number of pixels to scroll

        Returns:
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot scroll: browser driver is not available"

        if direction.lower() not in ["up", "down"]:
            return "Invalid direction. Use 'up' or 'down'."

        try:
            # Ensure selenium is available
            _require_selenium()

            scroll_value = pixels if direction.lower() == "down" else -pixels
            self.driver.execute_script(f"window.scrollBy(0, {scroll_value});")

            return f"Scrolled {direction} by {pixels} pixels"
        except ImportError:
            return "Cannot scroll: selenium is not installed"
        except Exception as e:
            return f"Error scrolling: {e}"


@register_tool(name="search_text")
class SearchTextTool(BrowserTool):
    """Tool for searching text on a page."""

    name = "search_text"
    description = "Search for text on the current page"
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = {
            "text": {"type": "string", "description": "The text to search for", "required": True},
            "nth_result": {
                "type": "integer",
                "description": "Which occurrence to focus on (1-based index)",
                "required": False,
            },
        }

    def forward(self, text: str, nth_result: int = 1) -> str:
        """
        Search for text on the current page and focus on the nth occurrence.

        Args:
        ----
            text: The text to search for
            nth_result: Which occurrence to focus on (1-based index)

        Returns:
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot search text: browser driver is not available"

        try:
            # Ensure selenium is available
            _require_selenium()

            # Check if required classes are available
            if By is None:
                return "Cannot search text: selenium components are not available"

            elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")

            if not elements:
                return f"No matches found for '{text}'"

            if nth_result > len(elements):
                return f"Match n°{nth_result} not found (only {len(elements)} matches found)"

            result = f"Found {len(elements)} matches for '{text}'."
            elem = elements[nth_result - 1]
            self.driver.execute_script("arguments[0].scrollIntoView(true);", elem)

            # Highlight the element temporarily
            original_style = self.driver.execute_script(
                "return arguments[0].getAttribute('style');", elem
            )
            self.driver.execute_script(
                "arguments[0].setAttribute('style', arguments[1]);",
                elem,
                "background-color: yellow; border: 2px solid red;",
            )
            time.sleep(0.5)
            self.driver.execute_script(
                "arguments[0].setAttribute('style', arguments[1]);", elem, original_style or ""
            )

            result += f" Focused on element {nth_result} of {len(elements)}"
            return result

        except ImportError:
            return "Cannot search text: selenium is not installed"
        except Exception as e:
            return f"Error searching for text: {e}"


@register_tool(name="close_popups")
class ClosePopupsTool(BrowserTool):
    """Tool for closing popups."""

    name = "close_popups"
    description = "Close any visible modal or popup on the page"
    output_type = "string"

    def forward(self):
        """
        Close any visible modal or popup on the page.

        Returns
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot close popups: browser driver is not available"

        try:
            # Ensure selenium is available
            driver_module = _require_selenium()

            # Check if required classes are available
            if By is None or Keys is None:
                return "Cannot close popups: selenium components are not available"

            # Try pressing Escape key
            # Check if ActionChains is available
            try:
                action_chains_class = getattr(driver_module, "ActionChains", None)
                if action_chains_class is not None and Keys is not None and hasattr(Keys, "ESCAPE"):
                    action_chains_class(self.driver).send_keys(Keys.ESCAPE).perform()
            except Exception:
                # If ActionChains or Keys.ESCAPE is not available or fails, skip this step
                pass

            # Try clicking common close buttons
            close_selectors = [
                "//button[contains(@class, 'close')]",
                "//div[contains(@class, 'close')]",
                "//button[contains(text(), 'Close')]",
                "//button[contains(text(), 'Accept')]",
                "//button[contains(text(), 'I accept')]",
                "//button[contains(text(), 'Agree')]",
                "//button[contains(text(), 'Got it')]",
                "//button[contains(text(), 'OK')]",
                "//button[contains(text(), 'No thanks')]",
                "//button[contains(text(), 'Not now')]",
                "//button[contains(@aria-label, 'Close')]",
                "//div[contains(@aria-label, 'Close')]",
                "//span[contains(@aria-label, 'Close')]",
                "//button[contains(@title, 'Close')]",
                "//div[contains(@role, 'dialog')]//button",
                "//div[contains(@class, 'modal')]//button",
                "//div[contains(@class, 'popup')]//button",
                "//div[contains(@class, 'cookie')]//button",
            ]

            for selector in close_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        for element in elements:
                            try:
                                if element.is_displayed():
                                    element.click()
                                    time.sleep(0.5)  # Wait for popup to close
                                    return "Closed popup by clicking close button"
                            except Exception:
                                continue
                except Exception:
                    continue

            return "Attempted to close popups"

        except ImportError:
            return "Cannot close popups: selenium is not installed"
        except Exception as e:
            return f"Error closing popups: {e}"


@register_tool(name="go_back")
class GoBackTool(BrowserTool):
    """Tool for navigating back to the previous page."""

    name = "go_back"
    description = "Go back to the previous page"
    output_type = "string"

    def forward(self):
        """
        Go back to the previous page.

        Returns
        -------
            A message indicating the result

        """
        if self.driver is None:
            return "Cannot go back: browser driver is not available"

        try:
            # Ensure selenium is available
            _require_selenium()

            self.driver.back()
            return "Navigated back to previous page"
        except ImportError:
            return "Cannot go back: selenium is not installed"
        except Exception as e:
            return f"Error navigating back: {e}"


@register_tool(name="get_page_text")
class GetPageTextTool(BrowserTool):
    """Tool for getting the text content of the current page."""

    name = "get_page_text"
    description = "Get the text content of the current page"
    output_type = "string"

    def forward(self):
        """
        Get the text content of the current page.

        Returns
        -------
            The text content of the page

        """
        if self.driver is None:
            return "Cannot get page text: browser driver is not available"

        try:
            # Ensure selenium is available
            _require_selenium()

            # Check if required classes are available
            if By is None:
                return "Cannot get page text: selenium components are not available"

            return self.driver.find_element(By.TAG_NAME, "body").text
        except ImportError:
            return "Cannot get page text: selenium is not installed"
        except Exception as e:
            return f"Error getting page text: {e}"


@register_tool(name="wait")
class WaitTool(BrowserTool):
    """Tool for waiting a specified amount of time."""

    name = "wait"
    description = "Wait for a specified number of seconds"
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = {
            "seconds": {
                "type": "number",
                "description": "The number of seconds to wait",
                "required": True,
            }
        }

    def forward(self, seconds: float) -> str:
        """
        Wait for a specified number of seconds.

        Args:
        ----
            seconds: The number of seconds to wait

        Returns:
        -------
            A message indicating the result

        """
        # Limit maximum wait time to prevent excessive waiting
        seconds = min(seconds, 30.0)
        time.sleep(seconds)
        return f"Waited for {seconds} seconds"


def get_browser_tools():
    """
    Get all browser interaction tools.

    Returns
    -------
        A list of browser interaction tools

    """
    return [
        GoToTool(),
        ClickTool(),
        ScrollTool(),
        SearchTextTool(),
        ClosePopupsTool(),
        GoBackTool(),
        GetPageTextTool(),
        WaitTool(),
    ]


def initialize_browser(headless: bool = False, window_size: tuple[int, int] = (1000, 1350)) -> None:
    """
    Initialize the browser for agent interactions.

    Args:
    ----
        headless: Whether to run the browser in headless mode
        window_size: The size of the browser window (width, height)

    """
    if not SELENIUM_AVAILABLE:
        logging.warning("Cannot initialize browser because selenium is not installed.")
        return

    try:
        from saplings.di import container

        container.register(
            BrowserManager,
            factory=lambda: BrowserManager(headless=headless, window_size=window_size),
            singleton=True,
        )
        browser_manager = container.resolve(BrowserManager)
        if browser_manager is not None:
            browser_manager.initialize_driver()
    except Exception as e:
        logging.warning(f"Error initializing browser: {e}")


def close_browser():
    """Close the browser and clean up resources."""
    if not SELENIUM_AVAILABLE:
        return

    browser_manager = get_browser_manager()
    if browser_manager is not None:
        browser_manager.close()
