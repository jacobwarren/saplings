from __future__ import annotations

"""
Browser tools for Saplings.

This module provides tools for interacting with web browsers.
"""

import logging

from saplings._internal.tools.base import Tool

logger = logging.getLogger(__name__)

# Check if browser tools are available
try:
    import playwright  # noqa: F401

    BROWSER_TOOLS_AVAILABLE = True
except ImportError:
    BROWSER_TOOLS_AVAILABLE = False
    logger.warning(
        "Browser tools not available. Please install the playwright library: pip install saplings[browser]"
    )


def is_browser_tools_available() -> bool:
    """
    Check if browser tools are available.

    Returns
    -------
        True if browser tools are available, False otherwise

    """
    return BROWSER_TOOLS_AVAILABLE


# Global browser instance
_browser = None
_page = None


class BrowserManager:
    """
    Manages browser instances for agent interactions.

    This class handles browser initialization, cleanup, and provides
    access to the browser driver.
    """

    def __init__(self, headless: bool = True) -> None:
        """
        Initialize the browser manager.

        Args:
        ----
            headless: Whether to run the browser in headless mode

        """
        self.headless = headless
        self._browser = None
        self._page = None

    def initialize(self) -> None:
        """Initialize the browser."""
        global _browser, _page

        if not BROWSER_TOOLS_AVAILABLE:
            msg = "Browser tools not available. Please install the playwright library: pip install saplings[browser]"
            raise ImportError(msg)

        if self._browser is not None:
            return

        try:
            from playwright.sync_api import sync_playwright

            playwright_instance = sync_playwright().start()
            self._browser = playwright_instance.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()

            # Update global variables for compatibility with existing tools
            _browser = self._browser
            _page = self._page

            logger.info("Browser initialized")
        except Exception as e:
            logger.exception(f"Failed to initialize browser: {e}")
            msg = f"Failed to initialize browser: {e}"
            raise RuntimeError(msg)

    def close(self) -> None:
        """Close the browser."""
        global _browser, _page

        if self._browser is None:
            return

        try:
            self._page.close()
            self._browser.close()
            self._browser = None
            self._page = None

            # Update global variables for compatibility
            _browser = None
            _page = None

            logger.info("Browser closed")
        except Exception as e:
            logger.exception(f"Failed to close browser: {e}")

    def get_page(self):
        """Get the current page."""
        if self._page is None:
            self.initialize()
        return self._page


def initialize_browser(headless: bool = True) -> None:
    """
    Initialize the browser.

    Args:
    ----
        headless: Whether to run the browser in headless mode

    """
    global _browser, _page

    if not BROWSER_TOOLS_AVAILABLE:
        msg = "Browser tools not available. Please install the playwright library: pip install saplings[browser]"
        raise ImportError(msg)

    if _browser is not None:
        return

    try:
        from playwright.sync_api import sync_playwright

        playwright_instance = sync_playwright().start()
        _browser = playwright_instance.chromium.launch(headless=headless)
        _page = _browser.new_page()
        logger.info("Browser initialized")
    except Exception as e:
        logger.exception(f"Failed to initialize browser: {e}")
        msg = f"Failed to initialize browser: {e}"
        raise RuntimeError(msg)


def close_browser() -> None:
    """Close the browser."""
    global _browser, _page

    if _browser is None:
        return

    try:
        _page.close()
        _browser.close()
        _browser = None
        _page = None
        logger.info("Browser closed")
    except Exception as e:
        logger.exception(f"Failed to close browser: {e}")


def save_screenshot(path: str) -> None:
    """
    Save a screenshot of the current page.

    Args:
    ----
        path: Path to save the screenshot to

    """
    global _page

    if _page is None:
        msg = "Browser not initialized. Call initialize_browser() first."
        raise RuntimeError(msg)

    try:
        _page.screenshot(path=path)
        logger.info(f"Screenshot saved to {path}")
    except Exception as e:
        logger.exception(f"Failed to save screenshot: {e}")
        msg = f"Failed to save screenshot: {e}"
        raise RuntimeError(msg)


def get_browser_tools() -> list[Tool]:
    """
    Get all browser tools.

    Returns
    -------
        List of browser tools

    """
    return [
        GoToTool(),
        GetPageTextTool(),
        ClickTool(),
        SearchTextTool(),
        ScrollTool(),
        WaitTool(),
        GoBackTool(),
        ClosePopupsTool(),
    ]


class GoToTool(Tool):
    """Tool for navigating to a URL."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "go_to"
        self.description = "Navigate to a URL"
        self.parameters = {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
                "required": True,
            }
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, url: str) -> str:
        """
        Navigate to a URL.

        Args:
        ----
            url: URL to navigate to

        Returns:
        -------
            Status message

        """
        global _page

        if _page is None:
            initialize_browser()

        try:
            _page.goto(url)
            return f"Navigated to {url}"
        except Exception as e:
            logger.exception(f"Failed to navigate to {url}: {e}")
            return f"Failed to navigate to {url}: {e}"


class GetPageTextTool(Tool):
    """Tool for getting the text content of the current page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "get_page_text"
        self.description = "Get the text content of the current page"
        self.parameters = {}
        self.output_type = "string"
        self.is_initialized = True

    def forward(self) -> str:
        """
        Get the text content of the current page.

        Returns
        -------
            Text content of the page

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            return _page.content()
        except Exception as e:
            logger.exception(f"Failed to get page text: {e}")
            return f"Failed to get page text: {e}"


class ClickTool(Tool):
    """Tool for clicking on an element on the page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "click"
        self.description = "Click on an element on the page"
        self.parameters = {
            "selector": {
                "type": "string",
                "description": "CSS selector for the element to click",
                "required": True,
            }
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, selector: str) -> str:
        """
        Click on an element on the page.

        Args:
        ----
            selector: CSS selector for the element to click

        Returns:
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            _page.click(selector)
            return f"Clicked on element with selector: {selector}"
        except Exception as e:
            logger.exception(f"Failed to click on element with selector {selector}: {e}")
            return f"Failed to click on element with selector {selector}: {e}"


class SearchTextTool(Tool):
    """Tool for searching for text on the page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "search_text"
        self.description = "Search for text on the page"
        self.parameters = {
            "text": {
                "type": "string",
                "description": "Text to search for",
                "required": True,
            }
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, text: str) -> str:
        """
        Search for text on the page.

        Args:
        ----
            text: Text to search for

        Returns:
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            content = _page.content()
            if text in content:
                return f"Found text: {text}"
            return f"Text not found: {text}"
        except Exception as e:
            logger.exception(f"Failed to search for text {text}: {e}")
            return f"Failed to search for text {text}: {e}"


class ScrollTool(Tool):
    """Tool for scrolling the page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "scroll"
        self.description = "Scroll the page"
        self.parameters = {
            "direction": {
                "type": "string",
                "description": "Direction to scroll (up, down, left, right)",
                "required": True,
            },
            "amount": {
                "type": "integer",
                "description": "Amount to scroll in pixels",
                "required": False,
            },
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, direction: str, amount: int = 500) -> str:
        """
        Scroll the page.

        Args:
        ----
            direction: Direction to scroll (up, down, left, right)
            amount: Amount to scroll in pixels

        Returns:
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            if direction == "up":
                _page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "down":
                _page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "left":
                _page.evaluate(f"window.scrollBy(-{amount}, 0)")
            elif direction == "right":
                _page.evaluate(f"window.scrollBy({amount}, 0)")
            else:
                return f"Invalid direction: {direction}. Must be one of: up, down, left, right"

            return f"Scrolled {direction} by {amount} pixels"
        except Exception as e:
            logger.exception(f"Failed to scroll {direction} by {amount} pixels: {e}")
            return f"Failed to scroll {direction} by {amount} pixels: {e}"


class WaitTool(Tool):
    """Tool for waiting for a specified amount of time."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "wait"
        self.description = "Wait for a specified amount of time"
        self.parameters = {
            "seconds": {
                "type": "number",
                "description": "Number of seconds to wait",
                "required": True,
            }
        }
        self.output_type = "string"
        self.is_initialized = True

    def forward(self, seconds: float) -> str:
        """
        Wait for a specified amount of time.

        Args:
        ----
            seconds: Number of seconds to wait

        Returns:
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            _page.wait_for_timeout(seconds * 1000)
            return f"Waited for {seconds} seconds"
        except Exception as e:
            logger.exception(f"Failed to wait for {seconds} seconds: {e}")
            return f"Failed to wait for {seconds} seconds: {e}"


class GoBackTool(Tool):
    """Tool for navigating back to the previous page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "go_back"
        self.description = "Navigate back to the previous page"
        self.parameters = {}
        self.output_type = "string"
        self.is_initialized = True

    def forward(self) -> str:
        """
        Navigate back to the previous page.

        Returns
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            _page.go_back()
            return "Navigated back to the previous page"
        except Exception as e:
            logger.exception(f"Failed to navigate back: {e}")
            return f"Failed to navigate back: {e}"


class ClosePopupsTool(Tool):
    """Tool for closing popups on the page."""

    def __init__(self) -> None:
        """Initialize the tool."""
        self.name = "close_popups"
        self.description = "Close popups on the page"
        self.parameters = {}
        self.output_type = "string"
        self.is_initialized = True

    def forward(self) -> str:
        """
        Close popups on the page.

        Returns
        -------
            Status message

        """
        global _page

        if _page is None:
            msg = "Browser not initialized. Call initialize_browser() first."
            raise RuntimeError(msg)

        try:
            # Try to close common popup elements
            selectors = [
                "button[aria-label='Close']",
                "button.close",
                "div.modal button",
                "div.popup button",
                "div.modal-close",
                "div.popup-close",
                "button.modal-close",
                "button.popup-close",
                "button.dismiss",
                "button.reject",
                "button.accept",
                "button.agree",
                "button.consent",
                "button.cookie-accept",
                "button.cookie-dismiss",
                "button.cookie-close",
            ]

            closed = 0
            for selector in selectors:
                try:
                    elements = _page.query_selector_all(selector)
                    for element in elements:
                        element.click()
                        closed += 1
                except Exception:
                    # Ignore errors for individual selectors
                    pass

            if closed > 0:
                return f"Closed {closed} popups"
            return "No popups found to close"
        except Exception as e:
            logger.exception(f"Failed to close popups: {e}")
            return f"Failed to close popups: {e}"
