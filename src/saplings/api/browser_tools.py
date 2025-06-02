from __future__ import annotations

"""
Browser Tools API module for Saplings.

This module provides the public API for browser-related tools.
"""

from typing import Any, List

# Import the browser tools from the internal implementation
from saplings._internal.tools.browser_tools import (
    ClickTool as _ClickTool,
)
from saplings._internal.tools.browser_tools import (
    ClosePopupsTool as _ClosePopupsTool,
)
from saplings._internal.tools.browser_tools import (
    GetPageTextTool as _GetPageTextTool,
)
from saplings._internal.tools.browser_tools import (
    GoBackTool as _GoBackTool,
)
from saplings._internal.tools.browser_tools import (
    GoToTool as _GoToTool,
)
from saplings._internal.tools.browser_tools import (
    ScrollTool as _ScrollTool,
)
from saplings._internal.tools.browser_tools import (
    SearchTextTool as _SearchTextTool,
)
from saplings._internal.tools.browser_tools import (
    WaitTool as _WaitTool,
)
from saplings._internal.tools.browser_tools import (
    close_browser as _close_browser,
)
from saplings._internal.tools.browser_tools import (
    get_browser_tools as _get_browser_tools,
)
from saplings._internal.tools.browser_tools import (
    initialize_browser as _initialize_browser,
)
from saplings.api.stability import beta


@beta
class ClickTool:
    """
    Tool for clicking elements on a webpage.

    This tool allows an agent to click on elements on a webpage by providing
    a selector or text to click on.
    """

    def __init__(self):
        self._tool = _ClickTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class ClosePopupsTool:
    """
    Tool for closing popups on a webpage.

    This tool allows an agent to close common popups on a webpage.
    """

    def __init__(self):
        self._tool = _ClosePopupsTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class GetPageTextTool:
    """
    Tool for getting the text content of a webpage.

    This tool allows an agent to get the text content of the current webpage.
    """

    def __init__(self):
        self._tool = _GetPageTextTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class GoBackTool:
    """
    Tool for navigating back in the browser history.

    This tool allows an agent to navigate back to the previous page.
    """

    def __init__(self):
        self._tool = _GoBackTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class GoToTool:
    """
    Tool for navigating to a URL.

    This tool allows an agent to navigate to a specific URL.
    """

    def __init__(self):
        self._tool = _GoToTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class ScrollTool:
    """
    Tool for scrolling on a webpage.

    This tool allows an agent to scroll up, down, left, or right on a webpage.
    """

    def __init__(self):
        self._tool = _ScrollTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class SearchTextTool:
    """
    Tool for searching for text on a webpage.

    This tool allows an agent to search for text on the current webpage.
    """

    def __init__(self):
        self._tool = _SearchTextTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
class WaitTool:
    """
    Tool for waiting for a specified amount of time.

    This tool allows an agent to wait for a specified amount of time before
    continuing execution.
    """

    def __init__(self):
        self._tool = _WaitTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.parameters = self._tool.parameters

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)


@beta
def close_browser() -> None:
    """
    Close the browser.

    This function closes the browser that was opened by initialize_browser.
    """
    return _close_browser()


@beta
def get_browser_tools() -> List[Any]:
    """
    Get a list of browser tools.

    Returns
    -------
        List of browser tool instances

    """
    return _get_browser_tools()


@beta
def initialize_browser(headless: bool = False) -> None:
    """
    Initialize the browser.

    Args:
    ----
        headless: Whether to run the browser in headless mode

    """
    return _initialize_browser(headless=headless)


@beta
def save_screenshot(path: str) -> None:
    """
    Save a screenshot of the current webpage.

    Args:
    ----
        path: Path to save the screenshot to

    """
    # The internal implementation requires memory_step and agent parameters
    # but we're providing a simplified API here
    from saplings._internal.tools.browser_tools import get_browser_manager

    browser_manager = get_browser_manager()
    if browser_manager is None:
        return

    driver = browser_manager.get_driver()
    if driver is None:
        return

    # Take screenshot and save to file
    driver.save_screenshot(path)


__all__ = [
    # Tool classes
    "ClickTool",
    "ClosePopupsTool",
    "GetPageTextTool",
    "GoBackTool",
    "GoToTool",
    "ScrollTool",
    "SearchTextTool",
    "WaitTool",
    # Utility functions
    "close_browser",
    "get_browser_tools",
    "initialize_browser",
    "save_screenshot",
]
