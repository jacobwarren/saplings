"""
Pytest configuration file for Saplings tests.

This module provides configuration and fixtures for pytest.
"""

import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "requires_vllm: mark test as requiring vLLM to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_openai: mark test as requiring OpenAI to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: mark test as requiring Anthropic to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_huggingface: mark test as requiring HuggingFace to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_langsmith: mark test as requiring LangSmith to be installed"
    )
