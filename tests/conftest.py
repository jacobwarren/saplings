"""
Pytest configuration file for Saplings tests.

This module provides configuration and fixtures for pytest.
"""

import pytest


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark tests (these can be slow and may hang)",
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "requires_vllm: mark test as requiring vLLM to be installed")
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
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test that may be slow or unstable"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip benchmark tests unless --run-benchmarks is specified
    if not config.getoption("--run-benchmarks"):
        skip_benchmarks = pytest.mark.skip(reason="Need --run-benchmarks option to run")
        for item in items:
            if (
                "benchmark" in item.keywords
                or "TestComparisonBenchmark" in str(item)
                or "tests/benchmarks/" in str(item.fspath)
            ):
                item.add_marker(skip_benchmarks)
