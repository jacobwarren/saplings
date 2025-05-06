"""
Example of using dependency injection with constructor injection.

This example demonstrates how to use the new punq-based DI container
with constructor injection instead of singletons.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

# Import the DI container
from saplings.di import container, inject, register, reset_container

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1. Define service interfaces/protocols
class DataSource(Protocol):
    """Protocol for a data source."""

    def get_data(self):
        """Get data from the source."""
        ...

    def add_data(self, item: str) -> None:
        """Add data to the source."""
        ...


class DataProcessor(Protocol):
    """Protocol for a data processor."""

    def process(self, data: List[str]) -> List[str]:
        """Process data and return results."""
        ...


class Reporter(Protocol):
    """Protocol for a reporter."""

    def report(self, data: List[str]) -> None:
        """Report processed data."""
        ...


# 2. Implement the services with constructor injection
@register(DataSource)
class MemoryDataSource:
    """Memory-based data source."""

    def __init__(self, initial_data: Optional[List[str]] = None) -> None:
        """Initialize with optional initial data."""
        self._data = initial_data or []
        logger.info("MemoryDataSource initialized with %d items", len(self._data))

    def get_data(self):
        """Get all data items."""
        logger.info("Retrieving %d items from MemoryDataSource", len(self._data))
        return self._data.copy()

    def add_data(self, item: str) -> None:
        """Add an item to the data source."""
        self._data.append(item)
        logger.info("Added item to MemoryDataSource: %s", item)


@register(DataProcessor)
class SimpleProcessor:
    """Simple data processor that adds a prefix to each item."""

    def __init__(self, prefix: str = "Processed: ") -> None:
        """Initialize with a prefix."""
        self._prefix = prefix
        logger.info("SimpleProcessor initialized with prefix '%s'", prefix)

    def process(self, data: List[str]) -> List[str]:
        """Process the data by adding a prefix to each item."""
        logger.info("Processing %d items with SimpleProcessor", len(data))
        return [f"{self._prefix}{item}" for item in data]


@register(Reporter)
class ConsoleReporter:
    """Reporter that logs data to the console."""

    def report(self, data: List[str]) -> None:
        """Report data to the console."""
        logger.info("Reporting %d items with ConsoleReporter", len(data))
        for item in data:
            print(f"[REPORT] {item}")


# 3. Create a service that uses constructor injection
class DataManager:
    """Service that coordinates data sources, processors and reporters."""

    def __init__(self, data_source: DataSource, processor: DataProcessor, reporter: Reporter):
        """
        Initialize with required services.

        This is constructor injection - all dependencies are provided via the constructor.
        No global singletons are used.
        """
        self._data_source = data_source
        self._processor = processor
        self._reporter = reporter
        logger.info(
            "DataManager initialized with %s, %s, and %s",
            data_source.__class__.__name__,
            processor.__class__.__name__,
            reporter.__class__.__name__,
        )

    def run(self):
        """Run the data processing workflow."""
        # Get data from source
        raw_data = self._data_source.get_data()

        # Process the data
        processed_data = self._processor.process(raw_data)

        # Report the processed data
        self._reporter.report(processed_data)


# 4. Register the data manager with the container
container.register(
    DataManager,
    factory=lambda src, proc, rep: DataManager(data_source=src, processor=proc, reporter=rep),
    src=DataSource,
    proc=DataProcessor,
    rep=Reporter,
)


# 5. Example of using the @inject decorator for automatic dependency resolution
@inject
def run_with_injection(data_manager: DataManager, data_source: DataSource) -> None:
    """
    Run a demonstration with automatic dependency injection.

    The @inject decorator automatically resolves the dependencies from the container
    based on the type annotations.
    """
    logger.info("Adding data via run_with_injection function")
    data_source.add_data("Item from injected function")
    data_manager.run()


def main():
    """Run the example."""
    # Reset container for a clean example
    reset_container()

    # Configure container with initial data
    initial_data = ["Item 1", "Item 2", "Item 3"]
    container.register(DataSource, factory=lambda: MemoryDataSource(initial_data=initial_data))

    # Get services from the container - this way works in scripts
    logger.info("Getting services directly from container")
    data_source = container.resolve(DataSource)
    data_source.add_data("Item 4")

    # Get the data manager which has its dependencies injected automatically
    logger.info("Getting DataManager from container")
    data_manager = container.resolve(DataManager)
    data_manager.run()

    # Run the injected function demonstration
    logger.info("\nDemonstrating @inject decorator")
    run_with_injection()


if __name__ == "__main__":
    main()
