from __future__ import annotations

"""
Container configuration module for Agent.

This module provides functions to configure the dependency injection container
with the services needed by the Agent.
"""

import logging
from typing import Any

# Import the container directly from the internal module to avoid circular imports

logger = logging.getLogger(__name__)


def configure_container(config: Any) -> None:
    """
    Configure the container with the agent configuration.

    This function configures the container with the agent configuration,
    registering all necessary services.

    Args:
    ----
        config: The agent configuration

    """
    # Import here to avoid circular imports
    from saplings._internal.container_config import configure_services

    # Configure all services
    configure_services(config)

    logger.debug("Container configured with agent configuration")


__all__ = ["configure_container"]
