"""
Thread-safe container state management without global variables.

This module provides a ContainerState class that manages container configuration
state without relying on global variables, eliminating race conditions and
providing context-based configuration isolation.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when container configuration fails."""


class ContainerState:
    """Thread-safe container state management without global variables."""

    def __init__(self):
        """Initialize the container state manager."""
        self._configurations: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._initialized_contexts: Set[str] = set()
        logger.debug("ContainerState initialized")

    @contextmanager
    def configuration_context(self, context_id: str = "default"):
        """
        Create an isolated configuration context.

        Args:
        ----
            context_id: Unique identifier for the configuration context

        Yields:
        ------
            bool: True if this is the first time configuration for this context,
                  False if the context has already been configured

        """
        with self._lock:
            if context_id not in self._initialized_contexts:
                self._initialized_contexts.add(context_id)
                logger.debug(f"First configuration for context: {context_id}")
                yield True  # First time configuration
            else:
                logger.debug(f"Reconfiguration for context: {context_id}")
                yield False  # Already configured

    def is_context_initialized(self, context_id: str = "default") -> bool:
        """
        Check if a context has been initialized.

        Args:
        ----
            context_id: Context identifier to check

        Returns:
        -------
            bool: True if the context has been initialized

        """
        with self._lock:
            return context_id in self._initialized_contexts

    def get_configuration(self, context_id: str = "default") -> Optional[Any]:
        """
        Get the configuration for a specific context.

        Args:
        ----
            context_id: Context identifier

        Returns:
        -------
            The configuration object for the context, or None if not found

        """
        with self._lock:
            return self._configurations.get(context_id)

    def set_configuration(self, config: Any, context_id: str = "default") -> None:
        """
        Set the configuration for a specific context.

        Args:
        ----
            config: Configuration object to store
            context_id: Context identifier

        """
        with self._lock:
            self._configurations[context_id] = config
            logger.debug(f"Configuration set for context: {context_id}")

    def reset_context(self, context_id: str = "default") -> None:
        """
        Reset a specific configuration context.

        Args:
        ----
            context_id: Context identifier to reset

        """
        with self._lock:
            # Remove from initialized contexts
            self._initialized_contexts.discard(context_id)

            # Clear any cached configurations
            self._configurations.pop(context_id, None)

            logger.debug(f"Context reset: {context_id}")

    def reset_all_contexts(self) -> None:
        """Reset all configuration contexts."""
        with self._lock:
            self._initialized_contexts.clear()
            self._configurations.clear()
            logger.debug("All contexts reset")

    def get_context_count(self) -> int:
        """
        Get the number of initialized contexts.

        Returns
        -------
            int: Number of initialized contexts

        """
        with self._lock:
            return len(self._initialized_contexts)

    def get_context_ids(self) -> Set[str]:
        """
        Get all initialized context IDs.

        Returns
        -------
            Set[str]: Set of all initialized context IDs

        """
        with self._lock:
            return self._initialized_contexts.copy()


def safe_configure_container(config, context_id: str = "default", container_state=None):
    """
    Configure container with error recovery.

    This function provides a safe way to configure the container with
    error recovery and rollback mechanisms.

    Args:
    ----
        config: Configuration object
        context_id: Context identifier for isolation
        container_state: ContainerState instance (for testing)

    Returns:
    -------
        Container instance

    Raises:
    ------
        ConfigurationError: If configuration fails and cannot be recovered

    """
    if container_state is None:
        # Import here to avoid circular imports
        from saplings.api.di import _container_state

        container_state = _container_state

    try:
        # Import here to avoid circular imports
        from saplings.api.di import configure_container

        return configure_container(config, context_id)
    except Exception as e:
        logger.error(f"Container configuration failed: {e}")

        # Reset to known good state
        container_state.reset_context(context_id)

        # Retry with minimal configuration
        try:
            return configure_container_minimal(config, context_id)
        except Exception as retry_error:
            raise ConfigurationError(f"Failed to configure container: {retry_error}") from e


def configure_container_minimal(config, context_id: str = "default"):
    """
    Configure container with minimal configuration for error recovery.

    Args:
    ----
        config: Configuration object
        context_id: Context identifier

    Returns:
    -------
        Container instance with minimal configuration

    """
    # Import here to avoid circular imports
    from saplings.api.di import container

    # Register only the configuration object
    if config is not None:
        container.register(config.__class__, instance=config)

    logger.info(f"Container configured with minimal configuration for context: {context_id}")
    return container


def reset_container_context(context_id: str = "default"):
    """
    Reset container context safely.

    Args:
    ----
        context_id: Context identifier to reset

    """
    # Import here to avoid circular imports
    from saplings.api.di import _container_state, container

    with _container_state.configuration_context(context_id):
        # Clear container state
        container.clear()

        # Reset the context in container state
        _container_state.reset_context(context_id)

        logger.info(f"Container context reset: {context_id}")


__all__ = [
    "ContainerState",
    "ConfigurationError",
    "safe_configure_container",
    "configure_container_minimal",
    "reset_container_context",
]
