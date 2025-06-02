"""
Centralized import manager to resolve circular import issues.

This module provides a comprehensive solution for managing imports across the
entire Saplings codebase, preventing circular dependencies through lazy loading,
import ordering, and dependency injection patterns.
"""

from __future__ import annotations

import importlib
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Set

# Configure logging
logger = logging.getLogger(__name__)


class ImportManager:
    """
    Centralized import manager that prevents circular imports.

    This class manages all imports across the Saplings codebase, providing:
    - Lazy loading of modules to break circular dependencies
    - Import ordering to prevent cycles
    - Caching to improve performance
    - Thread-safe operations
    """

    _instance: Optional["ImportManager"] = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize the import manager."""
        self._module_cache: Dict[str, Any] = {}
        self._import_stack: Set[str] = set()
        self._lazy_imports: Dict[str, "LazyModule"] = {}
        self._import_lock = threading.RLock()
        self._initialization_order: Dict[str, int] = {}

        # Define safe import order to prevent cycles
        self._define_import_order()

    def __new__(cls) -> "ImportManager":
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def _define_import_order(self):
        """Define the order in which modules should be imported to prevent cycles."""
        # Lower numbers import first, higher numbers import later
        self._initialization_order = {
            # Core types and interfaces (no dependencies)
            "saplings.core._internal.types": 0,
            "saplings.api.core.interfaces": 1,
            "saplings.di._internal.exceptions": 2,
            "saplings.di._internal.scope": 3,
            # DI container (depends on interfaces)
            "saplings.di._internal.container": 10,
            "saplings.di": 11,
            # Service implementations (depend on interfaces and DI)
            "saplings.services._internal.providers": 20,
            "saplings.services._internal.managers": 21,
            # Service builders (depend on implementations)
            "saplings.services._internal.builders": 30,
            # Container configuration (depends on builders)
            "saplings._internal.container_config": 40,
            # API modules (depend on everything else)
            "saplings.api.di": 50,
            "saplings.api.memory": 51,
            "saplings.api.tools": 52,
            "saplings.api": 60,
            # Main package (depends on API)
            "saplings": 100,
        }

    def safe_import(self, module_name: str, lazy: bool = False) -> Any:
        """
        Safely import a module, preventing circular dependencies.

        Args:
        ----
            module_name: Name of the module to import
            lazy: Whether to use lazy loading

        Returns:
        -------
            The imported module or a lazy module proxy

        """
        with self._import_lock:
            # Check if we're already importing this module (circular dependency)
            if module_name in self._import_stack:
                logger.warning(f"Circular import detected for {module_name}, using lazy loading")
                return self._get_lazy_module(module_name)

            # Check cache first
            if module_name in self._module_cache:
                return self._module_cache[module_name]

            # Use lazy loading if requested or if import order suggests it
            if lazy or self._should_use_lazy_loading(module_name):
                return self._get_lazy_module(module_name)

            # Perform the import
            return self._do_import(module_name)

    def _should_use_lazy_loading(self, module_name: str) -> bool:
        """Determine if a module should use lazy loading based on import order."""
        current_order = self._get_import_order(module_name)

        # Check if any module in the import stack has a higher order
        for importing_module in self._import_stack:
            importing_order = self._get_import_order(importing_module)
            if importing_order >= current_order:
                return True

        return False

    def _get_import_order(self, module_name: str) -> int:
        """Get the import order for a module."""
        # Find the most specific match
        for pattern, order in sorted(self._initialization_order.items(), key=lambda x: -len(x[0])):
            if module_name.startswith(pattern):
                return order

        # Default order for unknown modules
        return 999

    def _do_import(self, module_name: str) -> Any:
        """Perform the actual import."""
        try:
            self._import_stack.add(module_name)
            logger.debug(f"Importing {module_name}")

            module = importlib.import_module(module_name)
            self._module_cache[module_name] = module

            logger.debug(f"Successfully imported {module_name}")
            return module

        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            raise
        finally:
            self._import_stack.discard(module_name)

    def _get_lazy_module(self, module_name: str) -> "LazyModule":
        """Get or create a lazy module proxy."""
        if module_name not in self._lazy_imports:
            self._lazy_imports[module_name] = LazyModule(module_name, self)
        return self._lazy_imports[module_name]

    @contextmanager
    def import_context(self, module_name: str):
        """Context manager for tracking import context."""
        self._import_stack.add(module_name)
        try:
            yield
        finally:
            self._import_stack.discard(module_name)

    def clear_cache(self):
        """Clear the module cache."""
        with self._import_lock:
            self._module_cache.clear()
            self._lazy_imports.clear()
            self._import_stack.clear()


class LazyModule:
    """
    Lazy module proxy that imports the module only when accessed.
    """

    def __init__(self, module_name: str, import_manager: ImportManager):
        """Initialize the lazy module."""
        self._module_name = module_name
        self._import_manager = import_manager
        self._module: Optional[Any] = None
        self._import_attempted = False

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the lazily imported module."""
        module = self._get_module()
        return getattr(module, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the lazy module to be called if it's callable."""
        module = self._get_module()
        return module(*args, **kwargs)

    def _get_module(self) -> Any:
        """Get the actual module, importing it if necessary."""
        if self._module is None and not self._import_attempted:
            self._import_attempted = True
            try:
                # Use the import manager to safely import
                self._module = self._import_manager._do_import(self._module_name)
                logger.debug(f"Lazy loaded module: {self._module_name}")
            except ImportError as e:
                logger.error(f"Failed to lazy load {self._module_name}: {e}")
                raise

        if self._module is None:
            raise ImportError(f"Module '{self._module_name}' could not be imported")

        return self._module


# Global import manager instance
_import_manager: Optional[ImportManager] = None


def get_import_manager() -> ImportManager:
    """Get the global import manager instance."""
    global _import_manager
    if _import_manager is None:
        _import_manager = ImportManager()
    return _import_manager


def safe_import(module_name: str, lazy: bool = False) -> Any:
    """
    Safely import a module using the global import manager.

    Args:
    ----
        module_name: Name of the module to import
        lazy: Whether to use lazy loading

    Returns:
    -------
        The imported module or a lazy module proxy

    """
    return get_import_manager().safe_import(module_name, lazy)


def lazy_import(module_name: str) -> LazyModule:
    """
    Create a lazy import for a module.

    Args:
    ----
        module_name: Name of the module to import lazily

    Returns:
    -------
        LazyModule proxy

    """
    return get_import_manager()._get_lazy_module(module_name)
