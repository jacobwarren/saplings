"""
Centralized lazy import system for Saplings.

This module provides a systematic approach to lazy loading of heavy dependencies,
improving import performance while maintaining functionality.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class OptionalDependency:
    """
    Represents an optional dependency with metadata and availability checking.

    This class encapsulates information about optional dependencies including
    their import names, install commands, version requirements, and availability.
    """

    def __init__(
        self,
        import_name: str,
        install_cmd: str,
        min_version: Optional[str] = None,
        package_name: Optional[str] = None,
        check_function: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize an optional dependency.

        Args:
        ----
            import_name: The name used for importing (e.g., 'torch')
            install_cmd: Command to install the dependency (e.g., 'pip install torch')
            min_version: Minimum required version
            package_name: Package name if different from import_name
            check_function: Custom function to check availability

        """
        self.import_name = import_name
        self.install_cmd = install_cmd
        self.min_version = min_version
        self.package_name = package_name or import_name
        self.check_function = check_function
        self._available: Optional[bool] = None
        self._module: Optional[Any] = None

    @property
    def available(self) -> bool:
        """Check if the dependency is available."""
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """Internal method to check dependency availability."""
        if self.check_function:
            try:
                return self.check_function()
            except Exception:
                return False

        try:
            spec = importlib.util.find_spec(self.import_name)
            if spec is None:
                return False

            # If version checking is required, try to import and check version
            if self.min_version:
                try:
                    module = importlib.import_module(self.import_name)
                    if hasattr(module, "__version__"):
                        from packaging import version

                        return version.parse(module.__version__) >= version.parse(self.min_version)
                except Exception:
                    return False

            return True
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def require(self) -> Any:
        """
        Import and return the module, raising ImportError if not available.

        Returns
        -------
            The imported module

        Raises
        ------
            ImportError: If the dependency is not available

        """
        if not self.available:
            raise ImportError(
                f"Optional dependency '{self.import_name}' is not available. "
                f"Install with: {self.install_cmd}"
            )

        if self._module is None:
            self._module = importlib.import_module(self.import_name)

        return self._module

    def get_or_none(self) -> Optional[Any]:
        """
        Import and return the module, or None if not available.

        Returns
        -------
            The imported module or None if not available

        """
        try:
            return self.require()
        except ImportError:
            return None


class LazyImporter:
    """
    A lazy importer that defers module loading until first access.

    This class provides a systematic way to implement lazy loading for heavy
    dependencies, improving import performance.
    """

    def __init__(
        self,
        module_name: str,
        error_message: Optional[str] = None,
        optional_dependency: Optional[OptionalDependency] = None,
    ):
        """
        Initialize a lazy importer.

        Args:
        ----
            module_name: Name of the module to import lazily
            error_message: Custom error message if import fails
            optional_dependency: OptionalDependency instance for this module

        """
        self.module_name = module_name
        self.error_message = error_message or f"Module '{module_name}' is not available"
        self.optional_dependency = optional_dependency
        self._module: Optional[Any] = None
        self._import_attempted = False

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the lazily imported module."""
        module = self._get_module()
        return getattr(module, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the lazy importer to be called if the module is callable."""
        module = self._get_module()
        return module(*args, **kwargs)

    def _get_module(self) -> Any:
        """Get the module, importing it if necessary."""
        if self._module is None and not self._import_attempted:
            self._import_attempted = True
            try:
                if self.optional_dependency:
                    self._module = self.optional_dependency.require()
                else:
                    self._module = importlib.import_module(self.module_name)
                logger.debug(f"Lazy loaded module: {self.module_name}")
            except ImportError as e:
                logger.warning(f"Failed to lazy load {self.module_name}: {e}")
                raise ImportError(self.error_message) from e

        if self._module is None:
            raise ImportError(self.error_message)

        return self._module

    @property
    def available(self) -> bool:
        """Check if the module is available without importing it."""
        if self.optional_dependency:
            return self.optional_dependency.available

        try:
            spec = importlib.util.find_spec(self.module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False


def lazy_import(
    module_name: str,
    error_message: Optional[str] = None,
    optional_dependency: Optional[OptionalDependency] = None,
) -> LazyImporter:
    """
    Create a lazy importer for a module.

    Args:
    ----
        module_name: Name of the module to import lazily
        error_message: Custom error message if import fails
        optional_dependency: OptionalDependency instance for this module

    Returns:
    -------
        LazyImporter instance

    """
    return LazyImporter(module_name, error_message, optional_dependency)


def require_dependencies(*dep_names: str) -> None:
    """
    Require multiple dependencies to be available.

    Args:
    ----
        *dep_names: Names of dependencies to require

    Raises:
    ------
        ImportError: If any dependency is not available

    """
    # Avoid circular import by importing here
    try:
        from saplings._internal.optional_deps import OPTIONAL_DEPENDENCIES
    except ImportError:
        # If optional_deps is not available, just check basic availability
        OPTIONAL_DEPENDENCIES = {}

    missing = []
    for dep_name in dep_names:
        if dep_name in OPTIONAL_DEPENDENCIES:
            dep = OPTIONAL_DEPENDENCIES[dep_name]
            if not dep.available:
                missing.append(dep_name)
        else:
            # Check if it's a standard library or available module
            try:
                spec = importlib.util.find_spec(dep_name)
                if spec is None:
                    missing.append(dep_name)
            except (ImportError, ModuleNotFoundError, ValueError):
                missing.append(dep_name)

    if missing:
        raise ImportError(f"Required dependencies not available: {', '.join(missing)}")


def check_dependencies(*dep_names: str) -> Dict[str, bool]:
    """
    Check availability of multiple dependencies.

    Args:
    ----
        *dep_names: Names of dependencies to check

    Returns:
    -------
        Dictionary mapping dependency names to availability status

    """
    # Avoid circular import by importing here
    try:
        from saplings._internal.optional_deps import OPTIONAL_DEPENDENCIES
    except ImportError:
        # If optional_deps is not available, just check basic availability
        OPTIONAL_DEPENDENCIES = {}

    result = {}
    for dep_name in dep_names:
        if dep_name in OPTIONAL_DEPENDENCIES:
            result[dep_name] = OPTIONAL_DEPENDENCIES[dep_name].available
        else:
            try:
                spec = importlib.util.find_spec(dep_name)
                result[dep_name] = spec is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                result[dep_name] = False

    return result
