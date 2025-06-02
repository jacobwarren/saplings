from __future__ import annotations

"""
Import hook for detecting usage of internal APIs.

This module provides an import hook that detects imports of internal modules
and symbols, and issues deprecation warnings when they are used.
"""

import importlib.abc
import importlib.machinery
import logging
import re
import sys
import types
import warnings
from collections.abc import Sequence
from typing import Any, Dict, Optional, Set, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Regular expressions for detecting internal modules and symbols
INTERNAL_MODULE_PATTERN = re.compile(r"^saplings\.(_.*|.*\._.*|.*\._internal.*)")
INTERNAL_SYMBOL_PATTERN = re.compile(r"^_.*")

# Public API alternatives for common internal modules
PUBLIC_API_ALTERNATIVES = {
    # Top-level internal modules
    "saplings._internal.agent": "saplings.api.agent",
    "saplings._internal.agent_builder": "saplings.api.agent",
    "saplings._internal.agent_config": "saplings.api.agent",
    "saplings._internal.agent_facade": "saplings.api.agent",
    "saplings._internal.agent_facade_builder": "saplings.api.agent",
    "saplings._internal.agent_module": "saplings.api.agent",
    "saplings._internal.agent_class": "saplings.api.agent",
    "saplings._internal.agent_builder_module": "saplings.api.agent",
    "saplings._internal.agent_builder_module_part2": "saplings.api.agent",
    "saplings._internal.agent_builder_build": "saplings.api.agent",
    "saplings._internal._agent_facade": "saplings.api.agent",
    "saplings._internal._agent_facade_builder": "saplings.api.agent",
    "saplings._internal.container": "saplings.api.container",
    "saplings._internal.di": "saplings.api.container",
    "saplings._internal.memory": "saplings.api.memory",
    "saplings._internal.tools": "saplings.api.tools",
    # Component internal modules
    "saplings.core._internal": "saplings.api",
    "saplings.memory._internal": "saplings.api.memory",
    "saplings.adapters._internal": "saplings.api.models",
    "saplings.tools._internal": "saplings.api.tools",
    "saplings.gasa._internal": "saplings.api.gasa",
    "saplings.services._internal": "saplings.api.services",
    "saplings.retrieval._internal": "saplings.api.retrieval",
    "saplings.monitoring._internal": "saplings.api.monitoring",
    "saplings.judge._internal": "saplings.api.judge",
    "saplings.validator._internal": "saplings.api.validator",
    "saplings.executor._internal": "saplings.api.executor",
    "saplings.planner._internal": "saplings.api.planner",
    "saplings.integration._internal": "saplings.api.integration",
    "saplings.modality._internal": "saplings.api.modality",
    "saplings.orchestration._internal": "saplings.api.orchestration",
    "saplings.security._internal": "saplings.api.security",
    "saplings.security._internal.hooks": "saplings.api.security",
    "saplings.security._internal.logging": "saplings.api.security",
    "saplings.security._internal.sanitization": "saplings.api.security",
    "saplings.self_heal._internal": "saplings.api.self_heal",
    "saplings.tokenizers._internal": "saplings.api.tokenizers",
    "saplings.tool_factory._internal": "saplings.api.tool_factory",
    "saplings.utils._internal": "saplings.api.utils",
    "saplings.di._internal": "saplings.api.container",
    "saplings.plugins._internal": "saplings.api.plugins",
    "saplings.runtimes._internal": "saplings.api.runtimes",
    "saplings.typings._internal": "saplings.api",
    # Specific internal modules
    "saplings.memory._internal.indexer": "saplings.api.memory.indexer",
    "saplings.memory._internal.vector_store": "saplings.api.vector_store",
    "saplings.memory._internal.document": "saplings.api.memory.document",
}

# Modules to exclude from warnings (e.g., typing-related modules)
EXCLUDED_MODULES = {
    # Typing modules
    "saplings._typing",
    "saplings.typings",
    # Top-level internal packages
    "saplings._internal",  # Allow importing the _internal package itself
    "saplings._internal.agent",  # Allow importing the agent package itself
    "saplings._internal.memory",  # Allow importing the memory package itself
    "saplings._internal.tools",  # Allow importing the tools package itself
    "saplings._internal.agent_module",  # Allow importing the agent module
    "saplings._internal.agent_class",  # Allow importing the agent class module
    "saplings._internal.agent_builder_module",  # Allow importing the agent builder module
    "saplings._internal.agent_builder_module_part2",  # Allow importing the agent builder module part 2
    "saplings._internal.agent_builder_build",  # Allow importing the agent builder build module
    "saplings._internal._agent_facade",  # Allow importing the agent facade module
    "saplings._internal._agent_facade_builder",  # Allow importing the agent facade builder module
    # Component internal packages
    "saplings.core._internal",  # Allow importing the core._internal package itself
    "saplings.adapters._internal",  # Allow importing the adapters._internal package itself
    "saplings.memory._internal",  # Allow importing the memory._internal package itself
    "saplings.tools._internal",  # Allow importing the tools._internal package itself
    "saplings.gasa._internal",  # Allow importing the gasa._internal package itself
    "saplings.services._internal",  # Allow importing the services._internal package itself
    "saplings.retrieval._internal",  # Allow importing the retrieval._internal package itself
    "saplings.monitoring._internal",  # Allow importing the monitoring._internal package itself
    "saplings.judge._internal",  # Allow importing the judge._internal package itself
    "saplings.validator._internal",  # Allow importing the validator._internal package itself
    "saplings.executor._internal",  # Allow importing the executor._internal package itself
    "saplings.planner._internal",  # Allow importing the planner._internal package itself
    "saplings.integration._internal",  # Allow importing the integration._internal package itself
    "saplings.modality._internal",  # Allow importing the modality._internal package itself
    "saplings.orchestration._internal",  # Allow importing the orchestration._internal package itself
    "saplings.security._internal",  # Allow importing the security._internal package itself
    "saplings.security._internal.hooks",  # Allow importing the security._internal.hooks package itself
    "saplings.security._internal.logging",  # Allow importing the security._internal.logging package itself
    "saplings.security._internal.sanitization",  # Allow importing the security._internal.sanitization package itself
    "saplings.self_heal._internal",  # Allow importing the self_heal._internal package itself
    "saplings.tokenizers._internal",  # Allow importing the tokenizers._internal package itself
    "saplings.tool_factory._internal",  # Allow importing the tool_factory._internal package itself
    "saplings.utils._internal",  # Allow importing the utils._internal package itself
    "saplings.di._internal",  # Allow importing the di._internal package itself
    "saplings.plugins._internal",  # Allow importing the plugins._internal package itself
    "saplings.runtimes._internal",  # Allow importing the runtimes._internal package itself
    "saplings.typings._internal",  # Allow importing the typings._internal package itself
}

# Cache of already warned modules to avoid duplicate warnings
_warned_imports: Set[str] = set()


class InternalAPIWarningFinder(importlib.abc.MetaPathFinder):
    """
    Import finder that detects imports of internal APIs.

    This finder wraps the standard import machinery to detect imports
    of internal modules and symbols, and issues deprecation warnings
    when they are used.
    """

    def __init__(self) -> None:
        """Initialize the finder."""
        self.original_import = __import__

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """
        Find the module spec for a module.

        Args:
        ----
            fullname: Full name of the module
            path: Path to search for the module
            target: Target module

        Returns:
        -------
            The module spec if found, None otherwise

        """
        # Check if this is an internal module
        if self._is_internal_module(fullname) and fullname not in EXCLUDED_MODULES:
            if fullname not in _warned_imports:
                # Try to suggest a public API alternative
                public_alternative = self._suggest_public_alternative(fullname)
                warning_message = (
                    f"The module '{fullname}' is an internal API and may change without notice. "
                    "Please use only public APIs from the saplings.api module."
                )
                if public_alternative:
                    warning_message += f" Consider using '{public_alternative}' instead."

                # Raise an error instead of a warning
                msg = f"\n\n⚠️  INTERNAL API USAGE ERROR ⚠️\n{warning_message}\n"
                # Add to warned imports to avoid duplicate warnings in case error is caught
                _warned_imports.add(fullname)
                raise ImportError(msg)

        # Let the standard import machinery handle the actual import
        return None

    def _is_internal_module(self, fullname: str) -> bool:
        """
        Check if a module is an internal module.

        Args:
        ----
            fullname: Full name of the module

        Returns:
        -------
            True if the module is an internal module, False otherwise

        """
        return bool(INTERNAL_MODULE_PATTERN.match(fullname))

    def _suggest_public_alternative(self, fullname: str) -> Optional[str]:
        """
        Suggest a public API alternative for an internal module.

        Args:
        ----
            fullname: Full name of the internal module

        Returns:
        -------
            Public API alternative if available, None otherwise

        """
        # Check for exact matches
        alternatives = PUBLIC_API_ALTERNATIVES
        if fullname in alternatives:
            return alternatives[fullname]

        # Check for partial matches (e.g., for submodules)
        for internal, public in alternatives.items():
            if fullname.startswith(internal):
                return public

        # Generic fallback for common patterns
        if "_internal" in fullname:
            # Extract the main module name (e.g., "saplings.memory._internal.xyz" -> "memory")
            parts = fullname.split(".")
            for i, part in enumerate(parts):
                if part == "_internal" and i > 0:
                    main_module = parts[i - 1]
                    return f"saplings.api.{main_module}"

        # No suggestion available
        return None


class InternalAPIWarningLoader:
    """
    Loader that wraps modules to detect access to internal symbols.

    This loader wraps imported modules to detect access to internal symbols
    and issues deprecation warnings when they are used.
    """

    @staticmethod
    def wrap_module(module: types.ModuleType) -> types.ModuleType:
        """
        Wrap a module to detect access to internal symbols.

        Args:
        ----
            module: The module to wrap

        Returns:
        -------
            The wrapped module

        """
        if not module.__name__.startswith("saplings."):
            return module

        # Store the original __getattr__ if it exists
        original_getattr = getattr(module, "__getattr__", None)

        def wrapped_getattr(name: str) -> Any:
            """
            Wrapped __getattr__ that detects access to internal symbols.

            Args:
            ----
                name: The name of the attribute to get

            Returns:
            -------
                The attribute value

            Raises:
            ------
                AttributeError: If the attribute does not exist

            """
            # Check if this is an internal symbol
            if INTERNAL_SYMBOL_PATTERN.match(name) and not name.startswith("__"):
                import_path = f"{module.__name__}.{name}"
                if import_path not in _warned_imports:
                    warning_message = (
                        f"The symbol '{name}' in module '{module.__name__}' is an internal API "
                        "and may change without notice. Please use only public APIs from the "
                        "saplings.api module."
                    )

                    warnings.warn(
                        warning_message,
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    logger.warning(
                        f"Internal API usage detected: {import_path}. "
                        "This is not part of the public API and may change without notice."
                    )
                    _warned_imports.add(import_path)

            # Call the original __getattr__ if it exists
            if original_getattr is not None:
                return original_getattr(name)

            # Otherwise, raise AttributeError
            msg = f"module '{module.__name__}' has no attribute '{name}'"
            raise AttributeError(msg)

        # Replace __getattr__ with our wrapped version
        if hasattr(module, "__getattr__"):
            module.__getattr__ = wrapped_getattr

        return module


def install_import_hook() -> None:
    """
    Install the import hook for detecting usage of internal APIs.

    This function installs an import hook that detects imports of internal
    modules and symbols, and issues deprecation warnings when they are used.
    """
    # Install the finder
    finder = InternalAPIWarningFinder()
    sys.meta_path.insert(0, finder)

    # Patch the import function
    original_import = __import__

    def import_wrapper(
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: Tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """
        Wrapped import function that detects imports of internal modules and symbols.

        Args:
        ----
            name: The name of the module to import
            globals: The global namespace
            locals: The local namespace
            fromlist: The list of symbols to import from the module
            level: The level of relative imports

        Returns:
        -------
            The imported module

        """
        module = original_import(name, globals, locals, fromlist, level)

        # Check if this is a saplings module
        if name.startswith("saplings.") or (
            level > 0 and globals and globals.get("__name__", "").startswith("saplings.")
        ):
            # Wrap the module to detect access to internal symbols
            module = InternalAPIWarningLoader.wrap_module(module)

            # Check if any symbols in fromlist are internal
            if fromlist:
                for symbol in fromlist:
                    if INTERNAL_SYMBOL_PATTERN.match(symbol) and not symbol.startswith("__"):
                        import_path = f"{module.__name__}.{symbol}"
                        if import_path not in _warned_imports:
                            # Try to suggest a public API alternative
                            public_alternative = None
                            if module.__name__.startswith("saplings."):
                                finder = InternalAPIWarningFinder()
                                public_alternative = finder._suggest_public_alternative(
                                    module.__name__
                                )

                            warning_message = (
                                f"The symbol '{symbol}' in module '{module.__name__}' is an internal API "
                                "and may change without notice. Please use only public APIs from the "
                                "saplings.api module."
                            )
                            if public_alternative:
                                warning_message += (
                                    f" Consider using '{public_alternative}' instead."
                                )

                            # Raise an error instead of a warning
                            logger.error(
                                f"Internal API usage detected: {import_path}. "
                                "This is not part of the public API and may change without notice."
                            )
                            # Add to warned imports to avoid duplicate errors in case error is caught
                            _warned_imports.add(import_path)
                            msg = f"\n\n⚠️  INTERNAL API USAGE ERROR ⚠️\n{warning_message}\n"
                            raise ImportError(msg)

        return module

    # Replace the built-in import function
    __builtins__["__import__"] = import_wrapper  # type: ignore

    logger.info("Import hook for detecting internal API usage installed")
