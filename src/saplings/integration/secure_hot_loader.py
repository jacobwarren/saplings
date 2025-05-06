from __future__ import annotations

"""
Secure hot-loading system for Saplings.

This module provides an enhanced version of the hot-loading mechanism with
proper sandboxing for all dynamically loaded code.
"""


import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from saplings.integration.hot_loader import HotLoader, HotLoaderConfig
from saplings.tool_factory.config import SandboxType, ToolFactoryConfig
from saplings.tool_factory.sandbox import Sandbox, get_sandbox

if TYPE_CHECKING:
    from saplings.core.plugin import (
        ToolPlugin,
    )

logger = logging.getLogger(__name__)


@dataclass
class SecureHotLoaderConfig(HotLoaderConfig):
    """Configuration for the secure hot-loading system."""

    sandbox_type: SandboxType = SandboxType.DOCKER
    """Type of sandbox to use for code execution."""

    sandbox_timeout: float = 30.0
    """Timeout in seconds for sandbox execution."""

    enable_sandboxing: bool = True
    """Whether to enable sandboxing for all dynamically loaded code."""

    docker_image: str | None = "python:3.9-slim"
    """Docker image to use for Docker sandbox."""

    e2b_api_key: str | None = None
    """API key for E2B sandbox."""

    allowed_imports: list[str] = field(default_factory=list)
    """List of modules that are allowed to be imported in the sandbox."""

    blocked_imports: list[str] = field(default_factory=lambda: ["os", "subprocess", "sys"])
    """List of modules that are blocked from being imported in the sandbox."""

    resource_limits: dict[str, float] = field(
        default_factory=lambda: {
            "memory_mb": 512,
            "cpu_seconds": 30,
            "file_size_kb": 1024,
        }
    )
    """Resource limits for the sandbox execution."""


class SecureHotLoader(HotLoader):
    """
    Secure hot-loading system for tools.

    This class extends the standard HotLoader with additional security measures,
    particularly sandboxing for all dynamically loaded code.
    """

    def __init__(self, config: SecureHotLoaderConfig | None = None) -> None:
        """
        Initialize the secure hot-loading system.

        Args:
        ----
            config: Configuration for the secure hot-loading system

        """
        self.secure_config = config or SecureHotLoaderConfig()
        super().__init__(self.secure_config)

        # Initialize sandbox
        self.sandbox: Sandbox | None = None
        if self.secure_config.enable_sandboxing:
            tool_factory_config = ToolFactoryConfig(
                sandbox_type=self.secure_config.sandbox_type,
                sandbox_timeout=int(self.secure_config.sandbox_timeout),
                docker_image=self.secure_config.docker_image,
                e2b_api_key=self.secure_config.e2b_api_key,
                # Add allowed and blocked imports to metadata since they're not direct parameters
                metadata={
                    "allowed_imports": self.secure_config.allowed_imports,
                    "blocked_imports": self.secure_config.blocked_imports,
                    "resource_limits": self.secure_config.resource_limits,
                },
            )
            self.sandbox = get_sandbox(tool_factory_config)
            logger.info(f"Initialized sandbox of type {self.secure_config.sandbox_type}")

    def load_module_from_file(self, file_path: str) -> Any | None:
        """
        Load a module from a file with sandboxing.

        Args:
        ----
            file_path: Path to the file

        Returns:
        -------
            Optional[Any]: Loaded module if successful, None otherwise

        """
        if not self.secure_config.enable_sandboxing:
            # Fall back to the original implementation if sandboxing is disabled
            return super().load_module_from_file(file_path)

        try:
            # Get the module name from the file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]

            # Read the module content
            with open(file_path) as f:
                code = f.read()

            # Execute the module code in a sandbox if available
            if self.sandbox is not None:
                # Create a wrapper that captures the module's globals
                wrapper_code = f"""
def get_module_globals():
    module_globals = dict()
    exec('''{code}''', module_globals)
    return module_globals
                """

                # Execute the wrapper in the sandbox
                sandbox_result = asyncio.run(
                    self.sandbox.execute(
                        code=wrapper_code,
                        function_name="get_module_globals",
                        args=[],
                        kwargs={},
                    )
                )

                # Create a module object
                module = types.ModuleType(module_name)

                # Populate the module with the globals from the sandbox
                for key, value in sandbox_result.items():
                    if key != "__builtins__":
                        setattr(module, key, value)

                return module
            # If sandbox initialization failed, log a warning and fall back to the original implementation
            logger.warning("Sandbox not available, falling back to unsafe dynamic import")
            return super().load_module_from_file(file_path)
        except Exception as e:
            logger.exception(f"Error loading module from {file_path}: {e}")
            return None

    def load_tool(self, tool_class: type[ToolPlugin]) -> type[ToolPlugin]:
        """
        Load a tool with additional security checks.

        Args:
        ----
            tool_class: Tool class to load

        Returns:
        -------
            Type[ToolPlugin]: Loaded tool class

        """
        # Create a secure wrapper around the tool's execute method if sandboxing is enabled
        if self.secure_config.enable_sandboxing and self.sandbox is not None:
            # Create a new class that inherits from the tool class
            # This allows us to add attributes and methods without modifying the original class
            class SecureToolClass(tool_class):  # type: ignore
                """Secure version of the tool class with sandboxed execution."""

                # Store a reference to the hot loader and sandbox
                _secure_hot_loader = self
                _secure_sandbox = self.sandbox

                async def execute(self, *args, **kwargs):
                    """Secure execute method that runs in a sandbox."""
                    # Get the code for the original execute method
                    import inspect

                    # Get the original method from the parent class
                    original_method = getattr(super(SecureToolClass, self), "execute", None)

                    if original_method is None:
                        msg = "Tool class does not have an execute method"
                        raise ValueError(msg)

                    # Get the source code of the original method
                    try:
                        code = inspect.getsource(original_method)
                    except (TypeError, OSError) as e:
                        msg = f"Could not get source code for execute method: {e}"
                        raise ValueError(msg)

                    # Execute the code in the sandbox
                    # Use the stored sandbox reference
                    return await self.__class__._secure_sandbox.execute(
                        code=code,
                        function_name="execute",
                        args=[self, *list(args)],
                        kwargs=kwargs,
                    )

            # Use the secure class instead of the original
            return super().load_tool(SecureToolClass)

        # Continue with the standard loading process
        return super().load_tool(tool_class)

    def cleanup(self) -> None:
        """Clean up resources used by the secure hot loader."""
        if self.sandbox is not None:
            self.sandbox.cleanup()
            self.sandbox = None

        # Stop auto-reload if running
        self.stop_auto_reload()


# For backward compatibility
import types


def create_secure_hot_loader(config: SecureHotLoaderConfig | None = None) -> SecureHotLoader:
    """
    Create a secure hot loader instance.

    This function is provided for backward compatibility with code that
    expects a HotLoader but wants the security benefits of SecureHotLoader.

    Args:
    ----
        config: Configuration for the secure hot loader

    Returns:
    -------
        SecureHotLoader: A secure hot loader instance

    """
    return SecureHotLoader(config)
