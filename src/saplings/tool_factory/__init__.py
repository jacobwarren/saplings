from __future__ import annotations

"""
Tool factory module for Saplings.

This module re-exports the public API from saplings.api.tool_factory.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides dynamic tool synthesis capabilities for Saplings, including:
- Tool specification schema
- Template-based tool generation
- Code validation and security checks
- Tool registry
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.tool_factory.

__all__ = [
    "CodeSigner",
    "DockerSandbox",
    "E2BSandbox",
    "Sandbox",
    "SandboxType",
    "SecureHotLoader",
    "SecureHotLoaderConfig",
    "SecurityLevel",
    "SignatureVerifier",
    "SigningLevel",
    "ToolFactory",
    "ToolFactoryConfig",
    "ToolSpecification",
    "ToolTemplate",
    "ToolValidator",
    "ValidationResult",
    "create_secure_hot_loader",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.tool_factory import (
            CodeSigner,
            DockerSandbox,
            E2BSandbox,
            Sandbox,
            SandboxType,
            SecureHotLoader,
            SecureHotLoaderConfig,
            SecurityLevel,
            SignatureVerifier,
            SigningLevel,
            ToolFactory,
            ToolFactoryConfig,
            ToolSpecification,
            ToolTemplate,
            ToolValidator,
            ValidationResult,
            create_secure_hot_loader,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "CodeSigner": CodeSigner,
            "DockerSandbox": DockerSandbox,
            "E2BSandbox": E2BSandbox,
            "Sandbox": Sandbox,
            "SandboxType": SandboxType,
            "SecureHotLoader": SecureHotLoader,
            "SecureHotLoaderConfig": SecureHotLoaderConfig,
            "SecurityLevel": SecurityLevel,
            "SignatureVerifier": SignatureVerifier,
            "SigningLevel": SigningLevel,
            "ToolFactory": ToolFactory,
            "ToolFactoryConfig": ToolFactoryConfig,
            "ToolSpecification": ToolSpecification,
            "ToolTemplate": ToolTemplate,
            "ToolValidator": ToolValidator,
            "ValidationResult": ValidationResult,
            "create_secure_hot_loader": create_secure_hot_loader,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
