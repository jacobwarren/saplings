from __future__ import annotations

"""
Tool factory module for Saplings.

This module provides dynamic tool synthesis capabilities for Saplings, including:
- Tool specification schema
- Template-based tool generation
- Code validation and security checks
- Tool registry
"""


from saplings.tool_factory.code_signing import CodeSigner, SignatureVerifier
from saplings.tool_factory.config import (
    SandboxType,
    SecurityLevel,
    SigningLevel,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
)
from saplings.tool_factory.sandbox import DockerSandbox, E2BSandbox, Sandbox
from saplings.tool_factory.tool_factory import ToolFactory
from saplings.tool_factory.tool_validator import ToolValidator, ValidationResult

__all__ = [
    "CodeSigner",
    "DockerSandbox",
    "E2BSandbox",
    "Sandbox",
    "SandboxType",
    "SecurityLevel",
    "SignatureVerifier",
    "SigningLevel",
    "ToolFactory",
    "ToolFactoryConfig",
    "ToolSpecification",
    "ToolTemplate",
    "ToolValidator",
    "ValidationResult",
]
