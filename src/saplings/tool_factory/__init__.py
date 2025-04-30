"""
Tool factory module for Saplings.

This module provides dynamic tool synthesis capabilities for Saplings, including:
- Tool specification schema
- Template-based tool generation
- Code validation and security checks
- Tool registry
"""

from saplings.tool_factory.config import (
    ToolSpecification,
    ToolFactoryConfig,
    ToolTemplate,
    SecurityLevel,
    SandboxType,
    SigningLevel,
)
from saplings.tool_factory.tool_factory import ToolFactory
from saplings.tool_factory.sandbox import Sandbox, DockerSandbox, E2BSandbox
from saplings.tool_factory.code_signing import CodeSigner, SignatureVerifier
from saplings.tool_factory.tool_validator import ToolValidator, ValidationResult

__all__ = [
    "ToolSpecification",
    "ToolFactoryConfig",
    "ToolTemplate",
    "SecurityLevel",
    "SandboxType",
    "SigningLevel",
    "ToolFactory",
    "Sandbox",
    "DockerSandbox",
    "E2BSandbox",
    "CodeSigner",
    "SignatureVerifier",
    "ToolValidator",
    "ValidationResult",
]
