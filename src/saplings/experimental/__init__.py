"""
Experimental features module for Saplings.

⚠️  WARNING: Experimental features may change significantly in future versions.
These features are under active development and should be used with caution
in production environments.

Features included:
- Tool Factory and dynamic tool generation
- Self-healing and adaptive capabilities
- Security and sandboxing features
- Hot-loading and code generation

Usage:
    import warnings
    from saplings.experimental import ToolFactory, PatchGenerator

    # Experimental features will show warnings
    # Install with: pip install saplings[experimental]
"""

from __future__ import annotations

import warnings

# Issue warning for experimental features
warnings.warn(
    "Experimental features may change significantly in future versions. "
    "Use with caution in production environments.",
    FutureWarning,
    stacklevel=2,
)

# Import experimental features from the API
from saplings.api.security import (
    RedactingFilter,
    Sanitizer,
    install_global_filter,
    install_import_hook,
    redact,
    sanitize,
)
from saplings.api.self_heal import (
    Adapter,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
    LoRaConfig,
    LoRaTrainer,
    Patch,
    PatchGenerator,
    PatchResult,
    PatchStatus,
    RetryStrategy,
    SelfHealingConfig,
    SuccessPairCollector,
    TrainingMetrics,
)
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

__all__ = [
    # Tool Factory
    "ToolFactory",
    "ToolFactoryConfig",
    "ToolSpecification",
    "ToolTemplate",
    "ToolValidator",
    "ValidationResult",
    "SecureHotLoader",
    "SecureHotLoaderConfig",
    "create_secure_hot_loader",
    # Security and Sandboxing
    "CodeSigner",
    "SignatureVerifier",
    "SecurityLevel",
    "SigningLevel",
    "DockerSandbox",
    "E2BSandbox",
    "Sandbox",
    "SandboxType",
    # Self-healing
    "PatchGenerator",
    "Patch",
    "PatchResult",
    "PatchStatus",
    "AdapterManager",
    "Adapter",
    "AdapterMetadata",
    "AdapterPriority",
    "LoRaTrainer",
    "LoRaConfig",
    "TrainingMetrics",
    "SuccessPairCollector",
    "SelfHealingConfig",
    "RetryStrategy",
    # Security
    "Sanitizer",
    "RedactingFilter",
    "sanitize",
    "redact",
    "install_global_filter",
    "install_import_hook",
]
