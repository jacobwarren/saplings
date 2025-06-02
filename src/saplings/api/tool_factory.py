from __future__ import annotations

"""
Public API for Tool Factory and Secure Hot Loading.

This module provides the public API for dynamic tool creation and secure hot loading,
including:
- Tool factory for dynamic tool synthesis
- Secure hot loading with sandboxing
- Tool templates and specifications
- Code validation and security checks
"""


# Import from internal modules
from saplings.api.integration import (
    SecureHotLoader as _SecureHotLoader,
)
from saplings.api.integration import (
    SecureHotLoaderConfig as _SecureHotLoaderConfig,
)
from saplings.api.integration import (
    create_secure_hot_loader as _create_secure_hot_loader,
)

# Import stability decorators
from saplings.api.stability import beta
from saplings.tool_factory._internal.factory import (
    SandboxType as _SandboxType,
)
from saplings.tool_factory._internal.factory import (
    SecurityLevel as _SecurityLevel,
)
from saplings.tool_factory._internal.factory import (
    SigningLevel as _SigningLevel,
)
from saplings.tool_factory._internal.factory import (
    ToolFactory as _ToolFactory,
)
from saplings.tool_factory._internal.factory import (
    ToolFactoryConfig as _ToolFactoryConfig,
)
from saplings.tool_factory._internal.factory import (
    ToolSpecification as _ToolSpecification,
)
from saplings.tool_factory._internal.factory import (
    ToolTemplate as _ToolTemplate,
)
from saplings.tool_factory._internal.sandbox.sandbox import (
    DockerSandbox as _DockerSandbox,
)
from saplings.tool_factory._internal.sandbox.sandbox import (
    E2BSandbox as _E2BSandbox,
)
from saplings.tool_factory._internal.sandbox.sandbox import (
    Sandbox as _Sandbox,
)
from saplings.tool_factory._internal.security import (
    CodeSigner as _CodeSigner,
)
from saplings.tool_factory._internal.security import (
    SignatureVerifier as _SignatureVerifier,
)
from saplings.tool_factory._internal.security import (
    ToolValidator as _ToolValidator,
)
from saplings.tool_factory._internal.security import (
    ValidationResult as _ValidationResult,
)

# Add stability annotations to re-exported classes
ToolFactoryConfig = beta(_ToolFactoryConfig)
ToolSpecification = beta(_ToolSpecification)
ToolTemplate = beta(_ToolTemplate)
SecurityLevel = beta(_SecurityLevel)
SandboxType = beta(_SandboxType)
SigningLevel = beta(_SigningLevel)
CodeSigner = beta(_CodeSigner)
SignatureVerifier = beta(_SignatureVerifier)
ToolValidator = beta(_ToolValidator)
ValidationResult = beta(_ValidationResult)
Sandbox = beta(_Sandbox)
DockerSandbox = beta(_DockerSandbox)
E2BSandbox = beta(_E2BSandbox)
SecureHotLoader = beta(_SecureHotLoader)
SecureHotLoaderConfig = beta(_SecureHotLoaderConfig)
create_secure_hot_loader = beta(_create_secure_hot_loader)


@beta
class ToolFactory(_ToolFactory):
    """
    Factory for dynamic tool synthesis.

    This class provides functionality for dynamically creating tools based on
    specifications and templates, with security checks and sandboxing.

    The factory uses lazy initialization to avoid circular dependencies and
    improve startup performance.
    """

    @classmethod
    def create(
        cls,
        model=None,
        executor=None,
        config=None,
        template_directories=None,
    ) -> "ToolFactory":
        """
        Create a new ToolFactory instance.

        This factory method provides a convenient way to create a tool factory
        with template directories.

        Args:
        ----
            model: LLM model to use for code generation
            executor: Executor to use for code generation (alternative to model)
            config: Configuration for the tool factory
            template_directories: Optional list of directories containing templates

        Returns:
        -------
            ToolFactory: A new tool factory instance

        """
        factory = cls(model=model, executor=executor, config=config)

        # Add template directories if provided
        if template_directories:
            for directory in template_directories:
                factory.add_template_directory(directory)

        return factory

    @classmethod
    def get_global_instance(
        cls,
        model=None,
        executor=None,
        config=None,
    ) -> "ToolFactory":
        """
        Get the global singleton instance of ToolFactory.

        This method provides access to a shared tool factory instance,
        creating it if it doesn't exist yet.

        Args:
        ----
            model: LLM model to use for code generation (if creating)
            executor: Executor to use for code generation (if creating)
            config: Configuration for the tool factory (if creating)

        Returns:
        -------
            ToolFactory: The global tool factory instance

        """
        instance = cls.get_instance(model=model, executor=executor, config=config)
        # Ensure we return an instance of the public API class
        if isinstance(instance, cls):
            return instance
        # If we got an instance of the internal class, create a new instance of the public class
        return cls(model=model, executor=executor, config=config)


# Re-export for public API
__all__ = [
    # Tool Factory
    "ToolFactory",
    "ToolFactoryConfig",
    "ToolSpecification",
    "ToolTemplate",
    "ToolValidator",
    "ValidationResult",
    # Security
    "CodeSigner",
    "SignatureVerifier",
    "SecurityLevel",
    "SigningLevel",
    # Sandboxing
    "SandboxType",
    "Sandbox",
    "DockerSandbox",
    "E2BSandbox",
    # Secure Hot Loading
    "SecureHotLoader",
    "SecureHotLoaderConfig",
    "create_secure_hot_loader",
]

# Update docstring for ToolTemplate
ToolTemplate.__doc__ = """
Template for generating tool code.

This class provides a template for generating tool code, with placeholders
for dynamic content.

Example:
```python
# Create a tool template
template = ToolTemplate(
    id="data_analysis_tool",
    name="Data Analysis Tool",
    description="A tool for data analysis",
    template_code="def {{function_name}}({{parameters}}):\\n    \\\"\\\"\\\"{{description}}\\\"\\\"\\\"\\n    {{code_body}}",
    required_parameters=["function_name", "parameters", "description", "code_body"],
    metadata={
        "category": "data_analysis",
        "complexity": "medium",
        "author": "Saplings",
    },
)
```
"""

# Update docstring for ToolSpecification
ToolSpecification.__doc__ = """
Specification for a tool to be generated.

This class provides a specification for a tool to be generated, including
the template to use and the parameters to fill in.

Example:
```python
# Create a tool specification
spec = ToolSpecification(
    id="correlation_analyzer",
    name="Correlation Analyzer",
    description="A tool for analyzing correlations in data",
    template_id="data_analysis_tool",
    parameters={
        "function_name": "analyze_correlations",
        "parameters": "data: dict[str, list[float]]",
        "description": "Analyze correlations between variables in the data",
        "code_body": "import numpy as np\\n\\n# Convert data to numpy arrays\\narrays = {k: np.array(v) for k, v in data.items()}\\n\\n# Calculate correlation matrix\\ncorr_matrix = {}\\nfor k1 in arrays:\\n    corr_matrix[k1] = {}\\n    for k2 in arrays:\\n        corr_matrix[k1][k2] = np.corrcoef(arrays[k1], arrays[k2])[0, 1]\\n\\nreturn corr_matrix",
    },
    metadata={
        "category": "data_analysis",
        "complexity": "medium",
        "author": "Saplings",
    },
)
```
"""

# Update docstrings for re-exported classes for better API documentation

# Update docstring for ToolValidator
ToolValidator.__doc__ = """
Validator for dynamically generated tools.

This class provides functionality for validating tool code to ensure it
meets security and quality standards.

Example:
```python
# Create a tool validator
validator = ToolValidator(
    config=ToolFactoryConfig(
        security_level=SecurityLevel.HIGH,
    )
)

# Validate code
validation_result = validator.validate(code)
if validation_result.is_valid:
    print("Code is valid!")
else:
    print(f"Code is invalid: {validation_result.error_message}")

# Check warnings
for warning in validation_result.warnings:
    print(f"Warning: {warning}")
```
"""

# Update docstring for ValidationResult
ValidationResult.__doc__ = """
Result of a tool code validation.

This class provides the result of a tool code validation, including
whether the code is valid, any error messages, and warnings.

Example:
```python
# Check validation result
if validation_result.is_valid:
    print("Code is valid!")
else:
    print(f"Code is invalid: {validation_result.error_message}")

# Check warnings
for warning in validation_result.warnings:
    print(f"Warning: {warning}")
```
"""

# Update docstring for ToolFactory
ToolFactory.__doc__ = """
Factory for dynamic tool synthesis.

This class provides functionality for dynamically creating tools based on
specifications and templates, with security checks and sandboxing.

Example:
```python
# Create a tool factory
tool_factory = ToolFactory(
    model=model,
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level=SecurityLevel.HIGH,
        sandbox_type=SandboxType.DOCKER,
    )
)

# Create a tool
tool_class = await tool_factory.create_tool(
    spec=ToolSpecification(
        id="data_analyzer",
        name="Data Analyzer",
        description="A tool for analyzing data",
        template_id="python_tool",
        parameters={
            "function_name": "analyze_data",
            "parameters": "data: dict",
            "description": "Analyze data and return insights",
        }
    )
)
```
"""

# Update docstring for SecureHotLoader
SecureHotLoader.__doc__ = """
Secure hot loading system for tools.

This class provides functionality for securely loading and executing dynamically
created tools, with sandboxing to prevent malicious code execution.

Example:
```python
# Create a secure hot loader
hot_loader = SecureHotLoader(
    config=SecureHotLoaderConfig(
        watch_directories=["./tools"],
        auto_reload=True,
        enable_sandboxing=True,
        sandbox_type=SandboxType.DOCKER,
    )
)

# Load a tool
tool_class = hot_loader.load_tool(tool_class)

# Get all loaded tools
tools = hot_loader.get_tools()
```
"""

# Update docstring for ToolFactoryConfig
ToolFactoryConfig.__doc__ = """
Configuration for the tool factory.

This class provides configuration options for the tool factory, including:
- Output directory for generated tools
- Security level for code validation
- Signing level for code signing
- Sandbox type for secure execution
- Docker image for Docker sandbox
- E2B API key for E2B sandbox
- Timeout settings for sandboxed execution
- Metadata for additional configuration

Example:
```python
# Create a tool factory config
config = ToolFactoryConfig(
    output_dir="./tools",
    security_level=SecurityLevel.HIGH,
    signing_level=SigningLevel.ADVANCED,
    signing_key_path="./keys/private_key.pem",
    sandbox_type=SandboxType.DOCKER,
    docker_image="python:3.9-slim",
    timeout_seconds=30,
    metadata={
        "allowed_imports": ["numpy", "pandas"],
        "blocked_imports": ["os", "subprocess", "sys"],
        "resource_limits": {
            "memory_mb": 512,
            "cpu_seconds": 30,
            "file_size_kb": 1024,
        },
    },
)
```
"""

# Update docstring for SecureHotLoaderConfig
SecureHotLoaderConfig.__doc__ = """
Configuration for the secure hot loader.

This class provides configuration options for the secure hot loader, including:
- Watch directories for auto-loading tools
- Auto-reload settings
- Sandboxing options
- Security settings
- Resource limits
- Import restrictions

Example:
```python
# Create a secure hot loader config
config = SecureHotLoaderConfig(
    watch_directories=["./tools"],
    auto_reload=True,
    reload_interval=1.0,
    enable_sandboxing=True,
    sandbox_type=SandboxType.DOCKER,
    docker_image="python:3.9-slim",
    timeout_seconds=30,
    allowed_imports=["numpy", "pandas"],
    blocked_imports=["os", "subprocess", "sys"],
    resource_limits={
        "memory_mb": 512,
        "cpu_seconds": 30,
        "file_size_kb": 1024,
    },
)
```
"""
