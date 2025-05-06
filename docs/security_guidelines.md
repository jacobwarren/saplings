# Security Guidelines

The Security system in Saplings provides comprehensive protection against common security threats and vulnerabilities, ensuring that agents operate safely and securely.

## Overview

The Security system consists of several key components:

- **Prompt Sanitization**: Prevents injection attacks by sanitizing user inputs
- **Tool Validation**: Ensures that dynamically generated tools meet security standards
- **Sandboxing**: Isolates tool execution in secure environments
- **Code Signing**: Verifies the integrity and authenticity of code
- **Function Authorization**: Controls access to sensitive functions
- **Log Filtering**: Redacts sensitive information from logs

This system provides multiple layers of defense to protect against various security threats while maintaining flexibility and usability.

## Core Concepts

### Security Levels

Saplings defines different security levels that can be applied to various components:

```python
class SecurityLevel(str, Enum):
    """Security level for tool generation."""
    LOW = "low"  # Basic security checks
    MEDIUM = "medium"  # Standard security checks
    HIGH = "high"  # Strict security checks
```

### Sandbox Types

For tool execution, Saplings provides different sandboxing options:

```python
class SandboxType(str, Enum):
    """Type of sandbox for tool execution."""
    NONE = "none"  # No sandboxing
    DOCKER = "docker"  # Docker-based sandbox
    E2B = "e2b"  # E2B-based sandbox
```

### Authorization Levels

Function authorization is controlled through authorization levels:

```python
class AuthorizationLevel(Enum):
    """Authorization level for function calls."""
    PUBLIC = 0  # Anyone can call
    USER = 1  # Authenticated users can call
    ADMIN = 2  # Administrators can call
    SYSTEM = 3  # System only
```

## Security Components

### Prompt Sanitization

The prompt sanitizer removes dangerous substrings from user inputs:

```python
def sanitize_prompt(
    raw: str,
    max_len: int = 8_192,
    remove_urls: bool = True,
    remove_shell_metacharacters: bool = True,
    remove_sql_patterns: bool = True,
    remove_path_traversal: bool = True,
    remove_html: bool = True,
    custom_patterns: Optional[List[Pattern[str]]] = None
) -> str:
    """
    Remove dangerous substrings and truncate at max_len.

    Args:
        raw: The raw input string to sanitize
        max_len: Maximum length of the output string
        remove_urls: Whether to redact URLs
        remove_shell_metacharacters: Whether to remove shell metacharacters
        remove_sql_patterns: Whether to remove SQL injection patterns
        remove_path_traversal: Whether to remove path traversal patterns
        remove_html: Whether to remove HTML and script tags
        custom_patterns: Additional custom patterns to remove

    Returns:
        Sanitized string
    """
```

### Tool Validation

The tool validator checks dynamically generated code for security issues:

```python
class ToolValidator:
    """
    Validates tool code for security and correctness.

    This class checks tool code for security issues, syntax errors,
    and other potential problems before allowing it to be executed.
    """

    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the tool validator.

        Args:
            config: Configuration for the validator
        """

    def validate(self, code: str) -> ValidationResult:
        """
        Validate tool code.

        Args:
            code: Code to validate

        Returns:
            ValidationResult: Result of the validation
        """
```

### Sandboxing

Saplings provides several sandbox implementations for secure tool execution:

```python
class Sandbox:
    """
    Base class for sandbox execution environments.

    This class provides a common interface for different sandbox implementations.
    """

    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the sandbox.

        Args:
            config: Configuration for the sandbox
        """

    async def execute(
        self,
        code: str,
        function_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Execute code in the sandbox.

        Args:
            code: Code to execute
            function_name: Name of the function to call
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

        Returns:
            Any: Result of the function call
        """
```

Available sandbox implementations:

- **LocalSandbox**: Executes code in the local Python interpreter with minimal isolation
- **DockerSandbox**: Executes code in a Docker container with network isolation
- **E2BSandbox**: Executes code in the E2B cloud sandbox

### Function Authorization

The function authorizer controls access to sensitive functions:

```python
class FunctionAuthorizer:
    """Utility for authorizing function calls."""

    def __init__(self):
        """Initialize the function authorizer."""

    def set_function_level(self, name: str, level: AuthorizationLevel) -> None:
        """
        Set the authorization level for a function.

        Args:
            name: Name of the function
            level: Authorization level
        """

    def set_current_level(self, level: AuthorizationLevel) -> None:
        """
        Set the current authorization level.

        Args:
            level: Authorization level
        """

    def is_authorized(self, name: str) -> bool:
        """
        Check if the current level is authorized to call a function.

        Args:
            name: Name of the function

        Returns:
            bool: True if authorized, False otherwise
        """

    def authorize_function_call(self, name: str) -> None:
        """
        Authorize a function call.

        Args:
            name: Name of the function

        Raises:
            PermissionError: If not authorized
        """
```

### Log Filtering

The log filter redacts sensitive information from logs:

```python
def install_global_filter(
    patterns: Optional[List[Pattern[str]]] = None,
    replacement: str = "****"
) -> None:
    """
    Install the redacting filter globally on the root logger.

    Args:
        patterns: Additional patterns to redact beyond the defaults
        replacement: String to use as replacement for redacted content
    """
```

## Best Practices

### Secure Tool Development

When developing tools for Saplings, follow these guidelines:

1. **Validate Inputs**: Always validate and sanitize inputs to prevent injection attacks
2. **Limit Permissions**: Use the principle of least privilege when accessing resources
3. **Avoid Dangerous Functions**: Avoid using functions that could be exploited (e.g., `eval`, `exec`, `os.system`)
4. **Use Sandboxing**: Run untrusted code in a sandbox to isolate it from the rest of the system
5. **Handle Errors Securely**: Avoid exposing sensitive information in error messages

### Secure Agent Configuration

When configuring agents, follow these guidelines:

1. **Set Appropriate Security Levels**: Use the highest security level that meets your requirements
2. **Enable Sandboxing**: Use sandboxing for tool execution, especially for dynamically generated tools
3. **Limit Tool Access**: Only provide agents with the tools they need to complete their tasks
4. **Monitor Agent Activity**: Use the monitoring system to track agent activity and detect suspicious behavior
5. **Validate Outputs**: Use the validation system to ensure that agent outputs meet security standards

### Secure Deployment

When deploying Saplings in production, follow these guidelines:

1. **Use HTTPS**: Always use HTTPS for communication between components
2. **Secure API Keys**: Store API keys and other secrets securely (e.g., using environment variables or a secrets manager)
3. **Implement Rate Limiting**: Limit the number of requests to prevent abuse
4. **Use Authentication**: Require authentication for accessing sensitive functionality
5. **Regular Updates**: Keep Saplings and its dependencies up to date to address security vulnerabilities

## Example: Secure Tool Factory Configuration

```python
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, SecurityLevel, SandboxType
)

# Create a tool factory with high security
tool_factory = ToolFactory(
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level=SecurityLevel.HIGH,
        sandbox_type=SandboxType.DOCKER,
        docker_image="python:3.9-slim",
        sandbox_timeout=30,
    )
)
```

## Example: Secure Hot Loading

```python
from saplings.integration import (
    SecureHotLoader, SecureHotLoaderConfig
)

# Create a secure hot loader
hot_loader = SecureHotLoader(
    config=SecureHotLoaderConfig(
        enable_sandboxing=True,
        sandbox_type="docker",
        docker_image="python:3.9-slim",
        sandbox_timeout=30,
        allowed_imports=["numpy", "pandas"],
        blocked_imports=["os", "subprocess", "sys"],
        resource_limits={
            "memory_mb": 512,
            "cpu_seconds": 30,
            "file_size_kb": 1024,
        },
    )
)

# Load a tool securely
tool_class = hot_loader.load_tool("path/to/tool.py")
```

## Example: Function Authorization

```python
from saplings.core.function_authorization import (
    function_authorizer, AuthorizationLevel, requires_level
)

# Define a function that requires admin authorization
@requires_level(AuthorizationLevel.ADMIN)
def sensitive_operation():
    """Perform a sensitive operation."""
    # Implementation

# Set the current authorization level
function_authorizer.set_current_level(AuthorizationLevel.USER)

try:
    # This will raise a PermissionError
    sensitive_operation()
except PermissionError as e:
    print(f"Error: {e}")

# Set the current authorization level to admin
function_authorizer.set_current_level(AuthorizationLevel.ADMIN)

# Now this will succeed
sensitive_operation()
```

## Security Considerations for Third-Party Integrations

When integrating with third-party services, follow these guidelines:

1. **Validate Responses**: Always validate responses from third-party services
2. **Limit Access**: Only provide the minimum required access to third-party services
3. **Use Timeouts**: Set appropriate timeouts for requests to third-party services
4. **Handle Errors**: Implement robust error handling for third-party service failures
5. **Monitor Usage**: Monitor usage of third-party services to detect unusual patterns

## Conclusion

Security is a critical aspect of agent development. Saplings provides a comprehensive security system that helps protect against common threats while maintaining flexibility and usability. By following the guidelines and best practices outlined in this document, you can build secure and reliable agent applications.
