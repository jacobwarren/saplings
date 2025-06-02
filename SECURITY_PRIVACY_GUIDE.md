# Security & Privacy Guide

This guide covers Saplings' security architecture, privacy protection mechanisms, and best practices for secure deployment.

## Table of Contents

- [Security Architecture](#security-architecture)
- [Privacy Protection](#privacy-protection)
- [Input Sanitization](#input-sanitization)
- [Tool Security](#tool-security)
- [Memory & Data Security](#memory--data-security)
- [Authentication & Authorization](#authentication--authorization)
- [Secure Development](#secure-development)
- [Best Practices](#best-practices)

## Security Architecture

### Core Security Components

```python
from saplings.api.security import Sanitizer, RedactingFilter
from saplings.api.tool_factory import SecurityLevel, ToolValidator
from saplings.api.memory import MemoryConfig, PrivacyLevel

# Input sanitization
sanitizer = Sanitizer()
sanitized_input = sanitizer.sanitize(user_input)

# Content filtering
from saplings.api.security import install_global_filter, redact

# Install global redacting filter
install_global_filter()

# Manual redaction
redacted_text = redact("My password is secret123", ["password"])
```

### Security Levels

The framework supports different security levels:

```python
from saplings.api.tool_factory import SecurityLevel

# Configure tool factory security
config = ToolFactoryConfig(
    security_level=SecurityLevel.HIGH,  # LOW, MEDIUM, HIGH
    enable_sandboxing=True,
    timeout_seconds=30
)
```

## Privacy Protection

### Memory Privacy

Saplings provides privacy protection for sensitive data in memory:

```python
from saplings.api.memory import MemoryConfig, PrivacyLevel
from saplings.plugins.memory_stores import SecureMemoryStore

# Configure secure memory with privacy protection
memory_config = MemoryConfig.secure()
memory_config.secure_store.privacy_level = PrivacyLevel.HASH_AND_DP
memory_config.secure_store.hash_salt = "your-secure-salt"
memory_config.secure_store.dp_epsilon = 1.0

# Use secure memory store
secure_store = SecureMemoryStore(memory_config)
```

### Privacy Levels

```python
from saplings.memory._internal.config import PrivacyLevel

# Available privacy levels:
# - PrivacyLevel.NONE: No privacy measures
# - PrivacyLevel.HASH_ONLY: Hash document IDs and metadata
# - PrivacyLevel.HASH_AND_DP: Hash + differential privacy noise
```

### Differential Privacy

```python
# Configure differential privacy parameters
memory_config.secure_store.dp_epsilon = 1.0      # Privacy budget
memory_config.secure_store.dp_delta = 1e-5       # Privacy parameter
memory_config.secure_store.dp_sensitivity = 0.1  # Sensitivity
```

## Input Sanitization

### Text Sanitization

```python
from saplings.api.security import Sanitizer, sanitize

# Create sanitizer
sanitizer = Sanitizer()

# Sanitize user input
user_input = "Execute rm -rf / command"
safe_input = sanitizer.sanitize(user_input)

# Direct sanitization
safe_text = sanitize("Dangerous input with <script>alert('xss')</script>")
```

### Content Redaction

```python
from saplings.api.security import redact, RedactingFilter

# Redact sensitive information
text = "My API key is sk-1234567890abcdef"
redacted = redact(text, patterns=["api_key", "password", "secret"])

# Install global redacting filter
redacting_filter = RedactingFilter(
    patterns=["password", "api_key", "secret", "token"]
)
```

### Import Hook Security

```python
from saplings.api.security import install_import_hook

# Install import hook to prevent dangerous imports
install_import_hook()

# This will now be blocked or logged
# import os  # Potentially dangerous
```

## Tool Security

### Tool Validation

```python
from saplings.api.tool_factory import ToolValidator, SecurityLevel

# Create validator with high security
validator = ToolValidator(config=ToolFactoryConfig(
    security_level=SecurityLevel.HIGH
))

# Validate tool code
code = """
def safe_calculator(a, b):
    return a + b
"""

result = validator.validate(code)
if result.is_valid:
    print("Tool is secure")
else:
    print(f"Security issues: {result.error_message}")
```

### Sandboxed Execution

```python
from saplings.api.tool_factory import DockerSandbox, SandboxType

# Configure Docker sandbox
config = ToolFactoryConfig(
    sandbox_type=SandboxType.DOCKER,
    docker_image="python:3.9-slim",
    timeout_seconds=30
)

# Create sandbox
sandbox = DockerSandbox(config)

# Execute code safely
result = await sandbox.execute(
    code=safe_code,
    function_name="calculate",
    args=[2, 3],
    kwargs={}
)
```

### Function Authorization

```python
from saplings.core._internal.function_authorization import (
    AuthorizationLevel,
    requires_level,
    set_current_level
)

# Protect sensitive functions
@requires_level(AuthorizationLevel.ADMIN)
def admin_function():
    return "Admin only"

# Set authorization level
set_current_level(AuthorizationLevel.USER)

# This will raise PermissionError
# admin_function()
```

## Memory & Data Security

### Secure Document Storage

```python
from saplings.api.memory import Document, DocumentMetadata
from saplings.plugins.memory_stores import SecureMemoryStore

# Create secure memory store
secure_memory = SecureMemoryStore(MemoryConfig.secure())

# Add document (automatically secured)
doc = Document(
    content="Sensitive information",
    metadata=DocumentMetadata(
        source="/path/to/sensitive/file",
        author="admin@company.com"
    )
)

secure_memory.add_document(doc)
```

### Encrypted Storage

```python
# Secure memory configuration with encryption-like privacy
config = MemoryConfig.secure()
config.secure_store.privacy_level = PrivacyLevel.HASH_AND_DP
config.secure_store.hash_salt = "unique-salt-per-deployment"
```

### Access Control

```python
from saplings.core._internal.function_authorization import (
    FunctionAuthorizer,
    AuthorizationLevel
)

# Configure function-level access control
authorizer = FunctionAuthorizer()
authorizer.set_function_level("sensitive_operation", AuthorizationLevel.ADMIN)
authorizer.set_current_level(AuthorizationLevel.USER)

# Check authorization
if authorizer.is_authorized("sensitive_operation"):
    # Perform operation
    pass
else:
    raise PermissionError("Insufficient privileges")
```

## Authentication & Authorization

### Function-Level Authorization

```python
from saplings.core._internal.function_authorization import (
    AuthorizationLevel,
    function_authorizer,
    requires_level
)

# Set up authorization levels for different functions
function_authorizer.set_function_level("read_data", AuthorizationLevel.USER)
function_authorizer.set_function_level("write_data", AuthorizationLevel.ADMIN)
function_authorizer.set_function_level("delete_data", AuthorizationLevel.SYSTEM)

# Decorate functions with required levels
@requires_level(AuthorizationLevel.ADMIN)
def sensitive_operation():
    return "Admin operation completed"

# Set current authorization level
function_authorizer.set_current_level(AuthorizationLevel.USER)
```

### Group-Based Authorization

```python
# Set authorization for function groups
function_authorizer.set_group_level("data_operations", AuthorizationLevel.USER)
function_authorizer.set_group_level("system_operations", AuthorizationLevel.ADMIN)

# Get authorized functions for current level
authorized = function_authorizer.get_authorized_functions()
```

## Secure Development

### Code Signing

```python
from saplings.api.tool_factory import CodeSigner, SigningLevel

# Configure code signing
config = ToolFactoryConfig(
    signing_level=SigningLevel.BASIC,  # NONE, BASIC, ADVANCED
    signing_key_path="path/to/signing/key"
)

# Sign code
signer = CodeSigner(config)
signature = signer.sign("def secure_function(): pass")
```

### Secure Hot Loading

```python
from saplings.api.tool_factory import SecureHotLoader, SecureHotLoaderConfig

# Configure secure hot loading
config = SecureHotLoaderConfig(
    security_level=SecurityLevel.HIGH,
    enable_sandboxing=True,
    sandbox_type=SandboxType.DOCKER
)

# Create secure hot loader
loader = SecureHotLoader(config)
```

### Validation Pipeline

```python
from saplings.api.tool_factory import ToolValidator, ValidationResult

# Multi-level validation
validator = ToolValidator(ToolFactoryConfig(
    security_level=SecurityLevel.HIGH
))

# Validate syntax, security, and quality
result: ValidationResult = validator.validate(tool_code)

if not result.is_valid:
    print(f"Validation failed: {result.error_message}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

## Best Practices

### 1. Input Validation

```python
# Always sanitize user inputs
from saplings.api.security import sanitize

def process_user_input(raw_input: str) -> str:
    # Sanitize before processing
    safe_input = sanitize(raw_input)
    
    # Additional validation
    if len(safe_input) > MAX_INPUT_LENGTH:
        raise ValueError("Input too long")
    
    return safe_input
```

### 2. Secure Configuration

```python
# Use secure defaults
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    enable_monitoring=True,  # Enable for security auditing
    memory_config=MemoryConfig.secure()  # Use secure memory
)
```

### 3. Tool Security

```python
# Validate all tools before use
from saplings.api.tools import validate_tool

def register_safe_tool(tool):
    # Validate tool before registration
    if validate_tool(tool):
        agent.register_tool(tool)
    else:
        raise ValueError("Tool failed security validation")
```

### 4. Error Handling

```python
try:
    result = agent.run(user_input)
except PermissionError as e:
    logger.warning(f"Authorization failed: {e}")
    # Handle gracefully without exposing internals
    return "Access denied"
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    return "Invalid input"
```

### 5. Logging & Monitoring

```python
import logging
from saplings.api.monitoring import TraceManager

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable monitoring for security auditing
trace_manager = TraceManager()
config.monitoring_service = trace_manager
```

### 6. Environment Security

```python
import os
from saplings.api.security import redact

# Secure environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not configured")

# Log configuration without exposing secrets
config_str = str(config)
safe_config = redact(config_str, ["api_key", "token", "password"])
logger.info(f"Agent configured: {safe_config}")
```

### 7. Secure Deployment

```python
# Production configuration example
production_config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    # Security settings
    enable_monitoring=True,
    memory_config=MemoryConfig.secure(),
    # Tool factory security
    tool_factory_config=ToolFactoryConfig(
        security_level=SecurityLevel.HIGH,
        enable_sandboxing=True,
        timeout_seconds=10
    )
)
```

## Compliance & Auditing

### Security Auditing

```python
from saplings.api.monitoring import TraceManager

# Enable comprehensive monitoring
trace_manager = TraceManager()

# Monitor all operations
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_monitoring_service(trace_manager) \
    .build()

# Review traces for security events
traces = trace_manager.get_traces()
for trace in traces:
    if trace.has_security_events():
        logger.warning(f"Security event in trace: {trace.id}")
```

### Compliance Features

- **Data Privacy**: Differential privacy and hashing for sensitive data
- **Access Control**: Function-level authorization
- **Audit Trails**: Comprehensive monitoring and tracing
- **Input Validation**: Multi-layer sanitization and validation
- **Secure Storage**: Encrypted memory storage with privacy protection

This security guide provides comprehensive coverage of Saplings' security features and best practices for building secure AI applications.