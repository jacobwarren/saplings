# Validator System

The Validator system in Saplings provides a flexible framework for validating prompts and outputs against specific criteria, ensuring quality, safety, and compliance.

## Overview

The Validator system consists of several key components:

- **Validator**: Base class for all validators
- **StaticValidator**: Validates prompts before execution
- **RuntimeValidator**: Validates outputs after execution
- **ValidatorRegistry**: Manages validators and orchestrates validation
- **ValidationResult**: Result of a validation operation
- **ValidatorConfig**: Configuration for the validator system
- **Built-in Validators**: Ready-to-use validators for common use cases

This system enables agents to validate their inputs and outputs, providing a foundation for quality control, safety, and compliance.

## Core Concepts

### Validator Types

The Validator system supports different types of validators:

- **Static Validators**: Run before execution and validate the prompt
- **Runtime Validators**: Run after execution and validate the output
- **Hybrid Validators**: Can run both before and after execution

Each type of validator serves a different purpose in the validation pipeline, allowing for comprehensive validation at different stages of the execution process.

### Validation Results

Validation results provide information about the outcome of a validation operation:

- **Status**: Whether the validation passed, failed, or encountered an error
- **Message**: A human-readable message explaining the result
- **Details**: Additional information about the validation
- **Metadata**: Information about the validation process itself

These results can be used to make decisions about whether to proceed with execution, retry with modifications, or take other actions.

### Plugin System

The Validator system is built on a plugin architecture, allowing for easy extension with custom validators:

- **Plugin Discovery**: Automatically discover validators from specified directories
- **Entry Points**: Load validators from Python entry points
- **Registration**: Manually register validators with the registry

This flexibility makes it easy to add new validators for specific use cases or integrate with existing validation systems.

## API Reference

### Validator

```python
class Validator(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the validator."""
        pass

    @property
    @abstractmethod
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the validator."""
        pass

    @abstractmethod
    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output."""
        pass
```

### StaticValidator

```python
class StaticValidator(Validator):
    @property
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        return ValidatorType.STATIC

    @abstractmethod
    async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
        """Validate a prompt."""
        pass

    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output (delegates to validate_prompt)."""
        return await self.validate_prompt(prompt, **kwargs)
```

### RuntimeValidator

```python
class RuntimeValidator(Validator):
    @property
    def validator_type(self) -> ValidatorType:
        """Type of the validator."""
        return ValidatorType.RUNTIME

    @abstractmethod
    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output."""
        pass

    async def validate(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output (delegates to validate_output)."""
        return await self.validate_output(output, prompt, **kwargs)
```

### ValidationResult

```python
class ValidationResult:
    def __init__(
        self,
        validator_id: str,
        status: ValidationStatus,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a validation result."""
        self.validator_id = validator_id
        self.status = status
        self.message = message
        self.details = details or {}
        self.metadata = metadata or {}
```

### ValidatorRegistry

```python
class ValidatorRegistry:
    def __init__(self):
        """Initialize the validator registry."""

    def configure(self, config: ValidatorConfig) -> None:
        """Configure the validator registry."""

    def register_validator(self, validator_class: Type[Validator]) -> None:
        """Register a validator class."""

    def get_validator(self, validator_id: str) -> Validator:
        """Get a validator by ID."""

    def list_validators(self) -> List[str]:
        """List all registered validator IDs."""

    def discover_validators(self) -> None:
        """Discover validators from plugins and entry points."""

    async def validate(
        self,
        output: str,
        prompt: str,
        validator_ids: Optional[List[str]] = None,
        validator_type: Optional[ValidatorType] = None,
        **kwargs,
    ) -> List[ValidationResult]:
        """Validate an output using the specified validators."""

    async def validate_with_validator(
        self,
        validator_id: str,
        output: str,
        prompt: str,
        **kwargs,
    ) -> ValidationResult:
        """Validate an output using a specific validator."""

    def reset_budget(self) -> None:
        """Reset the validation budget counter."""
```

### ValidatorConfig

```python
class ValidatorConfig(BaseModel):
    # General settings
    enabled: bool = True  # Whether validation is enabled
    fail_fast: bool = False  # Whether to stop validation on first failure

    # Plugin settings
    plugin_dirs: List[str] = []  # Directories to search for validator plugins
    use_entry_points: bool = True  # Whether to use entry points for validator discovery

    # Execution settings
    parallel_validation: bool = True  # Whether to run validators in parallel
    max_parallel_validators: int = 10  # Maximum number of validators to run in parallel

    # Timeout settings
    timeout_seconds: Optional[float] = None  # Timeout for validation in seconds

    # Budget settings
    enforce_budget: bool = False  # Whether to enforce budget constraints
    max_validations_per_session: int = 100  # Maximum number of validations per session
    max_validations_per_day: Optional[int] = None  # Maximum number of validations per day

    @classmethod
    def default(cls) -> "ValidatorConfig":
        """Create a default configuration."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ValidatorConfig":
        """Create a configuration from a dictionary."""
```

### Enums

```python
class ValidationStatus(str, Enum):
    """Status of a validation."""
    PASSED = "passed"  # Validation passed
    FAILED = "failed"  # Validation failed
    ERROR = "error"  # Validation encountered an error
    SKIPPED = "skipped"  # Validation was skipped

class ValidatorType(str, Enum):
    """Types of validators."""
    STATIC = "static"  # Static validators run before execution
    RUNTIME = "runtime"  # Runtime validators run during or after execution
    HYBRID = "hybrid"  # Hybrid validators can run both before and after execution
```

## Built-in Validators

### Basic Validators

#### LengthValidator

```python
class LengthValidator(RuntimeValidator):
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "characters",
    ):
        """
        Initialize the length validator.

        Args:
            min_length: Minimum length (inclusive)
            max_length: Maximum length (inclusive)
            unit: Unit of length ("characters", "words", "sentences")
        """
```

The LengthValidator checks if the output length is within the specified range, with support for different units of measurement.

#### KeywordValidator

```python
class KeywordValidator(RuntimeValidator):
    def __init__(
        self,
        required_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize the keyword validator.

        Args:
            required_keywords: Keywords that must be present in the output
            forbidden_keywords: Keywords that must not be present in the output
            case_sensitive: Whether the keyword matching is case-sensitive
        """
```

The KeywordValidator checks if the output contains required keywords and does not contain forbidden keywords.

#### SentimentValidator

```python
class SentimentValidator(RuntimeValidator):
    def __init__(
        self,
        min_sentiment: float = -1.0,
        max_sentiment: float = 1.0,
    ):
        """
        Initialize the sentiment validator.

        Args:
            min_sentiment: Minimum sentiment score (-1.0 to 1.0)
            max_sentiment: Maximum sentiment score (-1.0 to 1.0)
        """
```

The SentimentValidator checks if the output sentiment is within the specified range, using a simple sentiment analysis algorithm.

### Safety Validators

#### ProfanityValidator

```python
class ProfanityValidator(RuntimeValidator):
    def __init__(
        self,
        custom_profanity_list: Optional[List[str]] = None,
        threshold: float = 0.0,
    ):
        """
        Initialize the profanity validator.

        Args:
            custom_profanity_list: Custom list of profanity words
            threshold: Threshold for profanity detection (0.0 = any profanity fails)
        """
```

The ProfanityValidator checks if the output contains profanity, with support for custom profanity lists and thresholds.

#### PiiValidator

```python
class PiiValidator(RuntimeValidator):
    def __init__(
        self,
        check_emails: bool = True,
        check_phone_numbers: bool = True,
        check_credit_cards: bool = True,
        check_ssns: bool = True,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the PII validator.

        Args:
            check_emails: Whether to check for email addresses
            check_phone_numbers: Whether to check for phone numbers
            check_credit_cards: Whether to check for credit card numbers
            check_ssns: Whether to check for Social Security Numbers
            custom_patterns: Custom regex patterns to check for
        """
```

The PiiValidator checks if the output contains personally identifiable information (PII), with support for various types of PII and custom patterns.

### Plugin Validators

#### CodeValidator

```python
class CodeValidator(RuntimeValidator):
    def __init__(
        self,
        check_syntax: bool = True,
        check_security: bool = True,
        check_quality: bool = True,
        allowed_imports: Optional[List[str]] = None,
        forbidden_imports: Optional[List[str]] = None,
    ):
        """
        Initialize the code validator.

        Args:
            check_syntax: Whether to check for syntax errors
            check_security: Whether to check for security issues
            check_quality: Whether to check for code quality issues
            allowed_imports: List of allowed imports (if None, all imports are allowed)
            forbidden_imports: List of forbidden imports
        """
```

The CodeValidator checks if the output contains valid code, with support for syntax checking, security analysis, and code quality assessment.

#### FactualValidator

```python
class FactualValidator(RuntimeValidator):
    def __init__(
        self,
        knowledge_base: Optional[str] = None,
        threshold: float = 0.7,
    ):
        """
        Initialize the factual validator.

        Args:
            knowledge_base: Path to the knowledge base file
            threshold: Threshold for factual accuracy (0.0 to 1.0)
        """
```

The FactualValidator checks if the output is factually accurate, using a knowledge base for verification.

## Usage Examples

### Basic Usage

```python
from saplings.validator import ValidatorRegistry
from saplings.validator.validators import LengthValidator, KeywordValidator

# Create a validator registry
registry = ValidatorRegistry()

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)

# Validate an output
import asyncio
results = asyncio.run(registry.validate(
    output="This is a test output.",
    prompt="Generate a test output.",
))

# Check the results
for result in results:
    print(f"Validator: {result.validator_id}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print()
```

### Using Specific Validators

```python
from saplings.validator import ValidatorRegistry
from saplings.validator.validators import LengthValidator, KeywordValidator

# Create a validator registry
registry = ValidatorRegistry()

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)

# Create a custom length validator
length_validator = LengthValidator(min_length=10, max_length=100, unit="words")
registry.register_validator(lambda: length_validator)

# Create a custom keyword validator
keyword_validator = KeywordValidator(
    required_keywords=["important", "critical"],
    forbidden_keywords=["unnecessary", "optional"],
)
registry.register_validator(lambda: keyword_validator)

# Validate an output with specific validators
import asyncio
results = asyncio.run(registry.validate(
    output="This is an important and critical test output.",
    prompt="Generate a test output with specific keywords.",
    validator_ids=["length_validator", "keyword_validator"],
))

# Check the results
for result in results:
    print(f"Validator: {result.validator_id}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print()
```

### Using Validator Types

```python
from saplings.validator import ValidatorRegistry, ValidatorType
from saplings.validator.validators import LengthValidator, KeywordValidator

# Create a validator registry
registry = ValidatorRegistry()

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)

# Validate an output with runtime validators
import asyncio
results = asyncio.run(registry.validate(
    output="This is a test output.",
    prompt="Generate a test output.",
    validator_type=ValidatorType.RUNTIME,
))

# Check the results
for result in results:
    print(f"Validator: {result.validator_id}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print()
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.validator import ValidatorRegistry
from saplings.validator.validators import LengthValidator, KeywordValidator, ProfanityValidator

# Create a validator registry
registry = ValidatorRegistry()

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)
registry.register_validator(ProfanityValidator)

# Create an agent with the validator registry
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        validator_registry=registry,
    )
)

# Run a task
import asyncio
result = asyncio.run(agent.run(
    "Explain the concept of graph-based memory and its advantages."
))

# The agent will validate its output before returning it
print(result)
```

## Advanced Features

### Parallel Validation

The ValidatorRegistry can run validators in parallel for improved performance:

```python
from saplings.validator import ValidatorRegistry, ValidatorConfig
from saplings.validator.validators import LengthValidator, KeywordValidator, ProfanityValidator, PiiValidator

# Create a validator registry
registry = ValidatorRegistry()

# Configure for parallel validation
config = ValidatorConfig(
    parallel_validation=True,
    max_parallel_validators=10,
)
registry.configure(config)

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)
registry.register_validator(ProfanityValidator)
registry.register_validator(PiiValidator)

# Validate an output with all validators in parallel
import asyncio
results = asyncio.run(registry.validate(
    output="This is a test output.",
    prompt="Generate a test output.",
))

# Check the results
for result in results:
    print(f"Validator: {result.validator_id}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print()
```

### Budget Enforcement

The ValidatorRegistry can enforce budget constraints to prevent excessive validation:

```python
from saplings.validator import ValidatorRegistry, ValidatorConfig
from saplings.validator.validators import LengthValidator, KeywordValidator

# Create a validator registry
registry = ValidatorRegistry()

# Configure with budget constraints
config = ValidatorConfig(
    enforce_budget=True,
    max_validations_per_session=10,
    max_validations_per_day=100,
)
registry.configure(config)

# Register validators
registry.register_validator(LengthValidator)
registry.register_validator(KeywordValidator)

# Validate outputs until the budget is exhausted
import asyncio

for i in range(15):  # Try to validate 15 times
    print(f"Validation {i+1}:")
    results = asyncio.run(registry.validate(
        output=f"This is test output {i+1}.",
        prompt=f"Generate test output {i+1}.",
    ))

    if not results:
        print("Validation skipped due to budget constraints.")
        break

    for result in results:
        print(f"  Validator: {result.validator_id}")
        print(f"  Status: {result.status}")
        print(f"  Message: {result.message}")
        print()
```

### Custom Validators

You can create custom validators by extending the `StaticValidator` or `RuntimeValidator` classes:

```python
from saplings.validator import RuntimeValidator, ValidationResult, ValidationStatus

class CustomValidator(RuntimeValidator):
    @property
    def id(self) -> str:
        """ID of the validator."""
        return "custom_validator"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "A custom validator for demonstration purposes."

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output."""
        # Custom validation logic
        if "custom_keyword" in output.lower():
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message="Output contains the custom keyword.",
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message="Output does not contain the custom keyword.",
                details={"missing_keyword": "custom_keyword"},
            )
```

### Plugin Discovery

The ValidatorRegistry can discover validators from plugins and entry points:

```python
from saplings.validator import ValidatorRegistry, ValidatorConfig

# Create a validator registry
registry = ValidatorRegistry()

# Configure for plugin discovery
config = ValidatorConfig(
    plugin_dirs=["./plugins", "./custom_validators"],
    use_entry_points=True,
)
registry.configure(config)

# Discover validators
registry.discover_validators()

# List discovered validators
validator_ids = registry.list_validators()
print(f"Discovered validators: {validator_ids}")
```

## Implementation Details

### Validation Process

The validation process works as follows:

1. **Validator Selection**: Select validators based on ID, type, or use all registered validators
2. **Parallel Execution**: If enabled, run validators in parallel up to the maximum concurrency
3. **Budget Enforcement**: If enabled, check if the validation budget has been exceeded
4. **Timeout Handling**: If configured, apply a timeout to each validation operation
5. **Result Collection**: Collect and return the results from all validators

### Plugin Discovery

The plugin discovery process works as follows:

1. **Directory Scanning**: Scan specified directories for Python files containing validator classes
2. **Entry Point Loading**: Load validators from Python entry points
3. **Plugin Registration**: Register discovered validators with the registry

### Budget Enforcement

The budget enforcement process works as follows:

1. **Session Budget**: Track the number of validations in the current session
2. **Daily Budget**: Track the number of validations in the current day
3. **Budget Checking**: Before validation, check if the budget has been exceeded
4. **Budget Reset**: Reset the budget counter when requested

## Extension Points

The Validator system is designed to be extensible:

### Custom Validators

You can create custom validators by extending the `StaticValidator` or `RuntimeValidator` classes:

```python
from saplings.validator import StaticValidator, ValidationResult, ValidationStatus

class PromptLengthValidator(StaticValidator):
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

    @property
    def id(self) -> str:
        return "prompt_length_validator"

    @property
    def description(self) -> str:
        return f"Validates that the prompt is not longer than {self.max_length} characters."

    async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
        if len(prompt) > self.max_length:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message=f"Prompt is too long ({len(prompt)} > {self.max_length} characters).",
                details={"prompt_length": len(prompt), "max_length": self.max_length},
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message=f"Prompt length is within limits ({len(prompt)} <= {self.max_length} characters).",
                details={"prompt_length": len(prompt), "max_length": self.max_length},
            )
```

### Custom Registry

You can create a custom validator registry by extending the `ValidatorRegistry` class:

```python
from saplings.validator import ValidatorRegistry, ValidationResult

class CustomValidatorRegistry(ValidatorRegistry):
    async def validate(self, output, prompt, validator_ids=None, validator_type=None, **kwargs):
        # Add custom logic before validation
        print(f"Validating output: {output[:50]}...")

        # Call the parent method
        results = await super().validate(output, prompt, validator_ids, validator_type, **kwargs)

        # Add custom logic after validation
        passed = all(result.status == "passed" for result in results)
        print(f"Validation {'passed' if passed else 'failed'}")

        return results
```

### Custom Validation Status

You can create custom validation statuses by extending the `ValidationStatus` enum:

```python
from enum import Enum
from saplings.validator import ValidationStatus

class CustomValidationStatus(str, Enum):
    WARNING = "warning"  # Validation passed with warnings
    PARTIAL = "partial"  # Validation partially passed

    # Include the original statuses
    PASSED = ValidationStatus.PASSED
    FAILED = ValidationStatus.FAILED
    ERROR = ValidationStatus.ERROR
    SKIPPED = ValidationStatus.SKIPPED
```

## Conclusion

The Validator system in Saplings provides a flexible framework for validating prompts and outputs against specific criteria, ensuring quality, safety, and compliance. By using validators, you can ensure that your agents produce high-quality, safe, and compliant outputs, and catch potential issues before they reach users.
