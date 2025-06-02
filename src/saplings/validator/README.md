# Saplings Validators

This package provides validation functionality for Saplings agents.

## API Structure

The validators module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.validator` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the validators, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import (
    Validator,
    ValidationResult,
    LengthValidator,
    KeywordValidator,
    ExecutionValidator
)

# Alternative: Import directly from the API module
from saplings.api.validator import (
    Validator,
    ValidationResult,
    LengthValidator,
    KeywordValidator,
    ExecutionValidator
)
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.validator._internal import Validator  # Wrong
```

## Available Validators

The following validators are available:

- `Validator`: Base class for all validators
- `StaticValidator`: Base class for validators that run at compile time
- `RuntimeValidator`: Base class for validators that run at runtime
- `LengthValidator`: Validator that checks the length of a response
- `KeywordValidator`: Validator that checks for required keywords in a response
- `ExecutionValidator`: Validator that checks if a response can be executed

## Validator Registry

Validators can be registered with the global registry:

```python
from saplings import ValidatorRegistry, Validator

# Create a custom validator
class MyValidator(Validator):
    def validate(self, response, **kwargs):
        # Validation logic here
        return ValidationResult(...)

# Get the registry
registry = ValidatorRegistry()

# Register the validator
registry.register_validator(MyValidator)
```

## Validation Results

Validation results include a status and a message:

```python
from saplings import ValidationResult, ValidationStatus

# Create a validation result
result = ValidationResult(
    status=ValidationStatus.SUCCESS,
    message="Validation passed",
    metadata={"score": 0.95}
)
```

## Implementation Details

The validator implementations are located in the `_internal` directory:

- `_internal/validator.py`: Base validator classes
- `_internal/registry.py`: Validator registry implementation
- `_internal/result.py`: Validation result implementation
- `_internal/config.py`: Validator configuration
- `_internal/validators/`: Specific validator implementations

These internal implementations are wrapped by the public API in `saplings.api.validator` to provide stability annotations and a consistent interface.
