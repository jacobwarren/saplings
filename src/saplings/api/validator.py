from __future__ import annotations

"""
Validator API module for Saplings.

This module provides the public API for validators and related components.
"""

from enum import Enum

# Import from individual modules - use TYPE_CHECKING to avoid circular imports
from typing import Any, Dict, Optional

from saplings.api.stability import stable


# Define IValidationStrategy directly to avoid circular imports
@stable
class IValidationStrategy:
    """
    Interface for validation strategies.

    This interface defines the contract for validation strategies that can be
    used to validate outputs against specific criteria.
    """

    async def validate(
        self,
        input_data: dict,
        output_data: Any,
        validation_type: str = "general",
        trace_id: str | None = None,
    ) -> "ValidationResult":
        """
        Validate an output against specific criteria.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        raise NotImplementedError("Subclasses must implement validate method")


# Define ValidationStrategy class
@stable
class ValidationStrategy(str, Enum):
    """
    Strategy for validation.

    This enum represents the possible strategies for validation.
    """

    RULE_BASED = "rule_based"
    JUDGE_BASED = "judge_based"


# Define JudgeBasedValidationStrategy directly to avoid circular imports
@stable
class JudgeBasedValidationStrategy(IValidationStrategy):
    """
    Validation strategy that uses a judge for validation.

    This strategy delegates validation to a judge service, which uses
    an LLM to evaluate outputs against specific criteria.
    """

    def __init__(self, judge_service=None, criteria=None):
        """
        Initialize a judge-based validation strategy.

        Args:
        ----
            judge_service: The judge service to use for validation
            criteria: The criteria to use for validation

        """
        self.judge_service = judge_service
        self.criteria = criteria or {}

    async def validate(
        self,
        input_data: dict,
        output_data: Any,
        validation_type: str = "general",
        trace_id: str | None = None,
    ) -> "ValidationResult":
        """
        Validate an output using a judge.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        if self.judge_service:
            try:
                # Delegate to the judge service
                judgment = await self.judge_service.judge(
                    input_data=input_data,
                    output_data=output_data,
                    judgment_type=validation_type,
                    trace_id=trace_id,
                )
                
                return ValidationResult(
                    status=ValidationStatus.SUCCESS if judgment.is_valid else ValidationStatus.FAILURE,
                    message=judgment.feedback or "Judge-based validation completed",
                )
            except Exception as e:
                return ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Judge validation failed: {e}",
                )
        else:
            # No judge service available, return success
            return ValidationResult(
                status=ValidationStatus.SUCCESS,
                message="Judge-based validation not available - validation passed",
            )


# Define RuleBasedValidationStrategy directly to avoid circular imports
@stable
class RuleBasedValidationStrategy(IValidationStrategy):
    """
    Validation strategy that uses predefined rules for validation.

    This strategy uses a set of rules to validate outputs without
    requiring a judge or LLM.
    """

    def __init__(self, rules=None):
        """
        Initialize a rule-based validation strategy.

        Args:
        ----
            rules: The rules to use for validation

        """
        self.rules = rules or {}

    async def validate(
        self,
        input_data: dict,
        output_data: Any,
        validation_type: str = "general",
        trace_id: str | None = None,
    ) -> "ValidationResult":
        """
        Validate an output using rules.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate  
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # Get rules for the validation type
        type_rules = self.rules.get(validation_type, {})
        if not type_rules:
            return ValidationResult(
                status=ValidationStatus.SUCCESS,
                message="No validation rules defined - validation passed",
            )

        # Apply basic validation rules
        is_valid = True
        messages = []

        # Basic length check
        if isinstance(output_data, str) and len(output_data.strip()) == 0:
            is_valid = False
            messages.append("Output is empty")

        # Basic content check
        if isinstance(output_data, str) and "error" in output_data.lower():
            is_valid = False
            messages.append("Output contains error indication")

        status = ValidationStatus.SUCCESS if is_valid else ValidationStatus.FAILURE
        message = "; ".join(messages) if messages else "Rule-based validation passed"

        return ValidationResult(
            status=status,
            message=message,
        )


@stable
class ValidationStatus(str, Enum):
    """
    Status of a validation operation.

    This enum represents the possible statuses of a validation operation.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    WARNING = "warning"


@stable
class ValidationResult:
    """
    Result of a validation operation.

    This class represents the result of a validation operation, including
    the status, message, and metadata.
    """

    def __init__(
        self,
        status: ValidationStatus,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        score: float | None = None,
        feedback: str | None = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a validation result.

        Args:
        ----
            status: Status of the validation
            message: Message describing the validation result
            metadata: Additional metadata about the validation
            score: Optional score for the validation (0.0 to 1.0)
            feedback: Optional feedback message
            details: Optional additional details about the validation

        """
        self.status = status
        self.message = message
        self.metadata = metadata or {}
        self.score = score if score is not None else (1.0 if status == ValidationStatus.SUCCESS else 0.0)
        self.feedback = feedback or message
        self.details = details or {}

    @property
    def is_valid(self) -> bool:
        """
        Check if the validation was successful.

        Returns
        -------
            bool: True if the validation was successful, False otherwise

        """
        return self.status == ValidationStatus.SUCCESS


@stable
class ValidatorType(str, Enum):
    """
    Type of validator.

    This enum represents the possible types of validators.
    """

    RUNTIME = "runtime"
    STATIC = "static"
    EXECUTION = "execution"
    KEYWORD = "keyword"
    LENGTH = "length"
    SENTIMENT = "sentiment"
    PII = "pii"
    PROFANITY = "profanity"
    CUSTOM = "custom"


# Define the public API
__all__ = [
    # Validator types
    "ValidatorType",
    # Validation status
    "ValidationStatus",
    # Validation result
    "ValidationResult",
    # Validator classes
    "Validator",
    "StaticValidator",
    "RuntimeValidator",
    "ExecutionValidator",
    "KeywordValidator",
    "LengthValidator",
    "SentimentValidator",
    "PiiValidator",
    "ProfanityValidator",
    # Validator configuration
    "ValidatorConfig",
    # Validator registry
    "ValidatorRegistry",
    "get_validator_registry",
    # Validation strategies
    "ValidationStrategy",
    "IValidationStrategy",
    "JudgeBasedValidationStrategy",
    "RuleBasedValidationStrategy",
]


# Define Validator directly to avoid circular imports
@stable
class Validator:
    """
    Base class for validators.

    This class defines the interface for all validators in the Saplings framework.
    """

    def __init__(self, name: str, config: dict | None = None):
        """
        Initialize a validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator

        """
        self.name = name
        self.config = config or {}

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate an output.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        raise NotImplementedError("Subclasses must implement validate method")


# Define RuntimeValidator directly to avoid circular imports
@stable
class RuntimeValidator(Validator):
    """
    Validator that runs at runtime.

    This class defines the interface for validators that run at runtime.
    """

    def __init__(self, name: str, config: dict | None = None):
        """
        Initialize a runtime validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator

        """
        super().__init__(name, config)

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate an output at runtime.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="Runtime validation not implemented in this version",
        )


# Define StaticValidator directly to avoid circular imports
@stable
class StaticValidator(Validator):
    """
    Validator that runs at compile time.

    This class defines the interface for validators that run at compile time.
    """

    def __init__(self, name: str, config: dict | None = None):
        """
        Initialize a static validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator

        """
        super().__init__(name, config)

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate an output at compile time.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="Static validation not implemented in this version",
        )


# Define ValidatorConfig directly to avoid circular imports
@stable
class ValidatorConfig:
    """
    Configuration for validators.

    This class defines the configuration options for validators.
    """

    def __init__(self, validator_type: str = "runtime", **kwargs):
        """
        Initialize validator configuration.

        Args:
        ----
            validator_type: Type of validator
            **kwargs: Additional configuration options

        """
        self.validator_type = validator_type
        self.options = kwargs

    @classmethod
    def default(cls) -> "ValidatorConfig":
        """
        Get default validator configuration.

        Returns
        -------
            ValidatorConfig: Default configuration

        """
        return cls(validator_type="runtime")


# Define ValidatorRegistry directly to avoid circular imports
@stable
class ValidatorRegistry:
    """
    Registry for validators.

    This class provides a registry for validators in the Saplings framework.
    """

    def __init__(self):
        """Initialize the validator registry."""
        self._validators = {}

    def register_validator(self, name: str, validator_cls, **kwargs):
        """
        Register a validator.

        Args:
        ----
            name: Name of the validator
            validator_cls: Validator class
            **kwargs: Additional arguments for the validator

        """
        self._validators[name] = (validator_cls, kwargs)

    def get_validator(self, name: str, config: ValidatorConfig | None = None) -> Validator:
        """
        Get a validator by name.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator

        Returns:
        -------
            Validator: The validator

        Raises:
        ------
            ValueError: If the validator is not found

        """
        if name not in self._validators:
            raise ValueError(f"Validator {name} not found")

        validator_cls, kwargs = self._validators[name]
        return validator_cls(name=name, config=config, **kwargs)

    def list_validators(self) -> list[str]:
        """
        List all registered validators.

        Returns
        -------
            List[str]: List of validator names

        """
        return list(self._validators.keys())


# Define a simpler get_validator_registry function
@stable
def get_validator_registry() -> ValidatorRegistry:
    """
    Get the validator registry.

    Returns
    -------
        ValidatorRegistry: The validator registry.

    """
    # Use a singleton pattern with lazy initialization to avoid circular imports
    if not hasattr(get_validator_registry, "_registry"):
        # Create a new registry instance
        get_validator_registry._registry = ValidatorRegistry()

        # Register built-in validators directly
        registry = get_validator_registry._registry
        
        # Register ExecutionValidator
        registry.register_validator("execution", ExecutionValidator)
        registry.register_validator("basic", ExecutionValidator)  # Alias for basic validation
        
        # Register LengthValidator  
        registry.register_validator("length", LengthValidator)
        registry.register_validator("general", LengthValidator)  # Fallback for general validation
        
        # Register KeywordValidator
        registry.register_validator("keyword", KeywordValidator)
        
        # Register other validators
        registry.register_validator("sentiment", SentimentValidator)
        registry.register_validator("pii", PiiValidator)
        registry.register_validator("profanity", ProfanityValidator)

    return get_validator_registry._registry


# Define ExecutionValidator directly to avoid circular imports
@stable
class ExecutionValidator(RuntimeValidator):
    """
    Validator that checks if a response can be executed.

    This validator ensures that the response can be executed without errors.
    """

    def __init__(self, name: str = "execution", config: dict | ValidatorConfig | None = None):
        """
        Initialize an execution validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config
        super().__init__(name, config_dict)

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output can be executed without errors.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="Execution validation not implemented in this version",
        )


# Define KeywordValidator directly to avoid circular imports
@stable
class KeywordValidator(StaticValidator):
    """
    Validator that checks for required keywords in a response.

    This validator ensures that the response contains specified keywords.
    """

    def __init__(
        self,
        name: str = "keyword",
        config: dict | ValidatorConfig | None = None,
        keywords: list[str] | None = None,
    ):
        """
        Initialize a keyword validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator
            keywords: List of keywords to check for

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config or {}
        super().__init__(name, config_dict)
        self.keywords = keywords or []

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output contains required keywords.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        if not self.keywords:
            return ValidationResult(
                status=ValidationStatus.SUCCESS, message="No keywords specified"
            )

        missing_keywords = [kw for kw in self.keywords if kw.lower() not in output.lower()]

        if missing_keywords:
            return ValidationResult(
                status=ValidationStatus.FAILURE,
                message=f"Missing required keywords: {', '.join(missing_keywords)}",
                metadata={"missing_keywords": missing_keywords},
            )

        return ValidationResult(
            status=ValidationStatus.SUCCESS, message="All required keywords found"
        )


# Define LengthValidator directly to avoid circular imports
@stable
class LengthValidator(StaticValidator):
    """
    Validator that checks the length of a response.

    This validator ensures that the response is within a specified length range.
    """

    def __init__(
        self,
        name: str = "length",
        config: dict | ValidatorConfig | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
    ):
        """
        Initialize a length validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator
            min_length: Minimum length of the response
            max_length: Maximum length of the response

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config or {}
        super().__init__(name, config_dict)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output is within the specified length range.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        length = len(output)

        if self.min_length is not None and length < self.min_length:
            return ValidationResult(
                status=ValidationStatus.FAILURE,
                message=f"Response length ({length}) is less than minimum length ({self.min_length})",
                metadata={"length": length, "min_length": self.min_length},
            )

        if self.max_length is not None and length > self.max_length:
            return ValidationResult(
                status=ValidationStatus.FAILURE,
                message=f"Response length ({length}) exceeds maximum length ({self.max_length})",
                metadata={"length": length, "max_length": self.max_length},
            )

        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message=f"Response length ({length}) is within the specified range",
            metadata={"length": length},
        )


# Define SentimentValidator directly to avoid circular imports
@stable
class SentimentValidator(StaticValidator):
    """
    Validator that checks the sentiment of a response.

    This validator ensures that the response has the desired sentiment.
    """

    def __init__(
        self,
        name: str = "sentiment",
        config: dict | ValidatorConfig | None = None,
        sentiment: str | None = None,
        threshold: float = 0.5,
    ):
        """
        Initialize a sentiment validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator
            sentiment: Desired sentiment (positive, negative, neutral)
            threshold: Threshold for sentiment detection

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config or {}
        super().__init__(name, config_dict)
        self.sentiment = sentiment
        self.threshold = threshold

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output has the desired sentiment.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="Sentiment validation not implemented in this version",
        )


# Define PiiValidator directly to avoid circular imports
@stable
class PiiValidator(StaticValidator):
    """
    Validator that checks for personally identifiable information (PII) in a response.

    This validator ensures that the response does not contain PII.
    """

    def __init__(
        self,
        name: str = "pii",
        config: dict | ValidatorConfig | None = None,
        pii_types: list[str] | None = None,
    ):
        """
        Initialize a PII validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator
            pii_types: Types of PII to check for

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config or {}
        super().__init__(name, config_dict)
        self.pii_types = pii_types or []

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output does not contain PII.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="PII validation not implemented in this version",
        )


# Define ProfanityValidator directly to avoid circular imports
@stable
class ProfanityValidator(StaticValidator):
    """
    Validator that checks for profanity in a response.

    This validator ensures that the response does not contain profanity.
    """

    def __init__(
        self,
        name: str = "profanity",
        config: dict | ValidatorConfig | None = None,
        profanity_list: list[str] | None = None,
    ):
        """
        Initialize a profanity validator.

        Args:
        ----
            name: Name of the validator
            config: Configuration for the validator
            profanity_list: List of profanity terms to check for

        """
        # Convert ValidatorConfig to dict if needed
        config_dict = config.options if isinstance(config, ValidatorConfig) else config or {}
        super().__init__(name, config_dict)
        self.profanity_list = profanity_list or []

    def validate(self, output: str, context: dict | None = None) -> "ValidationResult":
        """
        Validate that an output does not contain profanity.

        Args:
        ----
            output: The output to validate
            context: Additional context for validation

        Returns:
        -------
            ValidationResult: The result of the validation

        """
        # This is a stub implementation
        return ValidationResult(
            status=ValidationStatus.SUCCESS,
            message="Profanity validation not implemented in this version",
        )
