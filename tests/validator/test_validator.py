"""
Tests for the Validator classes.
"""

import pytest

from saplings.validator.config import ValidatorType
from saplings.validator.validator import (
    RuntimeValidator,
    StaticValidator,
    ValidationResult,
    ValidationStatus,
    Validator,
)


class TestStaticValidator(StaticValidator):
    """Test static validator implementation."""

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "test_static_validator"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "Test static validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Test Static Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
        """Validate a prompt."""
        if "test" in prompt.lower():
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message="Prompt contains 'test'",
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message="Prompt does not contain 'test'",
            )


class TestRuntimeValidator(RuntimeValidator):
    """Test runtime validator implementation."""

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "test_runtime_validator"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "Test runtime validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Test Runtime Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output."""
        if "valid" in output.lower():
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message="Output contains 'valid'",
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message="Output does not contain 'valid'",
            )


class TestValidationResult:
    """Tests for the ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            validator_id="test_validator",
            status=ValidationStatus.PASSED,
            message="Test message",
            details={"key": "value"},
            metadata={"meta_key": "meta_value"},
        )

        assert result.validator_id == "test_validator"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Test message"
        assert result.details == {"key": "value"}
        assert result.metadata == {"meta_key": "meta_value"}

    def test_validation_result_to_dict(self):
        """Test converting a validation result to a dictionary."""
        result = ValidationResult(
            validator_id="test_validator",
            status=ValidationStatus.PASSED,
            message="Test message",
            details={"key": "value"},
            metadata={"meta_key": "meta_value"},
        )

        result_dict = result.to_dict()

        assert result_dict["validator_id"] == "test_validator"
        assert result_dict["status"] == ValidationStatus.PASSED
        assert result_dict["message"] == "Test message"
        assert result_dict["details"] == {"key": "value"}
        assert result_dict["metadata"] == {"meta_key": "meta_value"}

    def test_validation_result_from_dict(self):
        """Test creating a validation result from a dictionary."""
        result_dict = {
            "validator_id": "test_validator",
            "status": ValidationStatus.PASSED,
            "message": "Test message",
            "details": {"key": "value"},
            "metadata": {"meta_key": "meta_value"},
        }

        result = ValidationResult.from_dict(result_dict)

        assert result.validator_id == "test_validator"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Test message"
        assert result.details == {"key": "value"}
        assert result.metadata == {"meta_key": "meta_value"}


class TestStaticValidatorClass:
    """Tests for the StaticValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a test static validator."""
        return TestStaticValidator()

    def test_validator_type(self, validator):
        """Test the validator type."""
        assert validator.validator_type == ValidatorType.STATIC

    @pytest.mark.asyncio
    async def test_validate_prompt(self, validator):
        """Test validating a prompt."""
        # Test with a valid prompt
        result = await validator.validate_prompt("This is a test prompt")
        assert result.status == ValidationStatus.PASSED

        # Test with an invalid prompt
        result = await validator.validate_prompt("This is a prompt")
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate(self, validator):
        """Test the validate method."""
        # The validate method should delegate to validate_prompt
        result = await validator.validate(
            output="This is an output",
            prompt="This is a test prompt",
        )
        assert result.status == ValidationStatus.PASSED

        result = await validator.validate(
            output="This is an output",
            prompt="This is a prompt",
        )
        assert result.status == ValidationStatus.FAILED


class TestRuntimeValidatorClass:
    """Tests for the RuntimeValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a test runtime validator."""
        return TestRuntimeValidator()

    def test_validator_type(self, validator):
        """Test the validator type."""
        assert validator.validator_type == ValidatorType.RUNTIME

    @pytest.mark.asyncio
    async def test_validate_output(self, validator):
        """Test validating an output."""
        # Test with a valid output
        result = await validator.validate_output(
            output="This is a valid output",
            prompt="This is a prompt",
        )
        assert result.status == ValidationStatus.PASSED

        # Test with an invalid output
        result = await validator.validate_output(
            output="This is an output",
            prompt="This is a prompt",
        )
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate(self, validator):
        """Test the validate method."""
        # The validate method should delegate to validate_output
        result = await validator.validate(
            output="This is a valid output",
            prompt="This is a prompt",
        )
        assert result.status == ValidationStatus.PASSED

        result = await validator.validate(
            output="This is an output",
            prompt="This is a prompt",
        )
        assert result.status == ValidationStatus.FAILED
