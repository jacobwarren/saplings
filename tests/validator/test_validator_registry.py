"""
Tests for the ValidatorRegistry.
"""

import asyncio
import importlib
import os
import tempfile
from typing import Dict, List, Optional, Type

import pytest

from saplings.core.plugin import PluginType
from saplings.validator.config import ValidatorConfig, ValidatorType
from saplings.validator.registry import ValidatorRegistry, get_validator_registry
from saplings.validator.validator import (
    RuntimeValidator,
    StaticValidator,
    ValidationResult,
    ValidationStatus,
    Validator,
)


class MockStaticValidator(StaticValidator):
    """Mock static validator for testing."""

    def __init__(self, id_suffix=""):
        """Initialize the mock validator."""
        self._id = f"mock_static_validator{id_suffix}"
        self._description = "Mock static validator for testing"

    @property
    def id(self) -> str:
        """ID of the validator."""
        return self._id

    @property
    def description(self) -> str:
        """Description of the validator."""
        return self._description

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Mock Static Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
        """Validate a prompt."""
        # Simple validation: check if prompt contains "test"
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


class MockRuntimeValidator(RuntimeValidator):
    """Mock runtime validator for testing."""

    def __init__(self, id_suffix=""):
        """Initialize the mock validator."""
        self._id = f"mock_runtime_validator{id_suffix}"
        self._description = "Mock runtime validator for testing"

    @property
    def id(self) -> str:
        """ID of the validator."""
        return self._id

    @property
    def description(self) -> str:
        """Description of the validator."""
        return self._description

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Mock Runtime Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output."""
        # Simple validation: check if output contains "valid"
        if "valid" in output.lower():
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message="Output contains 'valid'",
            )
        else:
            # Special case for the test_validate_with_failing_validators test
            if output == "This is an invalid output":
                return ValidationResult(
                    validator_id=self.id,
                    status=ValidationStatus.FAILED,
                    message="Output does not contain 'valid'",
                )
            else:
                return ValidationResult(
                    validator_id=self.id,
                    status=ValidationStatus.FAILED,
                    message="Output does not contain 'valid'",
                )


class SlowMockValidator(RuntimeValidator):
    """Slow mock validator for testing timeouts."""

    def __init__(self, delay_seconds=1.0, id_suffix=""):
        """Initialize the mock validator."""
        self._id = f"slow_mock_validator{id_suffix}"
        self._description = "Slow mock validator for testing timeouts"
        self._delay_seconds = delay_seconds

    @property
    def id(self) -> str:
        """ID of the validator."""
        return self._id

    @property
    def description(self) -> str:
        """Description of the validator."""
        return self._description

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Slow Mock Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output with a delay."""
        # Simulate a slow validation
        await asyncio.sleep(self._delay_seconds)

        return ValidationResult(
            validator_id=self.id,
            status=ValidationStatus.PASSED,
            message="Slow validation completed",
        )


class ErrorMockValidator(RuntimeValidator):
    """Mock validator that raises an exception."""

    def __init__(self):
        """Initialize the mock validator."""
        self._id = "error_mock_validator"
        self._description = "Mock validator that raises an exception"

    @property
    def id(self) -> str:
        """ID of the validator."""
        return self._id

    @property
    def description(self) -> str:
        """Description of the validator."""
        return self._description

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Error Mock Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """Validate an output by raising an exception."""
        raise ValueError("Simulated validation error")


class TestValidatorRegistry:
    """Tests for the ValidatorRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh validator registry for testing."""
        # Reset the singleton
        ValidatorRegistry._instance = None
        return ValidatorRegistry()

    def test_singleton(self, registry):
        """Test that the registry is a singleton."""
        registry2 = ValidatorRegistry()
        assert registry is registry2

    def test_get_validator_registry(self, registry):
        """Test the get_validator_registry function."""
        registry2 = get_validator_registry()
        assert registry is registry2

    def test_register_validator(self, registry):
        """Test registering a validator."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        assert "mock_static_validator" in registry.list_validators()
        assert "mock_runtime_validator" in registry.list_validators()

    def test_register_duplicate_validator(self, registry):
        """Test registering a validator with a duplicate ID."""
        registry.register_validator(MockStaticValidator)

        with pytest.raises(ValueError):
            registry.register_validator(MockStaticValidator)

    def test_get_validator(self, registry):
        """Test getting a validator by ID."""
        registry.register_validator(MockStaticValidator)

        validator = registry.get_validator("mock_static_validator")
        assert validator.id == "mock_static_validator"
        assert isinstance(validator, MockStaticValidator)

    def test_get_nonexistent_validator(self, registry):
        """Test getting a validator that doesn't exist."""
        with pytest.raises(ValueError):
            registry.get_validator("nonexistent_validator")

    def test_list_validators(self, registry):
        """Test listing all validators."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        validators = registry.list_validators()
        assert "mock_static_validator" in validators
        assert "mock_runtime_validator" in validators

    def test_get_validators_by_type(self, registry):
        """Test getting validators by type."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        static_validators = registry.get_validators_by_type(ValidatorType.STATIC)
        runtime_validators = registry.get_validators_by_type(ValidatorType.RUNTIME)

        assert "mock_static_validator" in static_validators
        assert "mock_runtime_validator" not in static_validators

        assert "mock_runtime_validator" in runtime_validators
        assert "mock_static_validator" not in runtime_validators

    @pytest.mark.asyncio
    async def test_validate_with_specific_validators(self, registry):
        """Test validating with specific validators."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        # Validate with a static validator
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
            validator_ids=["mock_static_validator"],
        )

        assert len(results) == 1
        assert results[0].validator_id == "mock_static_validator"
        assert results[0].status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_validate_with_nonexistent_validator(self, registry):
        """Test validating with a non-existent validator."""
        registry.register_validator(MockStaticValidator)

        # Validate with a non-existent validator
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
            validator_ids=["nonexistent_validator"],
        )

        # Should not raise an exception, but return an empty list
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_validate_with_validator_type(self, registry):
        """Test validating with a specific validator type."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        # Validate with runtime validators
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
            validator_type=ValidatorType.RUNTIME,
        )

        assert len(results) == 1
        assert results[0].validator_id == "mock_runtime_validator"
        assert results[0].status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_validate_with_all_validators(self, registry):
        """Test validating with all validators."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        # Validate with all validators
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
        )

        assert len(results) == 2

        # Check that both validators were used
        validator_ids = [result.validator_id for result in results]
        assert "mock_static_validator" in validator_ids
        assert "mock_runtime_validator" in validator_ids

        # Check that both validations passed
        for result in results:
            assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_validate_with_failing_validators(self, registry):
        """Test validating with validators that fail."""
        # Create custom validator classes for this test
        class CustomStaticValidator(MockStaticValidator):
            async def validate_prompt(self, prompt: str, **kwargs) -> ValidationResult:
                return ValidationResult(
                    validator_id=self.id,
                    status=ValidationStatus.FAILED,
                    message="Prompt validation failed",
                )

        class CustomRuntimeValidator(MockRuntimeValidator):
            async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
                return ValidationResult(
                    validator_id=self.id,
                    status=ValidationStatus.FAILED,
                    message="Output validation failed",
                )

        # Register the custom validators
        registry.register_validator(CustomStaticValidator)
        registry.register_validator(CustomRuntimeValidator)

        # Validate with any prompt and output
        results = await registry.validate(
            output="Any output",
            prompt="Any prompt",
            validator_ids=["mock_static_validator", "mock_runtime_validator"],
        )

        assert len(results) == 2

        # Check that both validators were used
        validator_ids = [result.validator_id for result in results]
        assert "mock_static_validator" in validator_ids
        assert "mock_runtime_validator" in validator_ids

        # Check that both validations failed
        for result in results:
            assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate_with_fail_fast(self, registry):
        """Test validating with fail_fast enabled."""
        registry.register_validator(MockStaticValidator)
        registry.register_validator(MockRuntimeValidator)

        # Configure the registry to fail fast
        config = ValidatorConfig(fail_fast=True)
        registry.configure(config)

        # Validate with a failing prompt
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is not a valid prompt",
        )

        # Only one validator should have run
        assert len(results) == 1
        assert results[0].status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate_with_timeout(self, registry):
        """Test validating with a timeout."""
        registry.register_validator(SlowMockValidator)

        # Configure the registry with a short timeout
        config = ValidatorConfig(timeout_seconds=0.1)
        registry.configure(config)

        # Validate with a slow validator
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
        )

        assert len(results) == 1
        assert results[0].validator_id == "slow_mock_validator"
        assert results[0].status == ValidationStatus.ERROR
        assert "timed out" in results[0].message

    @pytest.mark.asyncio
    async def test_validate_with_error(self, registry):
        """Test validating with a validator that raises an error."""
        registry.register_validator(ErrorMockValidator)

        # Validate with an error-raising validator
        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
        )

        assert len(results) == 1
        assert results[0].validator_id == "error_mock_validator"
        assert results[0].status == ValidationStatus.ERROR
        assert "exception" in results[0].message

    @pytest.mark.asyncio
    async def test_parallel_validation(self, registry):
        """Test parallel validation."""
        # Register multiple slow validators
        for i in range(5):
            registry.register_validator(lambda: SlowMockValidator(id_suffix=f"_{i}"))

        # Configure the registry for parallel validation
        config = ValidatorConfig(
            parallel_validation=True,
            max_parallel_validators=5,
        )
        registry.configure(config)

        # Time the validation
        start_time = asyncio.get_event_loop().time()

        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
        )

        end_time = asyncio.get_event_loop().time()
        elapsed_time = end_time - start_time

        # All validators should have run
        assert len(results) == 5

        # The elapsed time should be close to the delay of a single validator
        # (not the sum of all validators)
        assert elapsed_time < 2.0  # Allow some overhead

    @pytest.mark.asyncio
    async def test_sequential_validation(self, registry):
        """Test sequential validation."""
        # Register multiple slow validators with a short delay
        delay = 0.1
        for i in range(5):
            # Create a custom validator class for each iteration
            class CustomSlowValidator(SlowMockValidator):
                def __init__(self):
                    super().__init__(delay_seconds=delay, id_suffix=f"_{i}")

            # Register the class, not an instance
            registry.register_validator(CustomSlowValidator)

        # Configure the registry for sequential validation
        config = ValidatorConfig(
            parallel_validation=False,
        )
        registry.configure(config)

        # Time the validation
        start_time = asyncio.get_event_loop().time()

        results = await registry.validate(
            output="This is a valid output",
            prompt="This is a test prompt",
        )

        end_time = asyncio.get_event_loop().time()
        elapsed_time = end_time - start_time

        # All validators should have run
        assert len(results) == 5

        # The elapsed time should be close to the sum of all validator delays
        assert elapsed_time >= delay * 5

    def test_discover_validators_from_directory(self, registry):
        """Test discovering validators from a directory."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python module with a validator
            module_path = os.path.join(temp_dir, "test_validator_module.py")
            with open(module_path, "w") as f:
                f.write("""
from saplings.validator.validator import RuntimeValidator, ValidationResult, ValidationStatus

class TestValidator(RuntimeValidator):
    @property
    def id(self):
        return "test_validator"

    @property
    def name(self):
        return "Test Validator"

    @property
    def version(self):
        return "0.1.0"

    @property
    def description(self):
        return "Test validator"

    async def validate_output(self, output, prompt, **kwargs):
        return ValidationResult(
            validator_id=self.id,
            status=ValidationStatus.PASSED,
            message="Test validation",
        )
""")

            # Configure the registry to discover validators from the directory
            config = ValidatorConfig(
                plugin_dirs=[temp_dir],
                use_entry_points=False,
            )
            registry.configure(config)

            # Discover validators
            registry.discover_validators()

            # Check that the validator was discovered
            assert "test_validator" in registry.list_validators()

    def test_discover_validators_from_nonexistent_directory(self, registry):
        """Test discovering validators from a non-existent directory."""
        # Configure the registry with a non-existent directory
        config = ValidatorConfig(
            plugin_dirs=["/nonexistent/directory"],
            use_entry_points=False,
        )
        registry.configure(config)

        # Discover validators (should not raise an exception)
        registry.discover_validators()

    def test_discover_validators_from_entry_points(self, registry, monkeypatch):
        """Test discovering validators from entry points."""
        # Mock the get_plugins_by_type function
        def mock_get_plugins_by_type(plugin_type):
            if plugin_type == PluginType.VALIDATOR:
                return {"mock_validator": MockStaticValidator}
            return {}

        # Apply the mock
        monkeypatch.setattr("saplings.validator.registry.get_plugins_by_type", mock_get_plugins_by_type)

        # Configure the registry to use entry points
        config = ValidatorConfig(
            use_entry_points=True,
        )
        registry.configure(config)

        # Discover validators
        registry.discover_validators()

        # Check that the validator was discovered
        assert "mock_static_validator" in registry.list_validators()

    def test_discover_validators_from_entry_points_with_error(self, registry, monkeypatch):
        """Test discovering validators from entry points with an error."""
        # Mock the entry_points function
        def mock_entry_points(group=None):
            class MockEntryPoint:
                def __init__(self, name, load_func):
                    self.name = name
                    self._load_func = load_func

                def load(self):
                    return self._load_func()

            if group == "saplings.validators":
                return [
                    MockEntryPoint("error_validator", lambda: Exception("Test error")),
                ]
            return []

        # Apply the mock
        monkeypatch.setattr("importlib_metadata.entry_points", mock_entry_points)

        # Configure the registry to use entry points
        config = ValidatorConfig(
            use_entry_points=True,
        )
        registry.configure(config)

        # Discover validators (should not raise an exception)
        registry.discover_validators()

        # No validators should be discovered
        assert "error_validator" not in registry.list_validators()

    def test_discover_validators_from_directory_with_error(self, monkeypatch):
        """Test discovering validators from a directory with an error."""
        # Create a fresh registry for this test
        ValidatorRegistry._instance = None
        test_registry = ValidatorRegistry()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python module with a syntax error
            module_path = os.path.join(temp_dir, "error_module.py")
            with open(module_path, "w") as f:
                f.write("""
# This module has a syntax error
class TestValidator(
""")

            # Mock importlib.import_module to raise an exception
            def mock_import_module(name):
                if name == "error_module":
                    raise SyntaxError("invalid syntax")
                return importlib.import_module(name)

            # Apply the mock
            monkeypatch.setattr("importlib.import_module", mock_import_module)

            # Configure the registry to discover validators from the directory
            config = ValidatorConfig(
                plugin_dirs=[temp_dir],
                use_entry_points=False,
            )
            test_registry.configure(config)

            # Discover validators (should not raise an exception)
            test_registry.discover_validators()

            # No validators should be discovered
            assert len(test_registry.list_validators()) == 0

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, registry):
        """Test that budget constraints are enforced."""
        # Create a validator that tracks validation calls
        class BudgetTrackingValidator(RuntimeValidator):
            def __init__(self):
                self.validation_count = 0

            @property
            def id(self) -> str:
                return "budget_tracking_validator"

            @property
            def description(self) -> str:
                return "Validator for testing budget enforcement"

            @property
            def name(self) -> str:
                return "Budget Tracking Validator"

            @property
            def version(self) -> str:
                return "0.1.0"

            async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
                self.validation_count += 1
                return ValidationResult(
                    validator_id=self.id,
                    status=ValidationStatus.PASSED,
                    message=f"Validation #{self.validation_count}",
                )

        # Register the validator
        registry.register_validator(BudgetTrackingValidator)

        # Get the validator instance
        validator = registry.get_validator("budget_tracking_validator")

        # Configure the registry with a budget limit
        config = ValidatorConfig(
            max_validations_per_session=3,  # Limit to 3 validations
            enforce_budget=True,
        )
        registry.configure(config)

        # Run validations up to the limit
        for i in range(3):
            results = await registry.validate(
                output=f"Output {i}",
                prompt=f"Prompt {i}",
            )
            assert len(results) == 1
            assert results[0].status == ValidationStatus.PASSED

        # The next validation should be skipped due to budget enforcement
        results = await registry.validate(
            output="Output beyond budget",
            prompt="Prompt beyond budget",
        )

        # Should return a result with a budget exceeded status
        assert len(results) == 1
        assert results[0].status == ValidationStatus.SKIPPED
        assert "budget" in results[0].message.lower()

        # Verify the validator was only called 3 times
        assert validator.validation_count == 3

        # Reset the budget
        registry.reset_budget()

        # Should be able to validate again
        results = await registry.validate(
            output="Output after reset",
            prompt="Prompt after reset",
        )
        assert len(results) == 1
        assert results[0].status == ValidationStatus.PASSED
        assert validator.validation_count == 4
