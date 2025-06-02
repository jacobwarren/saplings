from __future__ import annotations

"""
ValidatorRegistry module for Saplings.

This module provides the ValidatorRegistry class for managing validators.
"""

import asyncio
import importlib
import inspect
import logging
import os
import pkgutil
import sys
import time
from typing import TypeVar, cast

from importlib_metadata import entry_points

from saplings.validator._internal.config import ValidatorConfig, ValidatorType
from saplings.validator._internal.registry.plugin_utils import (
    PluginTypeEnum,
    get_plugins_by_type_lazy,
)
from saplings.validator._internal.result import ValidationResult, ValidationStatus
from saplings.validator._internal.validator import (
    RuntimeValidator,
    StaticValidator,
    Validator,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Validator)


class ValidatorRegistry:
    """
    Registry for validators.

    This class provides methods for registering, discovering, and retrieving validators.
    """

    def __init__(self) -> None:
        """Initialize the validator registry."""
        self._validators: dict[str, type[Validator]] = {}
        self._validator_instances: dict[str, Validator] = {}
        self._config = ValidatorConfig.default()
        self._validation_count = 0  # Counter for budget enforcement

        # Always register the execution validator by default
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validators that should always be available."""
        try:
            # Import the execution validator - use a more direct import to avoid circular dependencies
            from saplings.validator._internal.implementations.execution import ExecutionValidator

            # Create an instance
            execution_validator = ExecutionValidator()

            # Register it directly in the internal dictionaries
            self._validators[execution_validator.id] = ExecutionValidator
            self._validator_instances[execution_validator.id] = execution_validator

            logger.debug(
                f"Registered default execution validator with ID: {execution_validator.id}"
            )
        except Exception as e:
            logger.warning(f"Failed to register default execution validator: {e}")

    def reset_budget(self) -> None:
        """Reset the validation budget counter."""
        self._validation_count = 0

    def register_validator(self, validator_class: type[Validator]) -> None:
        """
        Register a validator class.

        Args:
        ----
            validator_class: Validator class to register

        Raises:
        ------
            ValueError: If a validator with the same ID is already registered

        """
        # Create a temporary instance to get the ID
        validator = validator_class()
        validator_id = validator.id

        if validator_id in self._validators:
            # If it's the same class, just log a warning and return
            if self._validators[validator_id] == validator_class:
                logger.warning(f"Validator with ID '{validator_id}' is already registered")
                return
            # If it's a different class, raise an error
            msg = f"Validator with ID '{validator_id}' is already registered"
            raise ValueError(msg)

        # Register the validator
        self._validators[validator_id] = validator_class
        logger.debug(f"Registered validator: {validator_id}")

    def get_validator(self, validator_id: str) -> Validator:
        """
        Get a validator by ID.

        Args:
        ----
            validator_id: ID of the validator

        Returns:
        -------
            Validator: Validator instance

        Raises:
        ------
            ValueError: If the validator is not found

        """
        # Check if we already have an instance
        if validator_id in self._validator_instances:
            return self._validator_instances[validator_id]

        # Check if we have the class
        if validator_id not in self._validators:
            msg = f"Validator not found: {validator_id}"
            raise ValueError(msg)

        # Create a new instance
        validator_class = self._validators[validator_id]
        validator = validator_class()

        # Cache the instance
        self._validator_instances[validator_id] = validator

        return validator

    def list_validators(self) -> list[str]:
        """
        List all registered validators.

        Returns
        -------
            List[str]: List of validator IDs

        """
        return list(self._validators.keys())

    def get_validators_by_type(self, validator_type: ValidatorType) -> list[str]:
        """
        Get validators of a specific type.

        Args:
        ----
            validator_type: Type of validators to get

        Returns:
        -------
            List[str]: List of validator IDs

        """
        result = []
        for validator_id, validator_class in self._validators.items():
            # Create a temporary instance to get the type
            validator = validator_class()
            if validator.validator_type == validator_type:
                result.append(validator_id)
        return result

    def discover_validators(self) -> None:
        """Discover validators from plugins and specified directories."""
        # Discover validators from plugins
        if self._config.use_entry_points:
            self._discover_validators_from_entry_points()

        # Discover validators from specified directories
        for directory in self._config.plugin_dirs:
            self._discover_validators_from_directory(directory)

    def _discover_validators_from_entry_points(self) -> None:
        """Discover validators from entry points."""
        # Get validators from plugin registry using lazy import to avoid circular dependencies
        validator_plugins = get_plugins_by_type_lazy(PluginTypeEnum.VALIDATOR)
        for plugin_class in validator_plugins.values():
            if issubclass(plugin_class, Validator):
                self.register_validator(cast("type[Validator]", plugin_class))

        # Get validators from entry points
        try:
            for entry_point in entry_points(group="saplings.validators"):
                try:
                    validator_class = entry_point.load()
                    if issubclass(validator_class, Validator):
                        self.register_validator(validator_class)
                except Exception as e:
                    logger.warning(
                        f"Failed to load validator from entry point {entry_point.name}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Failed to discover validators from entry points: {e}")

    def _discover_validators_from_directory(self, directory: str) -> None:
        """
        Discover validators from a directory.

        Args:
        ----
            directory: Directory to search for validators

        """
        if not os.path.isdir(directory):
            logger.warning(f"Validator directory not found: {directory}")
            return

        # Add the directory to the Python path
        sys.path.insert(0, directory)

        # Find all Python modules in the directory
        for _, name, _ in pkgutil.iter_modules([directory]):
            try:
                # Import the module
                module = importlib.import_module(name)

                # Find all classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a validator class
                    if (
                        inspect.isclass(attr)
                        and issubclass(attr, Validator)
                        and attr not in (Validator, StaticValidator, RuntimeValidator)
                    ):
                        self.register_validator(attr)
            except Exception as e:
                logger.warning(f"Failed to load validator from module {name}: {e}")

        # Remove the directory from the Python path
        sys.path.remove(directory)

    async def validate(
        self,
        output: str,
        prompt: str,
        validator_ids: list[str] | None = None,
        validator_type: ValidatorType | None = None,
        **kwargs,
    ) -> list[ValidationResult]:
        """
        Validate an output using the specified validators.

        Args:
        ----
            output: Output to validate
            prompt: Prompt that generated the output
            validator_ids: IDs of validators to use (if None, use all)
            validator_type: Type of validators to use (if None, use all)
            **kwargs: Additional validation parameters

        Returns:
        -------
            List[ValidationResult]: Validation results

        """
        # Increment the validation count
        self._validation_count += 1

        # Check if we've exceeded the budget
        if (
            self._config.enforce_budget
            and self._validation_count > self._config.max_validations_per_session
        ):
            logger.warning(
                f"Validation budget exceeded: {self._validation_count} > {self._config.max_validations_per_session}"
            )
            return [
                ValidationResult(
                    validator_id="budget_enforcer",
                    status=ValidationStatus.ERROR,
                    message="Validation budget exceeded",
                )
            ]

        # Get the validators to use
        validators_to_use = []

        if validator_ids is not None:
            # Use the specified validators
            for validator_id in validator_ids:
                try:
                    validators_to_use.append(self.get_validator(validator_id))
                except ValueError:
                    logger.warning(f"Validator not found: {validator_id}")
        else:
            # Use all validators of the specified type
            for validator_id in self.list_validators():
                validator = self.get_validator(validator_id)
                if validator_type is None or validator.validator_type == validator_type:
                    validators_to_use.append(validator)

        # Validate using the selected validators
        results = []

        if self._config.parallel_validation:
            # Run validators in parallel
            tasks = []
            for validator in validators_to_use:
                task = asyncio.create_task(
                    self._validate_with_validator(validator, output, prompt, **kwargs)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)

                    # Check if we should stop on first failure
                    if self._config.fail_fast and result.status in (
                        ValidationStatus.FAILED,
                        ValidationStatus.ERROR,
                    ):
                        # Cancel remaining tasks
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        break
                except Exception as e:
                    logger.warning(f"Validation task failed: {e}")
        else:
            # Run validators sequentially
            for validator in validators_to_use:
                result = await self._validate_with_validator(validator, output, prompt, **kwargs)
                results.append(result)

                # Check if we should stop on first failure
                if self._config.fail_fast and result.status in (
                    ValidationStatus.FAILED,
                    ValidationStatus.ERROR,
                ):
                    break

        return results

    async def _validate_with_validator(
        self, validator: Validator, output: str, prompt: str, **kwargs
    ) -> ValidationResult:
        """
        Validate an output using a specific validator.

        Args:
        ----
            validator: Validator to use
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
        -------
            ValidationResult: Validation result

        """
        start_time = time.time()

        try:
            # Set up the timeout
            if self._config.timeout_seconds is not None:
                # Run with timeout
                result = await asyncio.wait_for(
                    validator.validate(output, prompt, **kwargs),
                    timeout=self._config.timeout_seconds,
                )
            else:
                # Run without timeout
                result = await validator.validate(output, prompt, **kwargs)

            # Add latency to metadata
            result.metadata["latency_ms"] = int((time.time() - start_time) * 1000)

            return result
        except asyncio.TimeoutError:
            # Validation timed out
            return ValidationResult(
                validator_id=validator.id,
                status=ValidationStatus.ERROR,
                message=f"Validation timed out after {self._config.timeout_seconds} seconds",
                metadata={
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "error": "timeout",
                },
            )
        except Exception as e:
            # Validation failed with an exception
            return ValidationResult(
                validator_id=validator.id,
                status=ValidationStatus.ERROR,
                message=f"Validation failed with an exception: {e}",
                metadata={
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


# Global registry instance for backward compatibility
_global_registry = None


def get_validator_registry():
    """
    Get the validator registry.

    This function is maintained for backward compatibility.
    New code should use constructor injection via the DI container.

    Returns
    -------
        ValidatorRegistry: Validator registry instance

    """
    global _global_registry

    if _global_registry is None:
        _global_registry = ValidatorRegistry()

    return _global_registry
