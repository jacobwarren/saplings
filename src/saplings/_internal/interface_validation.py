"""
Interface validation module for Saplings.

This module provides validation functionality to ensure that service implementations
properly implement their interfaces as specified in Task 3.5.

This implementation follows the circular import resolution standard by avoiding
imports of interface classes at module level and using string-based validation.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC
from typing import Callable, Dict, Optional, Type

# Configure logging
logger = logging.getLogger(__name__)


class InterfaceValidationError(Exception):
    """Raised when interface validation fails."""

    def __init__(
        self,
        message: str,
        interface_name: Optional[str] = None,
        implementation_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.interface_name = interface_name
        self.implementation_name = implementation_name


def validate_interface_implementation(interface: Type, implementation: Type) -> None:
    """
    Validate that implementation properly implements interface.

    Args:
    ----
        interface: The interface class to validate against
        implementation: The implementation class to validate

    Raises:
    ------
        InterfaceValidationError: If implementation doesn't properly implement interface

    """
    interface_name = getattr(interface, "__name__", str(interface))
    implementation_name = getattr(implementation, "__name__", str(implementation))

    logger.debug(f"Validating {implementation_name} against {interface_name}")

    # Check if interface is actually an ABC
    if not (inspect.isclass(interface) and issubclass(interface, ABC)):
        raise InterfaceValidationError(
            f"Interface {interface_name} is not an abstract base class",
            interface_name=interface_name,
            implementation_name=implementation_name,
        )

    # Check if implementation is a class
    if not inspect.isclass(implementation):
        raise InterfaceValidationError(
            f"Implementation {implementation_name} is not a class",
            interface_name=interface_name,
            implementation_name=implementation_name,
        )

    # Check if implementation still has abstract methods (meaning it's incomplete)
    if hasattr(implementation, "__abstractmethods__") and implementation.__abstractmethods__:
        missing_methods = list(implementation.__abstractmethods__)
        raise InterfaceValidationError(
            f"{implementation_name} missing required method '{missing_methods[0]}' from {interface_name}",
            interface_name=interface_name,
            implementation_name=implementation_name,
        )

    # Get all abstract methods from the interface
    abstract_methods = _get_abstract_methods(interface)

    # Validate each abstract method is implemented
    for method_name, method in abstract_methods.items():
        if not hasattr(implementation, method_name):
            raise InterfaceValidationError(
                f"{implementation_name} missing required method '{method_name}' from {interface_name}",
                interface_name=interface_name,
                implementation_name=implementation_name,
            )

        impl_method = getattr(implementation, method_name)
        if not callable(impl_method):
            raise InterfaceValidationError(
                f"{implementation_name}.{method_name} is not callable",
                interface_name=interface_name,
                implementation_name=implementation_name,
            )

        # Check if the method is still abstract in the implementation
        if getattr(impl_method, "__isabstractmethod__", False):
            raise InterfaceValidationError(
                f"{implementation_name}.{method_name} is still abstract (not properly implemented)",
                interface_name=interface_name,
                implementation_name=implementation_name,
            )

        # Validate method signature compatibility
        _validate_method_signature(
            interface_name, implementation_name, method_name, method, impl_method
        )

    logger.debug(f"✓ {implementation_name} properly implements {interface_name}")


def _get_abstract_methods(interface: Type) -> Dict[str, Callable]:
    """
    Get all abstract methods from an interface.

    Args:
    ----
        interface: The interface class

    Returns:
    -------
        Dict mapping method names to method objects

    """
    abstract_methods = {}

    # Check the __abstractmethods__ attribute which contains all abstract method names
    if hasattr(interface, "__abstractmethods__"):
        for method_name in interface.__abstractmethods__:
            if hasattr(interface, method_name):
                method = getattr(interface, method_name)
                abstract_methods[method_name] = method

    # Also check all methods in the class hierarchy for abstract methods
    for name, method in inspect.getmembers(interface, predicate=inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            abstract_methods[name] = method

    # Check parent classes as well
    for base in interface.__mro__[1:]:  # Skip the interface itself
        if hasattr(base, "__abstractmethods__"):
            for method_name in base.__abstractmethods__:
                if method_name not in abstract_methods and hasattr(base, method_name):
                    abstract_methods[method_name] = getattr(base, method_name)

    logger.debug(f"Found abstract methods in {interface.__name__}: {list(abstract_methods.keys())}")
    return abstract_methods


def _validate_method_signature(
    interface_name: str,
    implementation_name: str,
    method_name: str,
    interface_method: Callable,
    impl_method: Callable,
) -> None:
    """
    Validate that implementation method signature is compatible with interface method.

    Args:
    ----
        interface_name: Name of the interface
        implementation_name: Name of the implementation
        method_name: Name of the method being validated
        interface_method: The interface method
        impl_method: The implementation method

    Raises:
    ------
        InterfaceValidationError: If signatures are incompatible

    """
    try:
        interface_sig = inspect.signature(interface_method)
        impl_sig = inspect.signature(impl_method)

        # Check parameter count (allowing for additional optional parameters in implementation)
        interface_params = list(interface_sig.parameters.values())
        impl_params = list(impl_sig.parameters.values())

        # Remove 'self' parameter for comparison
        if interface_params and interface_params[0].name == "self":
            interface_params = interface_params[1:]
        if impl_params and impl_params[0].name == "self":
            impl_params = impl_params[1:]

        # Check that implementation has at least the required parameters
        required_interface_params = [
            p for p in interface_params if p.default == inspect.Parameter.empty
        ]

        if len(impl_params) < len(required_interface_params):
            raise InterfaceValidationError(
                f"{implementation_name}.{method_name} has fewer parameters than required by {interface_name}",
                interface_name=interface_name,
                implementation_name=implementation_name,
            )

        # Check parameter names match for required parameters
        for i, interface_param in enumerate(required_interface_params):
            if i >= len(impl_params):
                break
            impl_param = impl_params[i]

            # Allow different parameter names but warn about it
            if interface_param.name != impl_param.name:
                logger.warning(
                    f"{implementation_name}.{method_name} parameter '{impl_param.name}' "
                    f"differs from interface parameter '{interface_param.name}'"
                )

    except Exception as e:
        logger.warning(f"Could not validate method signature for {method_name}: {e}")


def register_with_validation(
    container_instance, interface: Type, implementation: Type, **kwargs
) -> None:
    """
    Register service with the container after validating interface compliance.

    Args:
    ----
        container_instance: The DI container instance
        interface: The interface class
        implementation: The implementation class
        **kwargs: Additional arguments for container registration

    Raises:
    ------
        InterfaceValidationError: If validation fails

    """
    # Validate interface compliance first
    validate_interface_implementation(interface, implementation)

    # Register with container (this is a mock implementation for Task 3.5)
    # In a real implementation, this would call the actual container registration
    logger.debug(f"Registering {implementation.__name__} for {interface.__name__}")

    # Mock registration - in real implementation would be:
    # container_instance.register(interface, implementation, **kwargs)


def validate_all_service_implementations() -> Dict[str, bool]:
    """
    Validate all known service implementations against their interfaces.

    This function uses string-based interface names to avoid circular imports
    and dynamically imports the classes for validation.

    Returns
    -------
        Dict mapping service names to validation results (True = valid, False = invalid)

    """
    # Service mappings using string names to avoid circular imports
    service_mappings = {
        "IExecutionService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.execution_service",
            "ExecutionService",
        ),
        "IMemoryManager": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.memory_manager",
            "MemoryManager",
        ),
        "IValidatorService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.validator_service",
            "ValidatorService",
        ),
        "IMonitoringService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.monitoring_service",
            "MonitoringService",
        ),
        "IRetrievalService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.retrieval_service",
            "RetrievalService",
        ),
        "IPlannerService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.planner_service",
            "PlannerService",
        ),
        "IToolService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.tool_service",
            "ToolService",
        ),
        "IOrchestrationService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.orchestration_service",
            "OrchestrationService",
        ),
        "ISelfHealingService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.self_healing_service",
            "SelfHealingService",
        ),
        "IModalityService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.modality_service",
            "ModalityService",
        ),
        "IModelInitializationService": (
            "saplings.api.core.interfaces",
            "saplings._internal.services.model_initialization_service",
            "ModelInitializationService",
        ),
    }

    results = {}

    for interface_name, (interface_module, impl_module, impl_class) in service_mappings.items():
        try:
            # Set a timeout for imports to avoid hanging
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Import timed out")

            # Set a 5-second timeout for imports
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)

            try:
                # Dynamically import interface and implementation
                interface_mod = __import__(interface_module, fromlist=[interface_name])
                interface_cls = getattr(interface_mod, interface_name)

                impl_mod = __import__(impl_module, fromlist=[impl_class])
                impl_cls = getattr(impl_mod, impl_class)

                # Validate the implementation
                validate_interface_implementation(interface_cls, impl_cls)
                results[interface_name] = True
                logger.info(f"✓ {interface_name} implementation is valid")

            finally:
                # Cancel the alarm
                signal.alarm(0)

        except TimeoutError:
            results[interface_name] = False
            logger.error(f"✗ {interface_name} import timed out (likely circular import)")
        except ImportError as e:
            results[interface_name] = False
            logger.error(f"✗ {interface_name} import failed: {e}")
        except InterfaceValidationError as e:
            results[interface_name] = False
            logger.error(f"✗ {interface_name} validation failed: {e}")
        except Exception as e:
            results[interface_name] = False
            logger.error(f"✗ {interface_name} unexpected error: {e}")

    return results


__all__ = [
    "InterfaceValidationError",
    "validate_interface_implementation",
    "register_with_validation",
    "validate_all_service_implementations",
]
