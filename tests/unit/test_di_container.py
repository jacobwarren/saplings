from __future__ import annotations

"""
Unit tests for the dependency injection container.
"""


from unittest.mock import MagicMock

import pytest

from saplings.core.model_registry import ModelRegistry
from saplings.di import container, inject, register, reset_container


class TestDependencyInjection:
    """Test the dependency injection container."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Reset container before each test
        reset_container()

    def teardown_method(self) -> None:
        """Clean up after test."""
        # Reset container after each test
        reset_container()

    def test_singleton_resolution(self) -> None:
        """Test that singletons are resolved correctly."""
        # Get two instances of the same singleton
        registry1 = container.resolve(ModelRegistry)
        registry2 = container.resolve(ModelRegistry)

        # They should be the same instance
        assert registry1 is registry2

    def test_transient_resolution(self) -> None:
        """Test that transients are resolved correctly."""
        # Register a transient service
        container.register(str, factory=lambda: "transient", singleton=False)

        # Get two instances
        instance1 = container.resolve(str)
        instance2 = container.resolve(str)

        # They should be different instances
        assert instance1 == instance2  # Same value
        assert instance1 is not instance2  # Different objects

    def test_constructor_injection(self) -> None:
        """Test constructor injection."""

        # Create a mock dependency
        mock_dependency = MagicMock()
        container.register(dict, instance=mock_dependency)

        # Create a class with constructor injection
        class ServiceWithDependency:
            def __init__(self, storage: dict) -> None:
                self.storage = storage

        # Register the service
        container.register(
            ServiceWithDependency,
            factory=lambda storage: ServiceWithDependency(storage=storage),
            storage=dict,
        )

        # Resolve the service
        service = container.resolve(ServiceWithDependency)

        # Check that it has the dependency
        assert service.storage is mock_dependency

    def test_decorator_registration(self) -> None:
        """Test the @register decorator."""

        # Mock dependency
        mock_service = MagicMock()
        container.register(list, instance=mock_service)

        # Define a class with the decorator
        @register(TestDependencyInjection)
        class DecoratedService:
            def __init__(self, items: list) -> None:
                self.items = items

        # Resolve the service
        service = container.resolve(TestDependencyInjection)

        # Check that it's the right type and has dependencies
        assert isinstance(service, DecoratedService)
        assert service.items is mock_service

    def test_inject_decorator(self) -> None:
        """Test the @inject decorator."""

        # Register mock services
        mock_registry = MagicMock()
        container.register(ModelRegistry, instance=mock_registry)

        # Create a function with the decorator
        @inject
        def example_function(registry: ModelRegistry):
            return registry.get_model_count()

        # Call the function
        mock_registry.get_model_count.return_value = 42
        result = example_function()

        # Check that dependencies were injected
        assert result == 42
        mock_registry.get_model_count.assert_called_once()

    def test_reset_container(self) -> None:
        """Test resetting the container."""
        # Register something
        container.register(str, instance="test")

        # Reset the container
        reset_container()

        # Registration should be gone
        with pytest.raises(Exception):
            container.resolve(str)

    def test_interface_registration(self) -> None:
        """Test registering and resolving interfaces."""

        # Define an interface and implementation
        class IService:
            def do_something(self):
                pass

        class ServiceImpl(IService):
            def do_something(self):
                return "done"

        # Register the implementation for the interface
        container.register(IService, factory=lambda: ServiceImpl())

        # Resolve the interface
        service = container.resolve(IService)

        # Check that we got the implementation
        assert isinstance(service, ServiceImpl)
        assert service.do_something() == "done"

    def test_dependency_chain(self) -> None:
        """Test resolving a chain of dependencies."""

        # Define classes with dependencies
        class ServiceA:
            def get_value(self):
                return "A"

        class ServiceB:
            def __init__(self, a: ServiceA) -> None:
                self.a = a

            def get_value(self):
                return f"B -> {self.a.get_value()}"

        class ServiceC:
            def __init__(self, b: ServiceB) -> None:
                self.b = b

            def get_value(self):
                return f"C -> {self.b.get_value()}"

        # Register services
        container.register(ServiceA)
        container.register(ServiceB, factory=lambda a: ServiceB(a=a), a=ServiceA)
        container.register(ServiceC, factory=lambda b: ServiceC(b=b), b=ServiceB)

        # Resolve the top-level service
        service_c = container.resolve(ServiceC)

        # Check that the dependency chain works
        assert service_c.get_value() == "C -> B -> A"
