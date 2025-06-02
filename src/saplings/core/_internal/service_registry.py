from __future__ import annotations

"""
Service registry for tracking service initialization status.

This module provides a registry for tracking the initialization status of services
and managing dependencies between services.
"""

import logging
import threading
from typing import Dict, List, Optional, Set

from saplings.core._internal.exceptions import InitializationError
from saplings.core.events import CoreEvent, CoreEventType, get_event_bus
from saplings.core.lifecycle import ServiceLifecycle, ServiceState

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Registry for tracking service initialization status.

    This class provides a centralized registry for tracking the initialization
    status of services and managing dependencies between services.
    """

    _instance: Optional[ServiceRegistry] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        """Initialize the service registry."""
        self._services: Dict[str, ServiceLifecycle] = {}
        self._event_bus = get_event_bus()

    def __new__(cls) -> ServiceRegistry:
        """Create a singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServiceRegistry, cls).__new__(cls)
                cls._instance.__init__()
                logger.info("ServiceRegistry initialized")
            return cls._instance

    def register_service(self, service_name: str) -> ServiceLifecycle:
        """
        Register a service with the registry.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            ServiceLifecycle: Lifecycle object for the service

        """
        with self._lock:
            if service_name in self._services:
                return self._services[service_name]

            # Create a new lifecycle object
            lifecycle = ServiceLifecycle(service_name)
            self._services[service_name] = lifecycle

            # Publish event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.SERVICE_INITIALIZED,
                    data={"service_name": service_name},
                    source="ServiceRegistry",
                )
            )

            logger.info(f"Registered service: {service_name}")
            return lifecycle

    def get_service_lifecycle(self, service_name: str) -> Optional[ServiceLifecycle]:
        """
        Get the lifecycle object for a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Optional[ServiceLifecycle]: Lifecycle object for the service, or None if not found

        """
        return self._services.get(service_name)

    def add_dependency(self, service_name: str, dependency_name: str) -> None:
        """
        Add a dependency between services.

        Args:
        ----
            service_name: Name of the dependent service
            dependency_name: Name of the dependency service

        Raises:
        ------
            InitializationError: If either service is not registered

        """
        with self._lock:
            # Get the service lifecycle objects
            service = self.get_service_lifecycle(service_name)
            dependency = self.get_service_lifecycle(dependency_name)

            # Check if both services exist
            if service is None:
                raise InitializationError(f"Service not registered: {service_name}")
            if dependency is None:
                raise InitializationError(f"Dependency not registered: {dependency_name}")

            # Add the dependency
            service.add_dependency(dependency_name)
            dependency.add_dependent(service_name)

            # Check for circular dependencies
            try:
                self.get_initialization_order()
            except Exception as e:
                # Remove the dependency to maintain graph integrity
                service._dependencies.remove(dependency_name)
                dependency._dependents.remove(service_name)

                # Publish event for circular dependency
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_CYCLE_DETECTED,
                        data={
                            "service_name": service_name,
                            "dependency_name": dependency_name,
                            "error": str(e),
                        },
                        source="ServiceRegistry",
                    )
                )

                raise InitializationError(
                    f"Adding dependency {service_name} -> {dependency_name} would create a circular dependency: {e}"
                )

            # If the dependency is already in READY state, mark it as ready for the service
            if dependency.state == ServiceState.READY:
                service.mark_dependency_ready(dependency_name)

                # Publish event
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_READY,
                        data={
                            "service_name": service_name,
                            "dependency_name": dependency_name,
                        },
                        source="ServiceRegistry",
                    )
                )

            # If the dependency is in DISPOSED state with an error, mark it as failed
            elif dependency.state == ServiceState.DISPOSED and dependency.initialization_error:
                service.mark_dependency_failed(
                    dependency_name, error=dependency.initialization_error
                )

                # Publish event
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_FAILED,
                        data={
                            "service_name": service_name,
                            "dependency_name": dependency_name,
                            "error": str(dependency.initialization_error),
                        },
                        source="ServiceRegistry",
                    )
                )

            logger.info(f"Added dependency: {service_name} -> {dependency_name}")

    def get_dependencies(self, service_name: str) -> Set[str]:
        """
        Get the dependencies of a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Set[str]: Set of dependency service names

        Raises:
        ------
            InitializationError: If the service is not registered

        """
        with self._lock:
            service = self.get_service_lifecycle(service_name)
            if service is None:
                raise InitializationError(f"Service not registered: {service_name}")
            return service.dependencies

    def get_dependents(self, service_name: str) -> Set[str]:
        """
        Get the dependents of a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Set[str]: Set of dependent service names

        Raises:
        ------
            InitializationError: If the service is not registered

        """
        with self._lock:
            service = self.get_service_lifecycle(service_name)
            if service is None:
                raise InitializationError(f"Service not registered: {service_name}")
            return service.dependents

    def get_service_state(self, service_name: str) -> Optional[ServiceState]:
        """
        Get the state of a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Optional[ServiceState]: State of the service, or None if not found

        """
        with self._lock:
            service = self.get_service_lifecycle(service_name)
            if service is None:
                return None
            return service.state

    def get_initialization_order(self) -> List[str]:
        """
        Get the order in which services should be initialized.

        This method uses a topological sort to determine the initialization order
        based on service dependencies. If circular dependencies are detected,
        it raises an InitializationError with details about the cycle.

        Returns
        -------
            List[str]: List of service names in initialization order

        Raises
        ------
            InitializationError: If circular dependencies are detected

        """
        with self._lock:
            # Create a copy of the dependency graph
            graph: Dict[str, Set[str]] = {}
            for service_name, lifecycle in self._services.items():
                graph[service_name] = lifecycle.dependencies.copy()

            # Perform topological sort
            result: List[str] = []
            no_deps: List[str] = [service_name for service_name, deps in graph.items() if not deps]

            while no_deps:
                # Add a node with no dependencies to the result
                service_name = no_deps.pop(0)
                result.append(service_name)

                # Remove the node from the graph
                for deps in graph.values():
                    if service_name in deps:
                        deps.remove(service_name)

                # Find new nodes with no dependencies
                no_deps.extend(
                    [
                        name
                        for name, deps in graph.items()
                        if not deps and name not in result and name not in no_deps
                    ]
                )

            # Check for circular dependencies
            if len(result) != len(self._services):
                remaining = set(self._services.keys()) - set(result)
                logger.warning(f"Circular dependencies detected: {remaining}")

                # Find the cycle
                cycle = self._find_cycle(remaining)
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    error_msg = f"Circular dependency detected: {cycle_str}"
                    logger.error(error_msg)

                    # Publish event
                    self._event_bus.publish(
                        CoreEvent(
                            event_type=CoreEventType.DEPENDENCY_CYCLE_DETECTED,
                            data={
                                "cycle": cycle,
                                "error": error_msg,
                            },
                            source="ServiceRegistry",
                        )
                    )

                    raise InitializationError(error_msg)

            return result

    def _find_cycle(self, services: Set[str]) -> List[str]:
        """
        Find a cycle in the dependency graph.

        Args:
        ----
            services: Set of service names to check for cycles

        Returns:
        -------
            List[str]: List of service names forming a cycle, or empty list if no cycle found

        """
        if not services:
            return []

        # Create a subgraph with only the services in question
        subgraph: Dict[str, Set[str]] = {}
        for service_name in services:
            lifecycle = self._services.get(service_name)
            if lifecycle:
                # Only include dependencies that are also in the services set
                subgraph[service_name] = {dep for dep in lifecycle.dependencies if dep in services}

        # Use DFS to find a cycle
        visited: Set[str] = set()
        path: List[str] = []
        path_set: Set[str] = set()

        def dfs(node: str) -> Optional[List[str]]:
            if node in path_set:
                # Found a cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for neighbor in subgraph.get(node, set()):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            path.pop()
            path_set.remove(node)
            return None

        # Try to find a cycle starting from each node
        for node in subgraph:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle

        return []

    def get_all_services(self) -> Dict[str, ServiceState]:
        """
        Get all registered services and their states.

        Returns
        -------
            Dict[str, ServiceState]: Dictionary of service names and states

        """
        with self._lock:
            return {name: lifecycle.state for name, lifecycle in self._services.items()}


# Global service registry instance
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """
    Get the global service registry instance.

    Returns
    -------
        ServiceRegistry: The global service registry instance

    """
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry
