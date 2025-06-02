from __future__ import annotations

"""
Service Dependency Graph module for Saplings.

This module provides a graph structure for tracking service dependencies
and determining initialization order.
"""

import logging
from typing import Dict, List, Optional, Set

from saplings.core._internal.exceptions import InitializationError

logger = logging.getLogger(__name__)


class CircularDependencyError(InitializationError):
    """Error raised when a circular dependency is detected."""

    def __init__(self, cycle: List[str]) -> None:
        """
        Initialize the error with the detected cycle.

        Args:
        ----
            cycle: List of service names forming the cycle

        """
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")
        self.cycle = cycle


class ServiceDependencyGraph:
    """
    Graph structure for tracking service dependencies.

    This class provides methods for adding dependencies between services,
    detecting circular dependencies, and determining initialization order.
    """

    def __init__(self) -> None:
        """Initialize the dependency graph."""
        # Map of service name to set of dependency names
        self._dependencies: Dict[str, Set[str]] = {}
        # Map of service name to set of dependent names
        self._dependents: Dict[str, Set[str]] = {}
        # Initialization status of services
        self._initialized: Dict[str, bool] = {}

    def add_service(self, service_name: str) -> None:
        """
        Add a service to the graph.

        Args:
        ----
            service_name: Name of the service to add

        """
        if service_name not in self._dependencies:
            self._dependencies[service_name] = set()
        if service_name not in self._dependents:
            self._dependents[service_name] = set()
        if service_name not in self._initialized:
            self._initialized[service_name] = False

    def add_dependency(self, service_name: str, dependency_name: str) -> None:
        """
        Add a dependency between services.

        Args:
        ----
            service_name: Name of the service
            dependency_name: Name of the dependency

        Raises:
        ------
            CircularDependencyError: If adding the dependency would create a cycle

        """
        # Add services if they don't exist
        self.add_service(service_name)
        self.add_service(dependency_name)

        # Add the dependency
        self._dependencies[service_name].add(dependency_name)
        self._dependents[dependency_name].add(service_name)

        # Check for cycles
        cycle = self._detect_cycle()
        if cycle:
            # Remove the dependency to maintain graph integrity
            self._dependencies[service_name].remove(dependency_name)
            self._dependents[dependency_name].remove(service_name)
            raise CircularDependencyError(cycle)

    def get_dependencies(self, service_name: str) -> Set[str]:
        """
        Get the dependencies of a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Set of dependency names

        Raises:
        ------
            KeyError: If the service is not in the graph

        """
        if service_name not in self._dependencies:
            raise KeyError(f"Service not in graph: {service_name}")
        return self._dependencies[service_name].copy()

    def get_dependents(self, service_name: str) -> Set[str]:
        """
        Get the dependents of a service.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Set of dependent names

        Raises:
        ------
            KeyError: If the service is not in the graph

        """
        if service_name not in self._dependents:
            raise KeyError(f"Service not in graph: {service_name}")
        return self._dependents[service_name].copy()

    def mark_initialized(self, service_name: str) -> None:
        """
        Mark a service as initialized.

        Args:
        ----
            service_name: Name of the service

        Raises:
        ------
            KeyError: If the service is not in the graph

        """
        if service_name not in self._initialized:
            raise KeyError(f"Service not in graph: {service_name}")
        self._initialized[service_name] = True

    def is_initialized(self, service_name: str) -> bool:
        """
        Check if a service is initialized.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            True if the service is initialized, False otherwise

        Raises:
        ------
            KeyError: If the service is not in the graph

        """
        if service_name not in self._initialized:
            raise KeyError(f"Service not in graph: {service_name}")
        return self._initialized[service_name]

    def get_initialization_order(self) -> List[str]:
        """
        Get the order in which services should be initialized.

        This method performs a topological sort of the dependency graph.

        Returns
        -------
            List of service names in initialization order

        Raises
        ------
            CircularDependencyError: If the graph contains a cycle

        """
        # Create a copy of the dependency graph
        graph: Dict[str, Set[str]] = {}
        for service_name, deps in self._dependencies.items():
            graph[service_name] = deps.copy()

        # Perform topological sort
        result: List[str] = []
        no_deps: List[str] = [service_name for service_name, deps in graph.items() if not deps]

        while no_deps:
            # Get a service with no dependencies
            service_name = no_deps.pop(0)
            result.append(service_name)

            # Remove the service from the graph
            for deps in graph.values():
                if service_name in deps:
                    deps.remove(service_name)

            # Find new services with no dependencies
            for s, deps in graph.items():
                if s not in result and not deps and s not in no_deps:
                    no_deps.append(s)

        # Check if all services were processed
        if len(result) != len(self._dependencies):
            # There must be a cycle
            cycle = self._detect_cycle()
            if cycle:
                raise CircularDependencyError(cycle)
            else:
                # This should not happen, but just in case
                raise InitializationError("Could not determine initialization order")

        return result

    def _detect_cycle(self) -> Optional[List[str]]:
        """
        Detect a cycle in the dependency graph.

        Returns
        -------
            List of service names forming the cycle, or None if no cycle exists

        """
        # Use DFS to detect cycles
        visited: Set[str] = set()
        path: List[str] = []
        path_set: Set[str] = set()

        def dfs(service_name: str) -> Optional[List[str]]:
            """
            Depth-first search to detect cycles.

            Args:
            ----
                service_name: Current service name

            Returns:
            -------
                List of service names forming the cycle, or None if no cycle exists

            """
            if service_name in path_set:
                # Found a cycle
                cycle_start = path.index(service_name)
                return path[cycle_start:] + [service_name]

            if service_name in visited:
                return None

            visited.add(service_name)
            path.append(service_name)
            path_set.add(service_name)

            for dependency in self._dependencies[service_name]:
                cycle = dfs(dependency)
                if cycle:
                    return cycle

            path.pop()
            path_set.remove(service_name)
            return None

        # Check each service
        for service_name in self._dependencies:
            if service_name not in visited:
                cycle = dfs(service_name)
                if cycle:
                    return cycle

        return None
