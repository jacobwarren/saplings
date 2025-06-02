"""
Test elimination of dynamic imports in Agent API.

This test verifies that Task 4.1 is complete - the Agent API no longer uses
importlib.import_module or other dynamic import patterns.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestEliminateDynamicImports:
    """Test that dynamic imports have been eliminated from Agent API."""

    def test_agent_api_module_no_dynamic_imports(self):
        """Test that the agent API module doesn't use dynamic imports."""
        # Import the module and check its source
        import saplings.api.agent as agent_module

        # Get the source file
        source_file = inspect.getfile(agent_module)

        # Read the source code
        with open(source_file) as f:
            source_code = f.read()

        # Check for dynamic import patterns
        assert (
            "importlib.import_module" not in source_code
        ), "Agent API module should not use importlib.import_module"

        # Check for dynamic getter functions
        dynamic_getters = [
            "_get_agent()",
            "_get_agent_builder()",
            "_get_agent_config()",
            "_get_agent_facade()",
            "_get_agent_facade_builder()",
        ]

        for getter in dynamic_getters:
            assert (
                getter not in source_code
            ), f"Agent API module should not use dynamic getter {getter}"

    def test_agent_class_no_complex_new_method(self):
        """Test that Agent class doesn't use complex __new__ method."""
        from saplings.api.agent import Agent

        # Check that Agent doesn't use complex __new__ method
        if hasattr(Agent, "__new__"):
            # Get the source of __new__ if it exists
            try:
                source = inspect.getsource(Agent.__new__)
                # Should not contain dynamic imports or complex logic
                assert (
                    "importlib.import_module" not in source
                ), "Agent.__new__ should not use dynamic imports"
                assert (
                    "_get_agent()" not in source
                ), "Agent.__new__ should not use dynamic getter functions"
            except (OSError, TypeError):
                # If we can't get source, it's likely a built-in __new__, which is fine
                # TypeError occurs when trying to get source of built-in methods
                pass

    def test_agent_builder_no_complex_new_method(self):
        """Test that AgentBuilder class doesn't use complex __new__ method."""
        from saplings.api.agent import AgentBuilder

        # Check that AgentBuilder doesn't use complex __new__ method
        if hasattr(AgentBuilder, "__new__"):
            try:
                source = inspect.getsource(AgentBuilder.__new__)
                assert (
                    "importlib.import_module" not in source
                ), "AgentBuilder.__new__ should not use dynamic imports"
                assert (
                    "_get_agent_builder()" not in source
                ), "AgentBuilder.__new__ should not use dynamic getter functions"
            except (OSError, TypeError):
                # TypeError occurs when trying to get source of built-in methods
                pass

    def test_agent_config_no_complex_new_method(self):
        """Test that AgentConfig class doesn't use complex __new__ method."""
        from saplings.api.agent import AgentConfig

        # Check that AgentConfig doesn't use complex __new__ method
        if hasattr(AgentConfig, "__new__"):
            try:
                source = inspect.getsource(AgentConfig.__new__)
                assert (
                    "importlib.import_module" not in source
                ), "AgentConfig.__new__ should not use dynamic imports"
                assert (
                    "_get_agent_config()" not in source
                ), "AgentConfig.__new__ should not use dynamic getter functions"
            except (OSError, TypeError):
                # TypeError occurs when trying to get source of built-in methods
                pass

    def test_agent_facade_no_complex_new_method(self):
        """Test that AgentFacade class doesn't use complex __new__ method."""
        from saplings.api.agent import AgentFacade

        # Check that AgentFacade doesn't use complex __new__ method
        if hasattr(AgentFacade, "__new__"):
            try:
                source = inspect.getsource(AgentFacade.__new__)
                assert (
                    "importlib.import_module" not in source
                ), "AgentFacade.__new__ should not use dynamic imports"
                assert (
                    "_get_agent_facade()" not in source
                ), "AgentFacade.__new__ should not use dynamic getter functions"
            except (OSError, TypeError):
                # TypeError occurs when trying to get source of built-in methods
                pass

    def test_agent_facade_builder_no_complex_new_method(self):
        """Test that AgentFacadeBuilder class doesn't use complex __new__ method."""
        from saplings.api.agent import AgentFacadeBuilder

        # Check that AgentFacadeBuilder doesn't use complex __new__ method
        if hasattr(AgentFacadeBuilder, "__new__"):
            try:
                source = inspect.getsource(AgentFacadeBuilder.__new__)
                assert (
                    "importlib.import_module" not in source
                ), "AgentFacadeBuilder.__new__ should not use dynamic imports"
                assert (
                    "_get_agent_facade_builder()" not in source
                ), "AgentFacadeBuilder.__new__ should not use dynamic getter functions"
            except (OSError, TypeError):
                # TypeError occurs when trying to get source of built-in methods
                pass

    def test_agent_api_uses_static_imports(self):
        """Test that Agent API uses static imports instead of dynamic ones."""
        import saplings.api.agent as agent_module

        # Get the source file
        source_file = inspect.getfile(agent_module)

        # Read the source code
        with open(source_file) as f:
            source_code = f.read()

        # Should use static imports
        expected_imports = [
            "from saplings._internal.agent_class import Agent as _Agent",
            "from saplings._internal.agent_builder_module import AgentBuilder as _AgentBuilder",
            "from saplings._internal.agent_config import AgentConfig as _AgentConfig",
            "from saplings._internal._agent_facade import AgentFacade as _AgentFacade",
            "from saplings._internal._agent_facade_builder import AgentFacadeBuilder as _AgentFacadeBuilder",
        ]

        for expected_import in expected_imports:
            assert (
                expected_import in source_code
            ), f"Agent API should use static import: {expected_import}"

    def test_agent_classes_use_direct_inheritance(self):
        """Test that Agent API classes use direct inheritance pattern."""
        from saplings.api.agent import (
            Agent,
            AgentBuilder,
            AgentConfig,
            AgentFacade,
            AgentFacadeBuilder,
        )

        # Check that classes use direct inheritance from internal modules
        classes_to_check = [
            (Agent, "saplings._internal.agent_class"),
            (AgentBuilder, "saplings._internal.agent_builder_module"),
            (AgentConfig, "saplings._internal.agent_config"),
            (AgentFacade, "saplings._internal._agent_facade"),
            (AgentFacadeBuilder, "saplings._internal._agent_facade_builder"),
        ]

        for cls, expected_base_module in classes_to_check:
            # Check that the class has a base class from the expected internal module
            base_modules = [base.__module__ for base in cls.__bases__ if base != object]
            assert expected_base_module in base_modules, (
                f"{cls.__name__} should inherit from a class in {expected_base_module}, "
                f"but base modules are: {base_modules}"
            )
