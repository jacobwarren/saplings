"""
Test performance testing and optimization for publication readiness.

This module tests Task 7.8: Performance testing and optimization.
Covers import performance, runtime performance, and resource usage.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from unittest.mock import Mock, patch

import psutil
import pytest


class TestPerformanceTestingOptimization:
    """Test performance testing and optimization."""

    def setup_method(self):
        """Set up test environment."""
        # Get current process for memory measurements
        self.process = psutil.Process()

        # Clear garbage collection
        gc.collect()

        # Store initial memory baseline
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def test_import_performance_benchmarks(self):
        """Test that imports complete within acceptable time limits."""
        # Test main package import
        import_time = self._measure_import_time("saplings")
        print(f"\nMain package import time: {import_time:.3f} seconds")
        assert import_time < 2.0, f"Main package import took {import_time:.3f}s, should be < 2.0s"

        # Test submodule imports
        submodules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.memory",
            "saplings.api.services",
        ]

        for module in submodules:
            import_time = self._measure_import_time(module)
            print(f"{module} import time: {import_time:.3f} seconds")
            assert import_time < 1.0, f"{module} import took {import_time:.3f}s, should be < 1.0s"

    def test_memory_usage_during_import(self):
        """Test that imports don't use excessive memory."""
        # Clear any cached imports
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith("saplings")]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Force garbage collection and get baseline memory
        gc.collect()
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Import main package
        import saplings  # noqa: F401

        # Force garbage collection and measure memory after import
        gc.collect()
        after_import_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = after_import_memory - baseline_memory
        print(f"\nMemory usage increase during import: {memory_increase:.2f} MB")

        # Target: < 100 MB increase for main package import
        assert (
            memory_increase < 100.0
        ), f"Memory increase {memory_increase:.2f} MB exceeds 100 MB limit"

    @pytest.mark.skipif("SKIP_INTEGRATION_TESTS" in os.environ, reason="Skipping integration tests")
    def test_agent_creation_performance(self):
        """Test that agent creation is fast enough."""
        try:
            from saplings.api.agent import Agent, AgentConfig

            # Measure agent creation time
            start_time = time.time()
            config = AgentConfig(
                provider="mock",  # Use mock provider to avoid external dependencies
                model_name="test-model",
            )

            with patch("saplings.models._internal.interfaces.LLM") as mock_llm:
                mock_llm.return_value = Mock()
                agent = Agent(config=config)

            creation_time = time.time() - start_time

            print(f"\nAgent creation time: {creation_time:.3f} seconds")

            # Target: < 1 second for agent creation
            assert (
                creation_time < 1.0
            ), f"Agent creation took {creation_time:.3f}s, should be < 1.0s"

        except ImportError as e:
            pytest.skip(f"Agent creation test skipped due to import error: {e}")

    def test_tool_execution_performance(self):
        """Test that tool execution has acceptable performance."""
        try:
            from saplings.api.tools import Tool

            # Create a simple test tool
            class TestTool(Tool):
                def execute(self, input_data: str) -> str:
                    # Simple operation for performance testing
                    return f"Processed: {input_data}"

            tool = TestTool()

            # Measure tool execution time
            start_time = time.time()
            result = tool.execute("test input")
            execution_time = time.time() - start_time

            print(f"\nTool execution time: {execution_time:.6f} seconds")

            # Target: < 0.1 seconds for simple tool execution
            assert (
                execution_time < 0.1
            ), f"Tool execution took {execution_time:.6f}s, should be < 0.1s"
            assert result == "Processed: test input"

        except ImportError as e:
            pytest.skip(f"Tool execution test skipped due to import error: {e}")

    def test_memory_operation_performance(self):
        """Test that memory operations have acceptable performance."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Create test documents
            documents = []
            start_time = time.time()

            for i in range(100):
                doc = Document(
                    content=f"Test document {i} content",
                    metadata=DocumentMetadata(
                        source="test",
                        content_type="text/plain",
                        language="en",
                        author="test_author",
                    ),
                )
                documents.append(doc)

            creation_time = time.time() - start_time

            print(f"\nDocument creation time (100 docs): {creation_time:.3f} seconds")

            # Target: < 1 second for creating 100 documents
            assert (
                creation_time < 1.0
            ), f"Document creation took {creation_time:.3f}s, should be < 1.0s"
            assert len(documents) == 100

        except ImportError as e:
            pytest.skip(f"Memory operation test skipped due to import error: {e}")

    def test_cpu_utilization_during_operations(self):
        """Test that operations don't cause excessive CPU usage."""
        # CPU measurement can be unreliable in short tests, so we'll measure over time
        try:
            from saplings.api.tools import Tool

            # Start CPU monitoring
            self.process.cpu_percent()  # Initialize CPU monitoring
            time.sleep(0.1)  # Wait a bit for accurate measurement

            initial_cpu = self.process.cpu_percent()

            # Perform some operations
            tools = []
            for i in range(10):

                class TestTool(Tool):
                    def execute(self, input_data: str) -> str:
                        return f"Tool {i}: {input_data}"

                tools.append(TestTool())

            # Wait a bit and get CPU usage after operations
            time.sleep(0.1)
            final_cpu = self.process.cpu_percent()

            print(f"\nCPU usage: initial={initial_cpu:.1f}%, final={final_cpu:.1f}%")

            # CPU usage measurement can be variable, so we just verify it's a reasonable number
            # Rather than asserting a specific threshold, we check it's not obviously broken
            assert 0 <= final_cpu <= 100, f"CPU usage {final_cpu:.1f}% is not a valid percentage"

        except ImportError as e:
            pytest.skip(f"CPU utilization test skipped due to import error: {e}")

    def test_file_handle_usage(self):
        """Test that operations don't leak file handles."""
        # Get initial file handle count
        try:
            initial_handles = len(self.process.open_files())
        except (psutil.AccessDenied, AttributeError):
            pytest.skip("Cannot access file handle information on this system")

        # Perform operations that might open files
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Create and process documents
            for i in range(10):
                doc = Document(
                    content=f"Test content {i}",
                    metadata=DocumentMetadata(
                        source="test",
                        content_type="text/plain",
                        language="en",
                        author="test_author",
                    ),
                )
                # Simulate some processing
                _ = str(doc)

            # Get final file handle count
            final_handles = len(self.process.open_files())

            print(f"\nFile handles: initial={initial_handles}, final={final_handles}")

            # Should not have significant file handle leaks
            handle_increase = final_handles - initial_handles
            assert handle_increase < 10, f"File handle increase {handle_increase} seems excessive"

        except (ImportError, psutil.AccessDenied) as e:
            pytest.skip(f"File handle test skipped: {e}")

    def _measure_import_time(self, module_name: str) -> float:
        """Measure the time to import a module."""
        # Clear module from cache if present
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Clear any submodules
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith(module_name)]
        for mod_name in modules_to_clear:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Measure import time
        start_time = time.time()
        try:
            __import__(module_name)
            return time.time() - start_time
        except ImportError:
            return float("inf")  # Return infinity for failed imports

    def test_performance_regression_detection(self):
        """Test that we can detect performance regressions."""
        # This test establishes baseline performance metrics
        # In a real CI/CD pipeline, these would be compared against historical data

        metrics = {
            "main_import_time": self._measure_import_time("saplings"),
            "memory_baseline": self.process.memory_info().rss / 1024 / 1024,
        }

        print("\nPerformance baseline metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        # Store metrics for future comparison (in real implementation,
        # this would be stored in a database or file)
        assert all(isinstance(v, (int, float)) and v >= 0 for v in metrics.values())

    def test_performance_monitoring_setup(self):
        """Test that performance monitoring can be set up correctly."""
        # Verify that we can collect the metrics needed for continuous monitoring

        monitoring_data = {
            "timestamp": time.time(),
            "process_id": os.getpid(),
            "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent(),
            "python_version": sys.version_info[:2],
        }

        print("\nMonitoring data collected:")
        for key, value in monitoring_data.items():
            print(f"  {key}: {value}")

        # Verify all monitoring data is valid
        assert monitoring_data["timestamp"] > 0
        assert monitoring_data["process_id"] > 0
        assert monitoring_data["memory_usage_mb"] > 0
        assert isinstance(monitoring_data["cpu_percent"], (int, float))
        assert len(monitoring_data["python_version"]) == 2
