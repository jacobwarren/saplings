# Testing Guide

This guide covers testing strategies and best practices for Saplings applications, including unit testing, integration testing, and performance testing.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Agent Testing](#agent-testing)
- [Tool Testing](#tool-testing)
- [Memory Testing](#memory-testing)
- [Performance Testing](#performance-testing)
- [Test Automation](#test-automation)

## Testing Overview

### Testing Philosophy

Saplings applications should be thoroughly tested across multiple layers:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test scalability and resource usage

### Testing Framework

All tests use pytest as the primary testing framework:

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from saplings import Agent, AgentConfig, AgentBuilder
```

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_agent.py
│   ├── test_memory.py
│   └── test_tools.py
├── integration/            # Integration tests
│   ├── test_workflows.py
│   └── test_services.py
├── e2e/                    # End-to-end tests
│   └── test_complete_flows.py
└── conftest.py            # Shared fixtures
```

## Unit Testing

### Basic Agent Testing

```python
# tests/unit/test_agent.py
import pytest
from unittest.mock import Mock, patch
from saplings import Agent, AgentConfig

class TestAgent:
    """Unit tests for Agent class."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized with valid config."""
        config = AgentConfig(
            provider="mock",
            model_name="test-model"
        )
        
        agent = Agent(config)
        assert agent is not None
        assert agent.config.provider == "mock"
        assert agent.config.model_name == "test-model"
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_agent_run_basic(self, mock_llm):
        """Test basic agent run functionality."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate.return_value = "Test response"
        mock_llm.return_value = mock_instance
        
        # Create agent
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Test run
        result = await agent.run("Test prompt")
        
        assert result == "Test response"
        mock_instance.generate.assert_called_once()
    
    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Valid config
        valid_config = AgentConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        validation_result = valid_config.validate()
        assert validation_result.is_valid
        
        # Invalid config
        invalid_config = AgentConfig(
            provider="invalid_provider",
            model_name="invalid-model"
        )
        validation_result = invalid_config.validate()
        assert not validation_result.is_valid
        assert len(validation_result.suggestions) > 0
```

### Mock Configuration

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, patch
from saplings import AgentConfig

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return AgentConfig(
        provider="mock",
        model_name="test-model",
        memory_path="./test_memory",
        output_dir="./test_output"
    )

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    with patch("saplings.models._internal.interfaces.LLM") as mock:
        mock_instance = Mock()
        mock_instance.generate.return_value = "Mock response"
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_agent(mock_config, mock_llm):
    """Create a mock agent for testing."""
    from saplings import Agent
    return Agent(mock_config)
```

### Component Testing

```python
# tests/unit/test_agent_builder.py
import pytest
from unittest.mock import patch
from saplings import AgentBuilder

class TestAgentBuilder:
    """Test AgentBuilder functionality."""
    
    def test_builder_pattern(self):
        """Test builder pattern works correctly."""
        builder = AgentBuilder()
        
        # Chain builder methods
        builder = builder.for_openai("gpt-4o") \
                        .with_tools(["PythonInterpreterTool"]) \
                        .with_memory_path("./test_memory")
        
        assert builder._config_params["provider"] == "openai"
        assert builder._config_params["model_name"] == "gpt-4o"
    
    @patch("saplings.agent_builder.Agent")
    def test_builder_build(self, mock_agent):
        """Test builder creates agent correctly."""
        builder = AgentBuilder()
        builder.for_openai("gpt-4o")
        
        agent = builder.build()
        
        mock_agent.assert_called_once()
    
    def test_builder_validation(self):
        """Test builder validates required parameters."""
        builder = AgentBuilder()
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            builder.build()  # Should fail without provider/model
```

## Integration Testing

### Service Integration Tests

```python
# tests/integration/test_service_interactions.py
import pytest
from unittest.mock import MagicMock
from saplings import Agent, AgentConfig
from saplings.api.core.interfaces import IExecutionService, IMemoryManager

class TestServiceInteractions:
    """Test interactions between services."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return AgentConfig(
            provider="mock",
            model_name="test-model",
            enable_monitoring=False,
            enable_self_healing=False
        )
    
    def test_agent_creation_workflow(self, test_config):
        """Test complete agent creation workflow."""
        try:
            agent = Agent(test_config)
            assert agent is not None
            
            # Verify services are available
            assert hasattr(agent, "_facade")
            assert agent._facade is not None
            
        except Exception as e:
            pytest.fail(f"Agent creation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_memory_service_integration(self, mock_agent):
        """Test memory service integration."""
        # Add a document
        test_document = "This is a test document for integration testing."
        result = mock_agent.add_document(test_document)
        
        assert result is not None
        
        # Test retrieval
        search_result = await mock_agent.run("Find test documents")
        assert search_result is not None
```

### Workflow Testing

```python
# tests/integration/test_workflows.py
import pytest
from unittest.mock import Mock, patch
from saplings import Agent, AgentConfig

class TestWorkflows:
    """Test complete workflows."""
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_document_analysis_workflow(self, mock_llm):
        """Test document analysis workflow."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate.return_value = "Document analysis complete"
        mock_llm.return_value = mock_instance
        
        # Create agent
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Add documents
        documents = [
            "Document 1 content",
            "Document 2 content",
            "Document 3 content"
        ]
        
        for doc in documents:
            agent.add_document(doc)
        
        # Analyze documents
        result = await agent.run("Analyze all documents and provide summary")
        
        assert result == "Document analysis complete"
        assert mock_instance.generate.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_tool_integration_workflow(self, mock_agent):
        """Test tool integration workflow."""
        from saplings.api.tools import Tool
        
        # Create test tool
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "A test tool"
            
            def execute(self, input_data: str) -> str:
                return f"Processed: {input_data}"
        
        # Register tool
        test_tool = TestTool()
        mock_agent.register_tool(test_tool)
        
        # Test tool usage
        result = await mock_agent.run("Use the test tool")
        assert result is not None
```

## Agent Testing

### Agent Behavior Testing

```python
# tests/unit/test_agent_behavior.py
import pytest
from unittest.mock import Mock, patch
from saplings import Agent, AgentConfig

class TestAgentBehavior:
    """Test agent behavior patterns."""
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_agent_handles_empty_input(self, mock_llm):
        """Test agent handles empty input gracefully."""
        mock_instance = Mock()
        mock_instance.generate.return_value = "Please provide input"
        mock_llm.return_value = mock_instance
        
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        result = await agent.run("")
        assert result is not None
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_agent_handles_long_input(self, mock_llm):
        """Test agent handles long input."""
        mock_instance = Mock()
        mock_instance.generate.return_value = "Long input processed"
        mock_llm.return_value = mock_instance
        
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        long_input = "x" * 10000  # Very long input
        result = await agent.run(long_input)
        
        assert result == "Long input processed"
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling."""
        config = AgentConfig(provider="mock", model_name="test-model")
        
        with patch("saplings.models._internal.interfaces.LLM") as mock_llm:
            mock_instance = Mock()
            mock_instance.generate.side_effect = Exception("Test error")
            mock_llm.return_value = mock_instance
            
            agent = Agent(config)
            
            with pytest.raises(Exception):
                await agent.run("This should cause an error")
```

### Configuration Testing

```python
# tests/unit/test_agent_config.py
import pytest
from saplings import AgentConfig

class TestAgentConfig:
    """Test agent configuration."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = AgentConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = AgentConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        assert config.enable_monitoring is True
        assert config.enable_gasa is False
        assert config.memory_path is None
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = AgentConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        result = config.validate()
        assert result.is_valid
        assert result.message == "Configuration is valid"
    
    def test_config_validation_invalid_provider(self):
        """Test invalid provider fails validation."""
        config = AgentConfig(
            provider="invalid_provider",
            model_name="gpt-4o"
        )
        
        result = config.validate()
        assert not result.is_valid
        assert "Unsupported provider" in result.message
    
    def test_config_validation_invalid_model(self):
        """Test invalid model fails validation."""
        config = AgentConfig(
            provider="openai",
            model_name="invalid-model"
        )
        
        result = config.validate()
        assert not result.is_valid
        assert len(result.suggestions) > 0
```

## Tool Testing

### Basic Tool Testing

```python
# tests/unit/test_tools.py
import pytest
from saplings.api.tools import Tool, validate_tool

class TestTools:
    """Test tool functionality."""
    
    def test_tool_creation(self):
        """Test basic tool creation."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "A test tool"
            
            def execute(self, input_data: str) -> str:
                return f"Processed: {input_data}"
        
        tool = TestTool()
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
    
    def test_tool_execution(self):
        """Test tool execution."""
        class CalculatorTool(Tool):
            def execute(self, expression: str) -> str:
                try:
                    result = eval(expression)  # Note: unsafe, only for testing
                    return f"Result: {result}"
                except Exception as e:
                    return f"Error: {e}"
        
        calc_tool = CalculatorTool()
        result = calc_tool.execute("2 + 3")
        assert "Result: 5" in result
    
    def test_tool_validation(self):
        """Test tool validation."""
        class ValidTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "valid_tool"
                self.description = "A valid tool"
            
            def execute(self, input_data: str) -> str:
                return "Valid"
        
        tool = ValidTool()
        assert validate_tool(tool) is True
```

### Tool Integration Testing

```python
# tests/integration/test_tool_integration.py
import pytest
from unittest.mock import Mock, patch
from saplings import Agent, AgentConfig
from saplings.api.tools import Tool

class TestToolIntegration:
    """Test tool integration with agents."""
    
    class MockTool(Tool):
        """Mock tool for testing."""
        
        def __init__(self):
            super().__init__()
            self.name = "mock_tool"
            self.description = "Mock tool for testing"
            self.call_count = 0
        
        def execute(self, input_data: str) -> str:
            self.call_count += 1
            return f"Mock result {self.call_count}: {input_data}"
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_agent_tool_registration(self, mock_llm):
        """Test agent can register and use tools."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.generate.return_value = "Tool result"
        mock_llm.return_value = mock_instance
        
        # Create agent and tool
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        tool = self.MockTool()
        
        # Register tool
        agent.register_tool(tool)
        
        # Test tool usage
        result = await agent.run("Use the mock tool")
        assert result == "Tool result"
    
    def test_multiple_tool_registration(self):
        """Test registering multiple tools."""
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Create multiple tools
        tool1 = self.MockTool()
        tool1.name = "tool1"
        
        tool2 = self.MockTool()
        tool2.name = "tool2"
        
        # Register tools
        agent.register_tool(tool1)
        agent.register_tool(tool2)
        
        # Verify registration
        # Note: This would depend on actual implementation
        assert agent is not None  # Basic check
```

## Memory Testing

### Memory Store Testing

```python
# tests/unit/test_memory.py
import pytest
from saplings.api.memory import MemoryStore, Document, DocumentMetadata

class TestMemoryStore:
    """Test memory store functionality."""
    
    def test_memory_store_creation(self):
        """Test memory store can be created."""
        memory_store = MemoryStore()
        assert memory_store is not None
    
    def test_document_creation(self):
        """Test document creation."""
        doc = Document(
            content="Test document content",
            metadata=DocumentMetadata(
                source="test.txt",
                content_type="text/plain"
            )
        )
        
        assert doc.content == "Test document content"
        assert doc.metadata.source == "test.txt"
    
    def test_add_document(self):
        """Test adding document to memory store."""
        memory_store = MemoryStore()
        
        doc = memory_store.add_document(
            "Test content",
            metadata={"source": "test.txt"}
        )
        
        assert doc is not None
        assert doc.content == "Test content"
    
    def test_search_documents(self):
        """Test searching documents."""
        memory_store = MemoryStore()
        
        # Add test documents
        memory_store.add_document("Python programming tutorial")
        memory_store.add_document("JavaScript web development")
        memory_store.add_document("Python data science")
        
        # Search for Python documents
        results = memory_store.search("Python")
        
        # Should find documents containing "Python"
        assert len(results) >= 2
```

### Memory Integration Testing

```python
# tests/integration/test_memory_integration.py
import pytest
from unittest.mock import patch
from saplings import Agent, AgentConfig

class TestMemoryIntegration:
    """Test memory integration with agents."""
    
    @pytest.mark.asyncio
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_agent_memory_operations(self, mock_llm):
        """Test agent memory operations."""
        # Setup mock
        mock_instance = mock_llm.return_value
        mock_instance.generate.return_value = "Found documents"
        
        # Create agent
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Add documents
        documents = [
            "Document about artificial intelligence",
            "Document about machine learning",
            "Document about deep learning"
        ]
        
        for doc in documents:
            agent.add_document(doc)
        
        # Search documents
        result = await agent.run("Find documents about AI")
        assert result == "Found documents"
    
    def test_memory_persistence(self):
        """Test memory persistence across agent instances."""
        config1 = AgentConfig(
            provider="mock",
            model_name="test-model",
            memory_path="./test_memory_persist"
        )
        agent1 = Agent(config1)
        
        # Add document with first agent
        agent1.add_document("Persistent document")
        
        # Create second agent with same memory path
        config2 = AgentConfig(
            provider="mock",
            model_name="test-model",
            memory_path="./test_memory_persist"
        )
        agent2 = Agent(config2)
        
        # Second agent should have access to the document
        # Note: This would depend on actual implementation
        assert agent2 is not None
```

## Performance Testing

### Load Testing

```python
# tests/performance/test_load.py
import pytest
import asyncio
import time
from unittest.mock import patch
from saplings import Agent, AgentConfig

class TestPerformance:
    """Performance testing."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @patch("saplings.models._internal.interfaces.LLM")
    async def test_concurrent_requests(self, mock_llm):
        """Test agent can handle concurrent requests."""
        # Setup mock
        mock_instance = mock_llm.return_value
        mock_instance.generate.return_value = "Concurrent response"
        
        # Create agent
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Create concurrent tasks
        async def run_task(task_id):
            return await agent.run(f"Task {task_id}")
        
        tasks = [run_task(i) for i in range(10)]
        
        # Measure execution time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        assert len(results) == 10
        assert all(result == "Concurrent response" for result in results)
        
        # Performance check - should complete within reasonable time
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete in under 5 seconds
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple agents
        agents = []
        for i in range(10):
            config = AgentConfig(provider="mock", model_name="test-model")
            agent = Agent(config)
            agents.append(agent)
        
        # Add documents to each agent
        for agent in agents:
            for j in range(100):
                agent.add_document(f"Document {j}")
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Cleanup
        del agents
        gc.collect()
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.performance
    def test_agent_creation_performance(self):
        """Test agent creation is fast enough."""
        start_time = time.time()
        
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        creation_time = time.time() - start_time
        
        # Agent creation should be fast (< 1 second)
        assert creation_time < 1.0, f"Agent creation took {creation_time:.3f}s"
        assert agent is not None
```

### Stress Testing

```python
# tests/performance/test_stress.py
import pytest
import time
from unittest.mock import patch
from saplings import Agent, AgentConfig

class TestStress:
    """Stress testing."""
    
    @pytest.mark.stress
    @patch("saplings.models._internal.interfaces.LLM")
    def test_large_document_handling(self, mock_llm):
        """Test handling of large documents."""
        mock_instance = mock_llm.return_value
        mock_instance.generate.return_value = "Large document processed"
        
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        # Create large document (1MB)
        large_content = "x" * (1024 * 1024)
        
        start_time = time.time()
        agent.add_document(large_content)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should handle large document within reasonable time
        assert processing_time < 10.0, f"Large document processing took {processing_time:.3f}s"
    
    @pytest.mark.stress
    def test_many_documents(self):
        """Test handling many documents."""
        config = AgentConfig(provider="mock", model_name="test-model")
        agent = Agent(config)
        
        start_time = time.time()
        
        # Add many small documents
        for i in range(1000):
            agent.add_document(f"Document {i} content")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle many documents efficiently
        assert processing_time < 30.0, f"Adding 1000 documents took {processing_time:.3f}s"
```

## Test Automation

### Pytest Configuration

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    stress: Stress tests
    slow: Slow tests
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from saplings import AgentConfig, Agent

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="saplings_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_config(test_data_dir):
    """Create test configuration."""
    return AgentConfig(
        provider="mock",
        model_name="test-model",
        memory_path=str(test_data_dir / "memory"),
        output_dir=str(test_data_dir / "output")
    )

@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    with patch("saplings.models._internal.interfaces.LLM") as mock:
        mock_instance = Mock()
        mock_instance.generate.return_value = "Mock response"
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def test_agent(test_config, mock_llm_response):
    """Create test agent."""
    return Agent(test_config)
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit -v --cov=saplings --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
```

### Test Utilities

```python
# tests/utils.py
import time
from typing import Callable, Any
from unittest.mock import Mock

def assert_timing(func: Callable, max_time: float, *args, **kwargs) -> Any:
    """Assert function completes within specified time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < max_time, f"Function took {execution_time:.3f}s, expected < {max_time}s"
    
    return result

def create_mock_agent_config(**overrides):
    """Create mock agent config with overrides."""
    from saplings import AgentConfig
    
    defaults = {
        "provider": "mock",
        "model_name": "test-model",
        "memory_path": "./test_memory",
        "output_dir": "./test_output"
    }
    defaults.update(overrides)
    
    return AgentConfig(**defaults)

def create_test_documents(count: int = 5) -> list[str]:
    """Create test documents for testing."""
    return [f"Test document {i} content" for i in range(count)]
```

This testing guide provides comprehensive coverage of testing strategies for Saplings applications using the actual testing patterns and frameworks available in the codebase.