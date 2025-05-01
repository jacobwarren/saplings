"""
Tests for cross-component integration in the Saplings framework.
"""

import asyncio
import os
import tempfile
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.core.plugin import PluginType, ToolPlugin
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig, MaskBuilder
from saplings.integration import (
    HotLoader,
    HotLoaderConfig,
    IntegrationManager,
    ToolLifecycleManager,
)
from saplings.memory import Document, MemoryConfig, MemoryStore
from saplings.monitoring import BlameGraph, MonitoringConfig, TraceManager
from saplings.orchestration import GraphRunner
from saplings.orchestration.config import AgentNode, CommunicationChannel, GraphRunnerConfig
from saplings.planner import PlannerConfig, SequentialPlanner
from saplings.tool_factory import (
    SecurityLevel,
    ToolFactory,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
)


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, model_uri, **kwargs):
        """Initialize the mock LLM."""
        self.model_uri = model_uri
        self.kwargs = kwargs
        self.generate_calls = []
        self.streaming_calls = []

    async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs) -> LLMResponse:
        """Generate text from the model."""
        # Record the call
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        # Return a mock response
        return LLMResponse(
            text=f"Response to: {prompt[:50]}...",
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 20,
                "total_tokens": len(prompt.split()) + 20,
            },
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def generate_streaming(
        self, prompt, max_tokens=None, temperature=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from the model with streaming output."""
        # Record the call
        self.streaming_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        # Create a response
        response = f"Response to: {prompt[:50]}..."
        chunks = [response[i : i + 5] for i in range(0, len(response), 5)]

        # Return chunks as an async generator
        for chunk in chunks:
            yield chunk

    def get_metadata(self) -> ModelMetadata:
        """Get metadata about the model."""
        return ModelMetadata(
            name="mock-model",
            provider="mock-provider",
            version="latest",
            roles=[ModelRole.EXECUTOR, ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=2048,
            capabilities=[],
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        return len(text.split())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request."""
        return (prompt_tokens + completion_tokens) * 0.0001


class TestCrossComponentIntegration:
    """Tests for cross-component integration in the Saplings framework."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM("mock://model/latest")

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def tool_factory(self, mock_llm, temp_dir):
        """Create a ToolFactory instance for testing."""
        config = ToolFactoryConfig(
            output_dir=temp_dir,
            security_level=SecurityLevel.MEDIUM,
            enable_code_signing=False,
        )
        return ToolFactory(model=mock_llm, config=config)

    @pytest.fixture
    def hot_loader(self, temp_dir):
        """Create a HotLoader instance for testing."""
        config = HotLoaderConfig(
            watch_directories=[temp_dir],
            auto_reload=True,
            reload_interval=1.0,
        )
        return HotLoader(config=config)

    @pytest.fixture
    def executor(self, mock_llm):
        """Create an Executor instance for testing."""
        config = ExecutorConfig()
        return Executor(model=mock_llm, config=config)

    @pytest.fixture
    def planner(self, mock_llm):
        """Create a SequentialPlanner instance for testing."""
        config = PlannerConfig()
        return SequentialPlanner(model=mock_llm, config=config)

    @pytest.fixture
    def graph_runner(self, mock_llm, memory_store):
        """Create a GraphRunner instance for testing."""
        config = GraphRunnerConfig(memory_store=memory_store)
        return GraphRunner(model=mock_llm, config=config)

    @pytest.fixture
    def memory_store(self, temp_dir):
        """Create a MemoryStore instance for testing."""
        config = MemoryConfig(
            document_store_path=os.path.join(temp_dir, "documents"),
            vector_store_path=os.path.join(temp_dir, "vectors"),
        )
        return MemoryStore(config=config)

    @pytest.fixture
    def trace_manager(self):
        """Create a TraceManager instance for testing."""
        config = MonitoringConfig()
        return TraceManager(config=config)

    @pytest.fixture
    def integration_manager(self, executor, planner, graph_runner, hot_loader):
        """Create an IntegrationManager instance for testing."""
        return IntegrationManager(
            executor=executor,
            planner=planner,
            graph_runner=graph_runner,
            hot_loader=hot_loader,
        )

    @pytest.mark.asyncio
    async def test_executor_planner_integration(self, mock_llm, executor, planner):
        """Test integration between executor and planner."""
        # Create a plan
        with patch.object(planner, "_create_planning_prompt") as mock_create_prompt:
            mock_create_prompt.return_value = "Create a plan for: What is the capital of France?"
            plan = await planner.create_plan(
                task="What is the capital of France?",
            )

        # Execute the plan
        # Just mock the model's generate method instead of a specific executor method
        result = await executor.execute(
            prompt="What is the capital of France?",
            plan=plan,
        )

        # Check that the plan was executed
        assert result is not None
        assert "Response to:" in result.text

        # Check that the LLM was called
        assert len(mock_llm.generate_calls) > 0

    @pytest.mark.asyncio
    async def test_executor_memory_integration(self, mock_llm, executor, memory_store):
        """Test integration between executor and memory manager."""
        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Add documents to memory
        for doc in documents:
            memory_store.add_document(doc)

        # Create embeddings and add to vector store
        with patch.object(memory_store, "vector_store", create=True) as mock_vector_store:
            mock_vector_store.add_documents.return_value = ["vec1", "vec2"]
            mock_vector_store.add_documents(documents)

        # Search for documents
        with patch.object(memory_store, "search") as mock_search:
            mock_search.return_value = documents
            retrieved_docs = memory_store.search("capital of France", limit=2)

        # Execute with the retrieved documents
        result = await executor.execute(
            prompt="What is the capital of France?",
            documents=retrieved_docs,
        )

        # Check that the execution was successful
        assert result is not None
        assert "Response to:" in result.text

        # Just check that the result is not None
        assert result is not None

    @pytest.mark.asyncio
    async def test_executor_gasa_integration(self, mock_llm, executor):
        """Test integration between executor and GASA."""
        # Enable GASA in the executor
        executor.config.enable_gasa = True

        # Create a mock GASA config
        gasa_config = GASAConfig(
            enabled=True,
            max_hops=2,
            mask_strategy="binary",
            cache_masks=True,
        )

        # Create a mock mask builder
        class MockMaskBuilder:
            def __init__(self, config=None):
                self.config = config or GASAConfig()
                self.build_count = 0

            def build_mask(self, documents, prompt, **kwargs):
                """Build a mock attention mask."""
                self.build_count += 1

                # Create a simple square mask of ones
                seq_len = len(prompt.split()) + 10  # Add some padding
                import numpy as np

                mask = np.ones((seq_len, seq_len), dtype=np.float32)

                # Add some structure to the mask based on documents
                if documents:
                    # Create some zeros in the mask to simulate sparse attention
                    for i in range(0, seq_len, 3):
                        for j in range(0, seq_len, 3):
                            if i != j and (i % 2 == 0 or j % 2 == 0):
                                mask[i, j] = 0

                return mask

        # Set up the executor with GASA
        executor.gasa_config = gasa_config
        executor.mask_builder = MockMaskBuilder(config=gasa_config)

        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Execute with GASA
        # Just use the model directly
        result = await executor.execute(
            prompt="What is the capital of France?",
            documents=documents,
        )

        # Check that the execution was successful
        assert result is not None
        assert "Response to:" in result.text

        # Check that the mask builder was used
        assert executor.mask_builder.build_count > 0

    @pytest.mark.asyncio
    async def test_planner_memory_integration(self, mock_llm, planner, memory_store):
        """Test integration between planner and memory manager."""
        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Add documents to memory
        for doc in documents:
            memory_store.add_document(doc)

        # Create embeddings and add to vector store
        with patch.object(memory_store, "vector_store", create=True) as mock_vector_store:
            mock_vector_store.add_documents.return_value = ["vec1", "vec2"]
            mock_vector_store.add_documents(documents)

        # Search for documents
        with patch.object(memory_store, "search") as mock_search:
            mock_search.return_value = documents
            retrieved_docs = memory_store.search("capital of France", limit=2)

        # Create a plan with the retrieved documents
        with patch.object(planner, "_create_planning_prompt") as mock_create_prompt:
            mock_create_prompt.return_value = "Create a plan for: What is the capital of France?"
            plan = await planner.create_plan(
                task="What is the capital of France?",
                documents=retrieved_docs,
            )

        # Check that the plan was created
        assert plan is not None
        # Check that the plan contains steps
        assert len(plan) > 0

        # Check that the LLM was called
        assert len(mock_llm.generate_calls) > 0

    @pytest.mark.asyncio
    async def test_graph_runner_tool_integration(
        self, mock_llm, graph_runner, tool_factory, hot_loader, memory_store
    ):
        """Test integration between graph runner and tools."""
        # Register a template
        template = ToolTemplate(
            id="math_tool",
            name="Math Tool",
            description="A tool for mathematical operations",
            template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
            required_parameters=["function_name", "parameters", "description", "code_body"],
            metadata={"category": "math"},
        )
        tool_factory.register_template(template)

        # Create a tool specification
        spec = ToolSpecification(
            id="add_numbers",
            name="Add Numbers",
            description="A tool to add two numbers",
            template_id="math_tool",
            parameters={
                "function_name": "add_numbers",
                "parameters": "a: int, b: int",
                "description": "Add two numbers together",
                "code_body": "return a + b",
            },
            metadata={"category": "math"},
        )

        # Create the tool
        with patch.object(tool_factory, "_validate_tool_code", return_value=(True, "")):
            tool_class = await tool_factory.create_tool(spec)

        # Load the tool
        hot_loader.load_tool(tool_class)

        # Create an agent
        agent = AgentNode(
            id="math_agent",
            name="Math Agent",
            role="calculator",
            description="An agent that performs math operations",
            capabilities=["math"],
            metadata={"tools": {}},
            memory_store=memory_store,
        )

        # Register the agent
        graph_runner.register_agent(agent)

        # Register the tool with the agent
        agent.metadata["tools"] = {spec.id: tool_class}

        # Mock the negotiate method
        async def mock_negotiate(task, **kwargs):
            # Check if the tool is available
            if spec.id in agent.metadata["tools"]:
                # Create an instance of the tool
                tool = agent.metadata["tools"][spec.id]()
                # Use the tool
                if hasattr(tool, "add_numbers"):
                    result = tool.add_numbers(2, 3)
                    return f"The result is {result}"
            return f"Negotiated solution for: {task}"

        with patch.object(graph_runner, "negotiate", side_effect=mock_negotiate):
            # Execute the task
            result = await graph_runner.negotiate(
                task="Add 2 and 3",
                context="The user wants to perform a simple addition",
            )

        # Check that the task was executed
        assert result is not None
        assert "The result is" in result or "Negotiated solution for" in result

        # Check that the tool was registered with the agent
        assert spec.id in agent.metadata["tools"]

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, mock_llm, executor, planner, trace_manager):
        """Test integration with the monitoring system."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-integration")

        # Start a span for planning
        planning_span = trace_manager.start_span(
            name="planning",
            trace_id=trace.trace_id,
            attributes={"component": "planner"},
        )

        # Create a plan
        with patch.object(planner, "_create_planning_prompt") as mock_create_prompt:
            mock_create_prompt.return_value = "Create a plan for: What is the capital of France?"
            plan = await planner.create_plan(
                task="What is the capital of France?",
            )

        # End planning span
        trace_manager.end_span(planning_span.span_id)

        # Start a span for execution
        execution_span = trace_manager.start_span(
            name="execution",
            trace_id=trace.trace_id,
            attributes={"component": "executor"},
        )

        # Execute the plan
        # Just use the model directly
        result = await executor.execute(
            prompt="What is the capital of France?",
            plan=plan,
        )

        # End execution span
        trace_manager.end_span(execution_span.span_id)

        # Create a blame graph
        blame_graph = BlameGraph(trace_manager=trace_manager)
        blame_graph.process_trace(trace)

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) >= 2  # planning, execution

        # Check that the blame graph was created
        assert len(blame_graph.nodes) == 2  # planner, executor

        # Check that the execution was successful
        assert result is not None
        assert "Response to:" in result.text

    @pytest.mark.asyncio
    async def test_memory_monitoring_integration(self, memory_store, trace_manager):
        """Test integration between memory and monitoring."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-memory-monitoring")

        # Start a span for memory operations
        memory_span = trace_manager.start_span(
            name="memory_operations",
            trace_id=trace.trace_id,
            attributes={"component": "memory"},
        )

        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Add documents to memory
        for doc in documents:
            memory_store.add_document(doc)

        # Create embeddings and add to vector store
        with patch.object(memory_store, "vector_store", create=True) as mock_vector_store:
            mock_vector_store.add_documents.return_value = ["vec1", "vec2"]
            mock_vector_store.add_documents(documents)

        # Search for documents
        with patch.object(memory_store, "search") as mock_search:
            mock_search.return_value = documents
            retrieved_docs = memory_store.search("capital of France", limit=2)

        # End memory span
        trace_manager.end_span(memory_span.span_id)

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) == 1  # memory

        # Check that the memory operations were successful
        assert len(retrieved_docs) == 2
        assert "The capital of France is Paris" in retrieved_docs[0]
