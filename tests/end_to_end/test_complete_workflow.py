"""
End-to-end tests for complete workflows in the Saplings framework.
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional, Any, AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.core.plugin import PluginType, ToolPlugin
from saplings.executor import Executor, ExecutorConfig
from saplings.orchestration import GraphRunner, GraphRunnerConfig, AgentNode
from saplings.orchestration.config import CommunicationChannel
from saplings.planner import SequentialPlanner, PlannerConfig
from saplings.tool_factory import (
    ToolFactory,
    ToolFactoryConfig,
    ToolTemplate,
    ToolSpecification,
    SecurityLevel,
)
from saplings.integration import (
    HotLoader,
    HotLoaderConfig,
    ToolLifecycleManager,
    IntegrationManager,
)
from saplings.memory import (
    MemoryConfig,
    Document,
    MemoryStore,
    VectorStore,
    InMemoryVectorStore,
)
from saplings.monitoring import (
    TraceManager,
    MonitoringConfig,
    BlameGraph,
)


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, model_uri, **kwargs):
        """Initialize the mock LLM."""
        self.model_uri = model_uri
        self.kwargs = kwargs
        self.generate_calls = []
        self.streaming_calls = []

    async def generate(
        self, prompt, max_tokens=None, temperature=None, **kwargs
    ) -> LLMResponse:
        """Generate text from the model."""
        # Record the call
        self.generate_calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kwargs": kwargs,
        })

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
        self.streaming_calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kwargs": kwargs,
        })

        # Create a response
        response = f"Response to: {prompt[:50]}..."
        chunks = [response[i:i+5] for i in range(0, len(response), 5)]

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


class TestCompleteWorkflow:
    """Tests for complete workflows in the Saplings framework."""

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
    def graph_runner(self, mock_llm):
        """Create a GraphRunner instance for testing."""
        config = GraphRunnerConfig()
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
    async def test_single_agent_workflow(self, mock_llm, executor, planner, memory_store, trace_manager):
        """Test a complete workflow with a single agent."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-workflow-1")

        # Start a span for the workflow
        workflow_span = trace_manager.start_span(
            name="single_agent_workflow",
            trace_id=trace.trace_id,
            attributes={"component": "workflow"},
        )

        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Start a span for memory operations
        memory_span = trace_manager.start_span(
            name="memory_operations",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "memory"},
        )

        # Add documents to memory
        for doc in documents:
            memory_store.add_document(doc)

        # Create embeddings and add to vector store
        with patch.object(memory_store, "vector_store", create=True) as mock_vector_store:
            mock_vector_store.add_documents.return_value = ["vec1", "vec2"]
            mock_vector_store.add_documents(documents)

        # End memory span
        trace_manager.end_span(memory_span.span_id)

        # Start a span for planning
        planning_span = trace_manager.start_span(
            name="planning",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "planner"},
        )

        # Create a plan
        plan = await planner.create_plan(
            task="What is the capital of France and what is it known as?",
        )

        # End planning span
        trace_manager.end_span(planning_span.span_id)

        # Start a span for execution
        execution_span = trace_manager.start_span(
            name="execution",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "executor"},
        )

        # Execute the task with the retrieved documents
        with patch.object(memory_store, "search") as mock_search:
            mock_search.return_value = documents
            result = await executor.execute(
                prompt="What is the capital of France and what is it known as?",
                documents=documents,
            )

        # End execution span
        trace_manager.end_span(execution_span.span_id)

        # End workflow span
        trace_manager.end_span(workflow_span.span_id)

        # Check that the workflow was executed
        assert result is not None
        assert "Response to:" in result.text

        # Check that the LLM was called
        assert len(mock_llm.generate_calls) > 0

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) == 4  # workflow, memory, planning, execution

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, mock_llm, graph_runner, trace_manager):
        """Test a complete workflow with multiple agents."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-workflow-2")

        # Start a span for the workflow
        workflow_span = trace_manager.start_span(
            name="multi_agent_workflow",
            trace_id=trace.trace_id,
            attributes={"component": "workflow"},
        )

        # Start a span for agent creation
        agent_span = trace_manager.start_span(
            name="agent_creation",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "graph_runner"},
        )

        # Create agents
        researcher = AgentNode(
            id="researcher",
            name="Research Agent",
            role="researcher",
            description="An agent that researches information",
            capabilities=["research"],
            metadata={},
        )

        writer = AgentNode(
            id="writer",
            name="Writing Agent",
            role="writer",
            description="An agent that writes content",
            capabilities=["writing"],
            metadata={},
        )

        # Register agents with the graph runner
        graph_runner.register_agent(researcher)
        graph_runner.register_agent(writer)

        # Create a channel between the agents
        channel = CommunicationChannel(
            source_id=researcher.id,
            target_id=writer.id,
            channel_type="data",
            description="Channel from researcher to writer",
        )

        # Add the channel to the graph runner
        graph_runner.channels.append(channel)

        # End agent span
        trace_manager.end_span(agent_span.span_id)

        # Start a span for execution
        execution_span = trace_manager.start_span(
            name="graph_execution",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "graph_runner"},
        )

        # Mock the negotiate method
        async def mock_negotiate(task, **kwargs):
            return f"Negotiated solution for: {task}"

        with patch.object(graph_runner, "negotiate", side_effect=mock_negotiate):
            # Execute the task
            result = await graph_runner.negotiate(
                task="Research and write about the capital of France",
                context="The user wants information about Paris",
            )

        # End execution span
        trace_manager.end_span(execution_span.span_id)

        # End workflow span
        trace_manager.end_span(workflow_span.span_id)

        # Check that the workflow was executed
        assert result is not None
        assert "Negotiated solution for" in result

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) == 3  # workflow, agent, execution

    @pytest.mark.asyncio
    async def test_tool_integration_workflow(self, mock_llm, tool_factory, hot_loader, integration_manager, trace_manager):
        """Test a complete workflow with tool integration."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-workflow-3")

        # Start a span for the workflow
        workflow_span = trace_manager.start_span(
            name="tool_integration_workflow",
            trace_id=trace.trace_id,
            attributes={"component": "workflow"},
        )

        # Start a span for tool creation
        tool_span = trace_manager.start_span(
            name="tool_creation",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "tool_factory"},
        )

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

        # End tool span
        trace_manager.end_span(tool_span.span_id)

        # Start a span for tool loading
        loading_span = trace_manager.start_span(
            name="tool_loading",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "hot_loader"},
        )

        # Load the tool
        integration_manager.hot_loader.load_tool(tool_class)

        # Register tools with all components
        integration_manager.register_tools_with_executor()
        integration_manager.register_tools_with_planner()

        # End loading span
        trace_manager.end_span(loading_span.span_id)

        # Start a span for execution
        execution_span = trace_manager.start_span(
            name="tool_execution",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "executor"},
        )

        # Mock the executor's execute method
        async def mock_execute(prompt, **kwargs):
            return f"Executed task: {prompt}"

        with patch.object(integration_manager.executor, "execute", side_effect=mock_execute):
            # Execute a task using the executor
            result = await integration_manager.executor.execute(
                prompt="Add 2 and 3",
                tools=integration_manager.executor.tools,
            )

        # End execution span
        trace_manager.end_span(execution_span.span_id)

        # End workflow span
        trace_manager.end_span(workflow_span.span_id)

        # Check that the workflow was executed
        assert result is not None
        assert "Executed task" in result

        # Check that the tool was registered
        assert spec.id in integration_manager.executor.tools
        assert spec.id in integration_manager.planner.tools

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) == 4  # workflow, tool, loading, execution

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_monitoring(
        self, mock_llm, executor, planner, graph_runner, memory_store, trace_manager
    ):
        """Test a complete end-to-end workflow with monitoring."""
        # Create a trace for monitoring
        trace = trace_manager.create_trace(trace_id="test-workflow-4")

        # Start a span for the workflow
        workflow_span = trace_manager.start_span(
            name="end_to_end_workflow",
            trace_id=trace.trace_id,
            attributes={"component": "workflow"},
        )

        # Create document strings instead of Document objects
        documents = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]

        # Start a span for memory operations
        memory_span = trace_manager.start_span(
            name="memory_operations",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "memory"},
        )

        # Add documents to memory
        for doc in documents:
            memory_store.add_document(doc)

        # Create embeddings and add to vector store
        with patch.object(memory_store, "vector_store", create=True) as mock_vector_store:
            mock_vector_store.add_documents.return_value = ["vec1", "vec2"]
            mock_vector_store.add_documents(documents)

        # End memory span
        trace_manager.end_span(memory_span.span_id)

        # Start a span for planning
        planning_span = trace_manager.start_span(
            name="planning",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "planner"},
        )

        # Create a plan
        plan = await planner.create_plan(
            task="What is the capital of France and what is it known as?",
        )

        # End planning span
        trace_manager.end_span(planning_span.span_id)

        # Start a span for agent creation
        agent_span = trace_manager.start_span(
            name="agent_creation",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "graph_runner"},
        )

        # Create agents
        researcher = AgentNode(
            id="researcher",
            name="Research Agent",
            role="researcher",
            description="An agent that researches information",
            capabilities=["research"],
            metadata={},
        )

        writer = AgentNode(
            id="writer",
            name="Writing Agent",
            role="writer",
            description="An agent that writes content",
            capabilities=["writing"],
            metadata={},
        )

        # Register agents with the graph runner
        graph_runner.register_agent(researcher)
        graph_runner.register_agent(writer)

        # Create a channel between the agents
        channel = CommunicationChannel(
            source_id=researcher.id,
            target_id=writer.id,
            channel_type="data",
            description="Channel from researcher to writer",
        )

        # Add the channel to the graph runner
        graph_runner.channels.append(channel)

        # End agent span
        trace_manager.end_span(agent_span.span_id)

        # Start a span for execution
        execution_span = trace_manager.start_span(
            name="execution",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "executor"},
        )

        # Execute the task with the retrieved documents
        with patch.object(memory_store, "search") as mock_search:
            mock_search.return_value = documents
            result = await executor.execute(
                prompt="What is the capital of France and what is it known as?",
                documents=documents,
            )

        # End execution span
        trace_manager.end_span(execution_span.span_id)

        # Start a span for graph execution
        graph_span = trace_manager.start_span(
            name="graph_execution",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "graph_runner"},
        )

        # Mock the negotiate method
        async def mock_negotiate(task, **kwargs):
            return f"Negotiated solution for: {task}"

        with patch.object(graph_runner, "negotiate", side_effect=mock_negotiate):
            # Execute the task
            graph_result = await graph_runner.negotiate(
                task="Research and write about the capital of France",
                context="The user wants information about Paris",
            )

        # End graph span
        trace_manager.end_span(graph_span.span_id)

        # Start a span for monitoring
        monitoring_span = trace_manager.start_span(
            name="monitoring",
            trace_id=trace.trace_id,
            parent_id=workflow_span.span_id,
            attributes={"component": "monitoring"},
        )

        # Create a blame graph
        blame_graph = BlameGraph(trace_manager=trace_manager)
        blame_graph.process_trace(trace)

        # Identify error sources (there shouldn't be any in this test)
        error_sources = blame_graph.identify_error_sources(min_error_rate=0.1, min_call_count=1)

        # End monitoring span
        trace_manager.end_span(monitoring_span.span_id)

        # End workflow span
        trace_manager.end_span(workflow_span.span_id)

        # Check that the workflow was executed
        assert result is not None
        assert "Response to:" in result.text
        assert graph_result is not None
        assert "Negotiated solution for" in graph_result

        # Check that the LLM was called
        assert len(mock_llm.generate_calls) > 0

        # Check that the trace was created
        assert trace.trace_id in trace_manager.traces
        assert len(trace.spans) == 7  # workflow, memory, planning, agent, execution, graph, monitoring

        # Check that there are no error sources
        assert len(error_sources) == 0
