"""
Agent module for Saplings.

This module provides a high-level Agent class that integrates all components
of the Saplings framework, including:
- Memory and retrieval
- Planning and execution
- Validation and judging
- Self-healing and improvement
- Monitoring and tracing
- Graph-Aligned Sparse Attention (GASA)

The Agent class serves as the main entry point for using Saplings,
providing a clean, intuitive API while leveraging the full power
of the underlying components.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig

from saplings.memory import (
    DependencyGraph,
    Document,
    DocumentMetadata,
    MemoryStore
)
from saplings.memory.config import MemoryConfig
from saplings.memory.vector_store import get_vector_store
from saplings.memory.indexer import get_indexer
from saplings.monitoring import (
    MonitoringConfig,
    TraceManager,
    BlameGraph,
    TraceViewer
)
from saplings.planner import (
    SequentialPlanner,
    PlannerConfig,
    PlanStep
)
from saplings.retrieval import (
    CascadeRetriever,
    RetrievalConfig,
    TFIDFRetriever,
    EmbeddingRetriever,
    GraphExpander,
    EntropyCalculator
)
from saplings.self_heal import (
    PatchGenerator,
    SuccessPairCollector,
    AdapterManager
)
from saplings.tool_factory import ToolFactory, ToolFactoryConfig
from saplings.validator.registry import get_validator_registry

# Configure logging
logger = logging.getLogger(__name__)

class AgentConfig:
    """
    Configuration for the Agent class.

    This class centralizes all configuration options for the Agent,
    making it easy to customize behavior while providing sensible defaults.
    """

    def __init__(
        self,
        model_uri: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
        enable_self_healing: bool = True,
        enable_tool_factory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        gasa_max_hops: int = 2,
        retrieval_entropy_threshold: float = 0.1,
        retrieval_max_documents: int = 10,
        planner_budget_strategy: str = "token_count",
        executor_verification_strategy: str = "judge",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: List[str] = None,
        **model_parameters
    ):
        """
        Initialize the agent configuration.

        Args:
            model_uri: URI of the model to use (e.g., "openai://gpt-4", "anthropic://claude-3-opus")
            provider: Model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: Model name
            memory_path: Path to store agent memory
            output_dir: Directory to save outputs
            enable_gasa: Whether to enable Graph-Aligned Sparse Attention
            enable_monitoring: Whether to enable monitoring and tracing
            enable_self_healing: Whether to enable self-healing capabilities
            enable_tool_factory: Whether to enable dynamic tool creation
            max_tokens: Maximum number of tokens for model responses
            temperature: Temperature for model generation
            gasa_max_hops: Maximum number of hops for GASA mask
            retrieval_entropy_threshold: Entropy threshold for retrieval termination
            retrieval_max_documents: Maximum number of documents to retrieve
            planner_budget_strategy: Strategy for budget allocation
            executor_verification_strategy: Strategy for output verification
            tool_factory_sandbox_enabled: Whether to enable sandbox for tool execution
            allowed_imports: List of allowed imports for dynamic tools
            **model_parameters: Additional model parameters
        """
        # Handle model specification
        if model_uri is None and provider is not None and model_name is not None:
            # Create a model URI from provider and model_name
            params_str = ""
            if model_parameters:
                params_list = [f"{k}={v}" for k, v in model_parameters.items()]
                params_str = "?" + "&".join(params_list)
            model_uri = f"{provider}://{model_name}{params_str}"
        elif model_uri is None:
            raise ValueError("Either 'model_uri' or both 'provider' and 'model_name' must be provided")

        self.model_uri = model_uri
        self.provider = provider
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.memory_path = memory_path
        self.output_dir = output_dir
        self.enable_gasa = enable_gasa
        self.enable_monitoring = enable_monitoring
        self.enable_self_healing = enable_self_healing
        self.enable_tool_factory = enable_tool_factory
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gasa_max_hops = gasa_max_hops
        self.retrieval_entropy_threshold = retrieval_entropy_threshold
        self.retrieval_max_documents = retrieval_max_documents
        self.planner_budget_strategy = planner_budget_strategy
        self.executor_verification_strategy = executor_verification_strategy
        self.tool_factory_sandbox_enabled = tool_factory_sandbox_enabled
        self.allowed_imports = allowed_imports or ["os", "datetime", "json", "math", "numpy", "pandas"]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create memory directory
        os.makedirs(memory_path, exist_ok=True)


class Agent:
    """
    High-level agent class that integrates all Saplings components.

    The Agent class serves as the main entry point for using Saplings,
    providing a clean, intuitive API while leveraging the full power
    of the underlying components.

    Key features:
    - Structural memory with vector and graph stores
    - Cascaded, entropy-aware retrieval
    - Guard-railed generation with planning and execution
    - Judge and validator loop for self-improvement
    - Graph-Aligned Sparse Attention (GASA) for efficient processing
    - Comprehensive monitoring and tracing

    This class follows the philosophy of being lightweight yet powerful,
    with a focus on extensibility and adaptability.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with the provided configuration.

        Args:
            config: Agent configuration
        """
        self.config = config

        # Initialize components
        self._init_memory()
        self._init_monitoring()
        self._init_model()
        self._init_retrieval()
        self._init_validator()
        self._init_judge()
        self._init_executor()
        self._init_planner()
        self._init_self_healing()
        self._init_tool_factory()
        self._init_orchestration()

        logger.info(f"Agent initialized with model: {config.model_uri}")
        logger.info(f"GASA enabled: {config.enable_gasa}")
        logger.info(f"Monitoring enabled: {config.enable_monitoring}")
        logger.info(f"Self-healing enabled: {config.enable_self_healing}")
        logger.info(f"Tool factory enabled: {config.enable_tool_factory}")

    def _init_memory(self):
        """Initialize memory components."""
        # Create memory config
        memory_config = MemoryConfig(
            store_path=self.config.memory_path,
        )

        # Create memory store
        self.memory_store = MemoryStore(config=memory_config)

        # Create dependency graph
        self.graph = DependencyGraph()

        # Create vector store
        self.vector_store = get_vector_store(config=memory_config)

        # Create indexer
        self.indexer = get_indexer(config=memory_config)

        logger.info("Memory components initialized")

    def _init_monitoring(self):
        """Initialize monitoring components."""
        if self.config.enable_monitoring:
            # Create monitoring config
            self.monitoring_config = MonitoringConfig(
                visualization_output_dir=os.path.join(self.config.output_dir, "visualizations"),
            )

            # Create trace manager
            self.trace_manager = TraceManager(config=self.monitoring_config)

            # Create blame graph
            self.blame_graph = BlameGraph(
                trace_manager=self.trace_manager,
                config=self.monitoring_config,
            )

            # Create trace viewer
            self.trace_viewer = TraceViewer(
                trace_manager=self.trace_manager,
                config=self.monitoring_config,
            )

            logger.info("Monitoring components initialized")
        else:
            self.trace_manager = None
            self.blame_graph = None
            self.trace_viewer = None
            logger.info("Monitoring disabled")

    def _init_model(self):
        """Initialize the model."""
        # Use the new approach if provider and model_name are provided
        if self.config.provider is not None and self.config.model_name is not None:
            self.model = LLM.create(
                provider=self.config.provider,
                model=self.config.model_name,
                **self.config.model_parameters
            )
            logger.info(f"Model initialized: {self.config.provider}/{self.config.model_name}")
        else:
            # Fall back to the URI approach
            self.model = LLM.from_uri(self.config.model_uri)
            logger.info(f"Model initialized: {self.config.model_uri}")

    def _init_retrieval(self):
        """Initialize retrieval components."""
        # Create retrieval config
        retrieval_config = RetrievalConfig(
            entropy_threshold=self.config.retrieval_entropy_threshold,
            max_documents=self.config.retrieval_max_documents,
        )

        # Create TF-IDF retriever
        self.tfidf_retriever = TFIDFRetriever(
            memory_store=self.memory_store,
            config=retrieval_config,
        )

        # Create embedding retriever
        self.embedding_retriever = EmbeddingRetriever(
            memory_store=self.memory_store,
            config=retrieval_config,
        )

        # Create graph expander
        self.graph_expander = GraphExpander(
            memory_store=self.memory_store,
            config=retrieval_config,
        )

        # Create entropy calculator
        self.entropy_calculator = EntropyCalculator(
            config=retrieval_config,
        )

        # Create cascade retriever
        self.retriever = CascadeRetriever(
            memory_store=self.memory_store,
            config=retrieval_config,
        )

        logger.info("Retrieval components initialized")

    def _init_executor(self):
        """Initialize the executor."""
        # Create GASA config if enabled
        gasa_config = None
        if self.config.enable_gasa:
            gasa_config = GASAConfig(
                max_hops=self.config.gasa_max_hops,
                mask_strategy="binary",
                cache_masks=True,
            )

        # Create executor config
        executor_config = ExecutorConfig(
            enable_gasa=self.config.enable_gasa,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            verification_strategy=self.config.executor_verification_strategy,
        )

        # Create executor
        self.executor = Executor(
            model=self.model,
            config=executor_config,
            gasa_config=gasa_config,
            dependency_graph=self.graph if self.config.enable_gasa else None,
        )

        logger.info("Executor initialized")

    def _init_planner(self):
        """Initialize the planner."""
        # Create planner config
        planner_config = PlannerConfig(
            budget_strategy=self.config.planner_budget_strategy,
        )

        # Create planner
        self.planner = SequentialPlanner(
            model=self.model,
            config=planner_config,
        )

        logger.info("Planner initialized")

    def _init_validator(self):
        """Initialize the validator."""
        # Create validator
        self.validator = None  # Will be initialized later with executor

        # Get validator registry
        self.validator_registry = get_validator_registry()

        logger.info("Validator initialized")

    def _init_judge(self):
        """Initialize the judge."""
        # Create judge (will be initialized with executor later)
        self.judge = None

        logger.info("Judge initialized")

    def _init_self_healing(self):
        """Initialize self-healing components."""
        if self.config.enable_self_healing:
            # Create patch generator
            self.patch_generator = PatchGenerator(
                max_retries=3,
            )

            # Create success pair collector
            self.success_pair_collector = SuccessPairCollector(
                output_dir=os.path.join(self.config.output_dir, "success_pairs"),
            )

            # Create adapter manager
            self.adapter_manager = AdapterManager(
                model=self.model,
                adapter_dir=os.path.join(self.config.output_dir, "adapters"),
            )

            logger.info("Self-healing components initialized")
        else:
            self.patch_generator = None
            self.success_pair_collector = None
            self.adapter_manager = None
            logger.info("Self-healing disabled")

    def _init_tool_factory(self):
        """Initialize the tool factory."""
        if self.config.enable_tool_factory:
            # Create tool factory config
            tool_factory_config = ToolFactoryConfig(
                sandbox_enabled=self.config.tool_factory_sandbox_enabled,
                allowed_imports=self.config.allowed_imports,
            )

            # Create tool factory
            self.tool_factory = ToolFactory(
                executor=self.executor,
                config=tool_factory_config,
            )

            logger.info("Tool factory initialized")
        else:
            self.tool_factory = None
            logger.info("Tool factory disabled")

    def _init_orchestration(self):
        """Initialize orchestration components."""
        # Import here to avoid circular imports
        from saplings.orchestration.config import GraphRunnerConfig
        from saplings.orchestration.graph_runner import GraphRunner

        # Create graph runner config
        graph_runner_config = GraphRunnerConfig()

        # Create graph runner
        self.graph_runner = GraphRunner(
            model=self.model,
            config=graph_runner_config,
        )

        logger.info("Orchestration components initialized")

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Add a document to the agent's memory.

        Args:
            content: Document content
            metadata: Document metadata (optional)

        Returns:
            Document: The added document
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="add_document",
                trace_id=trace_id,
                attributes={"component": "agent"},
            )

        try:
            # Create default metadata if not provided
            if metadata is None:
                metadata = {}

            # Create document metadata
            metadata_copy = metadata.copy()
            # Use defaults if not provided
            if "source" not in metadata_copy:
                metadata_copy["source"] = "unknown"
            if "document_id" not in metadata_copy:
                metadata_copy["document_id"] = f"doc_{datetime.now().timestamp()}"

            doc_metadata = DocumentMetadata(**metadata_copy)

            # Create document
            document = Document(
                content=content,
                metadata=doc_metadata,
            )

            # Add to memory store
            self.memory_store.add_document(document)

            # Index the document
            self.indexer.index_document(document)

            logger.info(f"Added document: {document.metadata.document_id}")

            # End span
            if self.trace_manager and span:
                self.trace_manager.end_span(span.span_id)

            return document

        except Exception as e:
            logger.error(f"Error adding document: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def add_documents_from_directory(self, directory: str, extension: str = ".txt") -> List[Document]:
        """
        Add documents from a directory.

        Args:
            directory: Directory containing documents
            extension: File extension to filter by

        Returns:
            List[Document]: Added documents
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="add_documents_from_directory",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "directory": directory,
                    "extension": extension,
                },
            )

        try:
            documents = []

            # Check if directory exists
            if not os.path.isdir(directory):
                logger.error(f"Directory not found: {directory}")

                # End span with error
                if self.trace_manager and span:
                    span.set_status("ERROR")
                    span.add_event(
                        name="directory_not_found",
                        attributes={"directory": directory},
                    )
                    self.trace_manager.end_span(span.span_id)

                return documents

            # Process each file
            for filename in os.listdir(directory):
                if filename.endswith(extension):
                    file_path = os.path.join(directory, filename)

                    # Read file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create metadata
                    metadata = {
                        "source": file_path,
                        "document_id": filename,
                        "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        "file_size": os.path.getsize(file_path),
                    }

                    # Add document
                    document = await self.add_document(content, metadata)
                    documents.append(document)

            logger.info(f"Added {len(documents)} documents from {directory}")

            # End span
            if self.trace_manager and span:
                span.add_event(
                    name="documents_added",
                    attributes={"count": len(documents)},
                )
                self.trace_manager.end_span(span.span_id)

            return documents

        except Exception as e:
            logger.error(f"Error adding documents from directory: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def retrieve(self, query: str, limit: int = None) -> List[Document]:
        """
        Retrieve documents based on a query.

        Args:
            query: Query to retrieve documents
            limit: Maximum number of documents to retrieve (optional)

        Returns:
            List[Document]: Retrieved documents
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="retrieve",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "query": query,
                    "limit": limit or self.config.retrieval_max_documents,
                },
            )

        try:
            # Use cascade retriever
            documents = await self.retriever.retrieve(
                query=query,
                limit=limit or self.config.retrieval_max_documents,
                trace_id=trace_id,
            )

            logger.info(f"Retrieved {len(documents)} documents for query: {query}")

            # End span
            if self.trace_manager and span:
                span.add_event(
                    name="documents_retrieved",
                    attributes={"count": len(documents)},
                )
                self.trace_manager.end_span(span.span_id)

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def plan(self, task: str, context: Optional[List[Document]] = None) -> List[PlanStep]:
        """
        Create a plan for a task.

        Args:
            task: Task description
            context: Context documents (optional)

        Returns:
            List[PlanStep]: Plan steps
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="plan",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "task": task,
                },
            )

        try:
            # Create plan
            plan = await self.planner.create_plan(
                task=task,
                context=context,
                trace_id=trace_id,
            )

            logger.info(f"Created plan with {len(plan)} steps for task: {task}")

            # End span
            if self.trace_manager and span:
                span.add_event(
                    name="plan_created",
                    attributes={"step_count": len(plan)},
                )
                self.trace_manager.end_span(span.span_id)

            return plan

        except Exception as e:
            logger.error(f"Error creating plan: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def execute(self, prompt: str, context: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        Execute a prompt with the agent.

        Args:
            prompt: Prompt to execute
            context: Context documents (optional)

        Returns:
            Dict[str, Any]: Execution result
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="execute",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "prompt": prompt,
                },
            )

        try:
            # Retrieve context if not provided
            if context is None and prompt:
                context_span = None
                if self.trace_manager:
                    context_span = self.trace_manager.start_span(
                        name="retrieve_context",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "retriever"},
                    )

                context = await self.retrieve(prompt)

                if self.trace_manager and context_span:
                    self.trace_manager.end_span(context_span.span_id)

            # Execute with executor
            execution_span = None
            if self.trace_manager:
                execution_span = self.trace_manager.start_span(
                    name="execute_prompt",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "executor"},
                )

            result = await self.executor.execute(
                prompt=prompt,
                documents=context,
                trace_id=trace_id,
            )

            if self.trace_manager and execution_span:
                self.trace_manager.end_span(execution_span.span_id)

            # Validate result
            validation_span = None
            if self.trace_manager:
                validation_span = self.trace_manager.start_span(
                    name="validate_result",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "validator"},
                )

            validation_result = await self.validator.validate(
                input_data={"prompt": prompt, "context": [doc.content for doc in context] if context else []},
                output_data=result.text,
                validation_type="execution",
            )

            if self.trace_manager and validation_span:
                validation_span.add_event(
                    name="validation_complete",
                    attributes={"is_valid": validation_result.is_valid},
                )
                self.trace_manager.end_span(validation_span.span_id)

            # Collect success pair if valid
            if self.config.enable_self_healing and validation_result.is_valid and self.success_pair_collector:
                await self.success_pair_collector.collect(
                    input_text=prompt,
                    output_text=result.text,
                    context=[doc.content for doc in context] if context else [],
                    metadata={
                        "validation_score": validation_result.score,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            # End span
            if self.trace_manager and span:
                self.trace_manager.end_span(span.span_id)

            # Process trace for blame graph
            if self.trace_manager and self.blame_graph:
                trace = self.trace_manager.get_trace(trace_id)
                if trace:
                    self.blame_graph.process_trace(trace)

            return {
                "text": result.text,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "score": validation_result.score,
                    "feedback": validation_result.feedback,
                },
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"Error executing prompt: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def execute_plan(self, plan: List[PlanStep], context: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: Plan steps to execute
            context: Context documents (optional)

        Returns:
            Dict[str, Any]: Execution results
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="execute_plan",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "step_count": len(plan),
                },
            )

        try:
            results = []

            # Execute each step
            for i, step in enumerate(plan):
                step_span = None
                if self.trace_manager:
                    step_span = self.trace_manager.start_span(
                        name=f"execute_step_{i+1}",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={
                            "component": "agent",
                            "step_number": i + 1,
                            "step_description": step.description,
                        },
                    )

                # Execute step
                step_result = await self.executor.execute(
                    prompt=step.description,
                    documents=context,
                    trace_id=trace_id,
                )

                # Update step status
                step.status = "completed"
                step.result = step_result.text

                # Add to results
                results.append({
                    "step": step,
                    "result": step_result.text,
                })

                # End step span
                if self.trace_manager and step_span:
                    step_span.add_event(
                        name="step_completed",
                        attributes={"step_number": i + 1},
                    )
                    self.trace_manager.end_span(step_span.span_id)

            # End span
            if self.trace_manager and span:
                span.add_event(
                    name="plan_executed",
                    attributes={"completed_steps": len(results)},
                )
                self.trace_manager.end_span(span.span_id)

            # Process trace for blame graph
            if self.trace_manager and self.blame_graph:
                trace = self.trace_manager.get_trace(trace_id)
                if trace:
                    self.blame_graph.process_trace(trace)

            return {
                "results": results,
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"Error executing plan: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """
        Create a dynamic tool.

        Args:
            name: Tool name
            description: Tool description
            code: Tool code

        Returns:
            Any: Created tool
        """
        if not self.config.enable_tool_factory or not self.tool_factory:
            raise ValueError("Tool factory is disabled")

        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="create_tool",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "tool_name": name,
                },
            )

        try:
            # Create tool
            tool = await self.tool_factory.create_tool(
                name=name,
                description=description,
                code=code,
            )

            logger.info(f"Created tool: {name}")

            # End span
            if self.trace_manager and span:
                self.trace_manager.end_span(span.span_id)

            return tool

        except Exception as e:
            logger.error(f"Error creating tool: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def judge_output(self, input_data: Dict[str, Any], output_data: Any, judgment_type: str = "general") -> Dict[str, Any]:
        """
        Judge an output using the JudgeAgent.

        Args:
            input_data: Input data
            output_data: Output data to judge
            judgment_type: Type of judgment

        Returns:
            Dict[str, Any]: Judgment result
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="judge_output",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "judgment_type": judgment_type,
                },
            )

        try:
            # Judge output
            judgment = await self.judge.judge(
                input_data=input_data,
                output_data=output_data,
                judgment_type=judgment_type,
            )

            logger.info(f"Judged output with score: {judgment.score}")

            # End span
            if self.trace_manager and span:
                span.add_event(
                    name="judgment_complete",
                    attributes={"score": judgment.score},
                )
                self.trace_manager.end_span(span.span_id)

            return {
                "score": judgment.score,
                "feedback": judgment.feedback,
                "strengths": judgment.strengths,
                "weaknesses": judgment.weaknesses,
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"Error judging output: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def self_improve(self) -> Dict[str, Any]:
        """
        Improve the agent based on past performance.

        Returns:
            Dict[str, Any]: Improvement results
        """
        if not self.config.enable_self_healing:
            raise ValueError("Self-healing is disabled")

        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="self_improve",
                trace_id=trace_id,
                attributes={"component": "agent"},
            )

        try:
            # Get performance data from blame graph
            performance_data = {}
            if self.trace_manager and self.blame_graph:
                perf_span = None
                if self.trace_manager:
                    perf_span = self.trace_manager.start_span(
                        name="analyze_performance",
                        trace_id=trace_id,
                        parent_id=span.span_id if span else None,
                        attributes={"component": "blame_graph"},
                    )

                # Identify bottlenecks
                bottlenecks = self.blame_graph.identify_bottlenecks(
                    threshold_ms=100.0,
                    min_call_count=1,
                )

                # Identify error sources
                error_sources = self.blame_graph.identify_error_sources(
                    min_error_rate=0.1,
                    min_call_count=1,
                )

                if self.trace_manager and perf_span:
                    perf_span.add_event(
                        name="performance_analysis_complete",
                        attributes={
                            "bottleneck_count": len(bottlenecks),
                            "error_source_count": len(error_sources),
                        },
                    )
                    self.trace_manager.end_span(perf_span.span_id)

                performance_data = {
                    "bottlenecks": bottlenecks,
                    "error_sources": error_sources,
                }

            # Get validation data
            validation_data = self.validator.get_validation_history()

            # Get success pairs
            success_pairs = []
            if self.success_pair_collector:
                success_pairs = await self.success_pair_collector.get_all_pairs()

            # Create improvement prompt
            prompt = f"""
            Analyze the agent's performance and suggest improvements.

            Performance Data:
            {performance_data}

            Validation History:
            {validation_data}

            Success Pairs Count: {len(success_pairs)}

            Based on this data, suggest specific improvements to the agent's:
            1. Prompting strategies
            2. Document analysis approach
            3. Overall workflow

            Format your response as JSON with the following structure:
            {{
                "identified_issues": [list of issues identified],
                "improvement_suggestions": [list of specific improvement suggestions],
                "prompt_templates": {{
                    "analysis_prompt": "improved analysis prompt template",
                    "execution_prompt": "improved execution prompt template"
                }},
                "implementation_plan": [list of steps to implement improvements]
            }}
            """

            # Execute improvement analysis
            improvement_span = None
            if self.trace_manager:
                improvement_span = self.trace_manager.start_span(
                    name="generate_improvements",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "executor"},
                )

            result = await self.executor.execute(
                prompt=prompt,
                trace_id=trace_id,
            )

            if self.trace_manager and improvement_span:
                self.trace_manager.end_span(improvement_span.span_id)

            # Parse JSON response
            try:
                import json
                improvements = json.loads(result.text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                improvements = {"raw_text": result.text}

            # Judge improvements
            judge_span = None
            if self.trace_manager:
                judge_span = self.trace_manager.start_span(
                    name="judge_improvements",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "judge"},
                )

            judgment = await self.judge.judge(
                input_data={
                    "performance_data": performance_data,
                    "validation_data": validation_data,
                },
                output_data=improvements,
                judgment_type="agent_improvement",
            )

            if self.trace_manager and judge_span:
                judge_span.add_event(
                    name="judgment_complete",
                    attributes={"score": judgment.score},
                )
                self.trace_manager.end_span(judge_span.span_id)

            # Add judgment to improvements
            improvements["judgment"] = {
                "score": judgment.score,
                "feedback": judgment.feedback,
                "strengths": judgment.strengths,
                "weaknesses": judgment.weaknesses,
            }

            # End span
            if self.trace_manager and span:
                self.trace_manager.end_span(span.span_id)

            # Save improvements to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config.output_dir, f"improvements_{timestamp}.json")

            with open(output_path, "w") as f:
                json.dump(improvements, f, indent=2)

            logger.info(f"Improvement analysis completed and saved to {output_path}")

            return {
                "improvements": improvements,
                "trace_id": trace_id,
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"Error during agent improvement: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise

    async def run(self, task: str) -> Dict[str, Any]:
        """
        Run the agent on a task, handling the full lifecycle.

        This method orchestrates the entire agent workflow:
        1. Retrieve relevant context
        2. Create a plan
        3. Execute the plan
        4. Validate and judge the results
        5. Collect success pairs for self-improvement

        Args:
            task: Task description

        Returns:
            Dict[str, Any]: Results of the task execution
        """
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id

            span = self.trace_manager.start_span(
                name="run",
                trace_id=trace_id,
                attributes={
                    "component": "agent",
                    "task": task,
                },
            )

        try:
            # Step 1: Retrieve context
            context_span = None
            if self.trace_manager:
                context_span = self.trace_manager.start_span(
                    name="retrieve_context",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "retriever"},
                )

            context = await self.retrieve(task)

            if self.trace_manager and context_span:
                context_span.add_event(
                    name="context_retrieved",
                    attributes={"document_count": len(context)},
                )
                self.trace_manager.end_span(context_span.span_id)

            # Step 2: Create plan
            plan_span = None
            if self.trace_manager:
                plan_span = self.trace_manager.start_span(
                    name="create_plan",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "planner"},
                )

            plan = await self.plan(task, context)

            if self.trace_manager and plan_span:
                plan_span.add_event(
                    name="plan_created",
                    attributes={"step_count": len(plan)},
                )
                self.trace_manager.end_span(plan_span.span_id)

            # Step 3: Execute plan
            execution_span = None
            if self.trace_manager:
                execution_span = self.trace_manager.start_span(
                    name="execute_plan",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "executor"},
                )

            execution_results = await self.execute_plan(plan, context)

            if self.trace_manager and execution_span:
                execution_span.add_event(
                    name="plan_executed",
                    attributes={"completed_steps": len(execution_results["results"])},
                )
                self.trace_manager.end_span(execution_span.span_id)

            # Step 4: Judge results
            judge_span = None
            if self.trace_manager:
                judge_span = self.trace_manager.start_span(
                    name="judge_results",
                    trace_id=trace_id,
                    parent_id=span.span_id if span else None,
                    attributes={"component": "judge"},
                )

            # Prepare final result
            final_result = "\n\n".join([r["result"] for r in execution_results["results"]])

            judgment = await self.judge.judge(
                input_data={"task": task, "context": [doc.content for doc in context]},
                output_data=final_result,
                judgment_type="task_execution",
            )

            if self.trace_manager and judge_span:
                judge_span.add_event(
                    name="judgment_complete",
                    attributes={"score": judgment.score},
                )
                self.trace_manager.end_span(judge_span.span_id)

            # Step 5: Collect success pair if valid
            if self.config.enable_self_healing and judgment.score >= 0.7 and self.success_pair_collector:
                await self.success_pair_collector.collect(
                    input_text=task,
                    output_text=final_result,
                    context=[doc.content for doc in context],
                    metadata={
                        "judgment_score": judgment.score,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            # End span
            if self.trace_manager and span:
                self.trace_manager.end_span(span.span_id)

            # Process trace for blame graph
            if self.trace_manager and self.blame_graph:
                trace = self.trace_manager.get_trace(trace_id)
                if trace:
                    self.blame_graph.process_trace(trace)

            # Save results to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config.output_dir, f"task_results_{timestamp}.json")

            with open(output_path, "w") as f:
                import json
                json.dump({
                    "task": task,
                    "plan": [step.__dict__ for step in plan],
                    "results": execution_results["results"],
                    "judgment": {
                        "score": judgment.score,
                        "feedback": judgment.feedback,
                        "strengths": judgment.strengths,
                        "weaknesses": judgment.weaknesses,
                    },
                }, f, indent=2, default=str)

            logger.info(f"Task execution completed and saved to {output_path}")

            return {
                "task": task,
                "context": context,
                "plan": plan,
                "results": execution_results["results"],
                "final_result": final_result,
                "judgment": {
                    "score": judgment.score,
                    "feedback": judgment.feedback,
                    "strengths": judgment.strengths,
                    "weaknesses": judgment.weaknesses,
                },
                "trace_id": trace_id,
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"Error running task: {e}")

            # End span with error
            if self.trace_manager and span:
                span.set_status("ERROR")
                self.trace_manager.end_span(span.span_id)

            raise