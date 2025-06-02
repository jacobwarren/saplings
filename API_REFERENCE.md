# Saplings API Reference

This document provides comprehensive documentation for all public APIs in the Saplings framework.

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Tools](#tools)
- [Services](#services)
- [Model Adapters](#model-adapters)
- [Memory and Retrieval](#memory-and-retrieval)
- [Planning and Execution](#planning-and-execution)
- [Monitoring and Validation](#monitoring-and-validation)
- [Utilities](#utilities)

## Core Classes

### Agent

**Stability**: `stable`

The main agent class for the Saplings framework. Provides a high-level interface for running tasks, managing memory, and interacting with tools.

```python
class Agent:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        config: AgentConfig = None,
        **kwargs
    ):
        """
        Initialize an Agent instance.
        
        Args:
            provider: Model provider ("openai", "anthropic", "vllm", "huggingface")
            model_name: Name of the model to use
            config: Pre-configured AgentConfig instance
            **kwargs: Additional configuration parameters
        """
    
    async def run(
        self,
        task: str,
        input_modalities: list[str] | None = None,
        output_modalities: list[str] | None = None,
        use_tools: bool = True,
        skip_retrieval: bool = False,
        skip_planning: bool = False,
        skip_validation: bool = False,
        context: list[Any] | None = None,
        plan: list[Any] | None = None,
        timeout: float | None = None,
        save_results: bool = True,
    ) -> dict[str, Any] | str:
        """
        Execute a task asynchronously.
        
        Args:
            task: The task description to execute
            input_modalities: List of input modalities to use
            output_modalities: List of output modalities to generate
            use_tools: Whether to use available tools
            skip_retrieval: Skip the retrieval phase
            skip_planning: Skip the planning phase
            skip_validation: Skip output validation
            context: Additional context for the task
            plan: Pre-defined execution plan
            timeout: Maximum execution time in seconds
            save_results: Whether to save execution results
            
        Returns:
            Task execution result (string or structured dict)
        """
    
    def run_sync(self, task: str, **kwargs) -> dict[str, Any] | str:
        """
        Synchronous wrapper for the run() method.
        
        Args:
            task: The task description to execute
            **kwargs: Same arguments as run() method
            
        Returns:
            Task execution result
            
        Raises:
            RuntimeError: If called from within an async context
        """
    
    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Add a document to the agent's memory.
        
        Args:
            content: Document content
            metadata: Optional metadata dictionary
        """
    
    def add_documents_from_directory(
        self,
        directory: str,
        extension: str = ".md"
    ) -> None:
        """
        Add all documents from a directory.
        
        Args:
            directory: Path to the directory
            extension: File extension to filter by
        """
    
    async def add_document_from_url(self, url: str) -> None:
        """
        Add a document from a URL.
        
        Args:
            url: URL to fetch content from
        """
    
    def register_tool(self, tool: Any) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: Tool instance, function, or callable
        """
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        fast_mode: bool = False
    ) -> list[dict]:
        """
        Retrieve documents based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of documents to return
            fast_mode: Use faster but less accurate retrieval
            
        Returns:
            List of relevant documents
        """
    
    async def plan(
        self,
        task: str,
        context: list[Any] | None = None
    ) -> list[Any]:
        """
        Create an execution plan for a task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Execution plan steps
        """
    
    async def execute_plan(
        self,
        plan: list[Any],
        context: list[Any] | None = None,
        use_tools: bool = True
    ) -> dict[str, Any]:
        """
        Execute a predefined plan.
        
        Args:
            plan: Execution plan steps
            context: Additional context
            use_tools: Whether to use tools during execution
            
        Returns:
            Execution results
        """
```

### AgentConfig

**Stability**: `stable`

Configuration class for customizing agent behavior.

```python
class AgentConfig:
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str | None = None,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
        enable_self_healing: bool = True,
        self_healing_max_retries: int = 3,
        enable_tool_factory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        gasa_max_hops: int = 2,
        gasa_strategy: str = "binary",
        gasa_fallback: str = "block_diagonal",
        gasa_shadow_model: bool = False,
        gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B",
        gasa_prompt_composer: bool = False,
        retrieval_entropy_threshold: float = 0.1,
        retrieval_max_documents: int = 10,
        planner_budget_strategy: str = "proportional",
        planner_total_budget: float = 1.0,
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_validation_type: str = "judge",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: list[str] | None = None,
        tools: list[Any] | None = None,
        supported_modalities: list[str] | None = None,
        **model_parameters
    ):
        """
        Initialize agent configuration.
        
        Args:
            provider: Model provider name
            model_name: Model identifier
            api_key: API key for cloud providers
            memory_path: Path for persistent memory storage
            output_dir: Directory for saving outputs
            enable_gasa: Enable Graph-Aligned Sparse Attention
            enable_monitoring: Enable execution monitoring
            enable_self_healing: Enable automatic error recovery
            self_healing_max_retries: Max retry attempts for self-healing
            enable_tool_factory: Enable dynamic tool creation
            max_tokens: Maximum tokens for model responses
            temperature: Model generation temperature
            gasa_max_hops: Maximum hops for GASA attention
            gasa_strategy: GASA strategy ("binary", "soft", "learned")
            gasa_fallback: Fallback for unsupported models
            gasa_shadow_model: Use shadow model for tokenization
            gasa_shadow_model_name: Shadow model identifier
            gasa_prompt_composer: Enable graph-aware prompt composition
            retrieval_entropy_threshold: Entropy threshold for retrieval
            retrieval_max_documents: Max documents to retrieve
            planner_budget_strategy: Budget allocation strategy
            planner_total_budget: Total budget in USD
            planner_allow_budget_overflow: Allow budget overflow
            planner_budget_overflow_margin: Overflow margin fraction
            executor_validation_type: Validation strategy
            tool_factory_sandbox_enabled: Enable tool sandboxing
            allowed_imports: Allowed imports for dynamic tools
            tools: Initial tools to register
            supported_modalities: Supported input/output modalities
            **model_parameters: Additional model-specific parameters
        """
    
    @classmethod
    def minimal(cls, provider: str, model_name: str, **kwargs) -> "AgentConfig":
        """Create minimal configuration with essential features only."""
    
    @classmethod
    def standard(cls, provider: str, model_name: str, **kwargs) -> "AgentConfig":
        """Create standard configuration with balanced features."""
    
    @classmethod
    def full_featured(cls, provider: str, model_name: str, **kwargs) -> "AgentConfig":
        """Create full-featured configuration with all features enabled."""
    
    @classmethod
    def for_openai(cls, model_name: str, **kwargs) -> "AgentConfig":
        """Create configuration optimized for OpenAI models."""
    
    @classmethod
    def for_anthropic(cls, model_name: str, **kwargs) -> "AgentConfig":
        """Create configuration optimized for Anthropic models."""
    
    @classmethod
    def for_vllm(cls, model_name: str, **kwargs) -> "AgentConfig":
        """Create configuration optimized for vLLM models."""
    
    def validate(self) -> "ValidationResult":
        """Validate the configuration and return validation results."""
    
    def check_dependencies(self) -> "DependencyResult":
        """Check if all required dependencies are available."""
    
    def suggest_fixes(self) -> list[str]:
        """Get suggestions for fixing configuration issues."""
    
    def explain(self) -> str:
        """Get a human-readable explanation of the configuration."""
    
    def compare(self, other: "AgentConfig") -> str:
        """Compare this configuration with another."""
```

### AgentBuilder

**Stability**: `stable`

Builder pattern for creating Agent instances with fluent configuration.

```python
class AgentBuilder:
    def __init__(self):
        """Initialize a new AgentBuilder."""
    
    def with_provider(self, provider: str) -> "AgentBuilder":
        """Set the model provider."""
    
    def with_model_name(self, model_name: str) -> "AgentBuilder":
        """Set the model name."""
    
    def with_api_key(self, api_key: str) -> "AgentBuilder":
        """Set the API key."""
    
    def with_memory_path(self, path: str) -> "AgentBuilder":
        """Set the memory storage path."""
    
    def with_output_dir(self, path: str) -> "AgentBuilder":
        """Set the output directory."""
    
    def with_gasa_enabled(self, enabled: bool) -> "AgentBuilder":
        """Enable or disable GASA."""
    
    def with_monitoring_enabled(self, enabled: bool) -> "AgentBuilder":
        """Enable or disable monitoring."""
    
    def with_self_healing_enabled(self, enabled: bool) -> "AgentBuilder":
        """Enable or disable self-healing."""
    
    def with_tools(self, tools: list[Any]) -> "AgentBuilder":
        """Set the tools to use."""
    
    def with_model_parameters(self, parameters: dict[str, Any]) -> "AgentBuilder":
        """Set additional model parameters."""
    
    def build(self, use_container: bool = True) -> Agent:
        """
        Build the Agent instance.
        
        Args:
            use_container: Whether to use dependency injection container
            
        Returns:
            Configured Agent instance
        """
    
    @classmethod
    def minimal(cls, provider: str, model_name: str) -> "AgentBuilder":
        """Create builder with minimal configuration."""
    
    @classmethod
    def standard(cls, provider: str, model_name: str) -> "AgentBuilder":
        """Create builder with standard configuration."""
    
    @classmethod
    def full_featured(cls, provider: str, model_name: str) -> "AgentBuilder":
        """Create builder with full-featured configuration."""
    
    @classmethod
    def for_openai(cls, model_name: str) -> "AgentBuilder":
        """Create builder optimized for OpenAI."""
    
    @classmethod
    def for_anthropic(cls, model_name: str) -> "AgentBuilder":
        """Create builder optimized for Anthropic."""
    
    @classmethod
    def for_vllm(cls, model_name: str) -> "AgentBuilder":
        """Create builder optimized for vLLM."""
```

### AgentFacade

**Stability**: `beta`

Service-oriented facade for Agent implementation with direct service access.

```python
class AgentFacade:
    def __init__(self, config: AgentConfig, **services):
        """
        Initialize AgentFacade with configuration and services.
        
        Args:
            config: Agent configuration
            **services: Service instances for dependency injection
        """
    
    async def run(self, task: str, **kwargs) -> dict[str, Any] | str:
        """Run a task using the full agent lifecycle."""
    
    def add_document(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document to memory."""
    
    def add_documents_from_directory(self, directory: str, extension: str = ".md") -> None:
        """Add documents from a directory."""
    
    async def retrieve(self, query: str, limit: int = 10, fast_mode: bool = False) -> list[dict]:
        """Retrieve documents based on query."""
    
    async def plan(self, task: str, context: list[Any] | None = None) -> list[Any]:
        """Create execution plan."""
    
    async def execute(self, prompt: str, context: list[Any] | None = None, use_tools: bool = True) -> Any:
        """Execute a prompt."""
    
    async def execute_plan(self, plan: list[Any], context: list[Any] | None = None, use_tools: bool = True) -> dict[str, Any]:
        """Execute a predefined plan."""
    
    def register_tool(self, tool: Any) -> None:
        """Register a tool."""
    
    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """Create a dynamic tool."""
    
    async def judge_output(self, input_data: Any, output_data: Any, judgment_type: str) -> dict[str, Any]:
        """Judge/validate output quality."""
    
    async def self_improve(self) -> dict[str, Any]:
        """Trigger self-improvement based on past performance."""
```

## Configuration

### Configuration Presets

The library provides several configuration presets for common use cases:

```python
# Minimal configuration - essential features only
config = AgentConfig.minimal("openai", "gpt-4o")

# Standard configuration - balanced features
config = AgentConfig.standard("openai", "gpt-4o")

# Full-featured configuration - all features enabled
config = AgentConfig.full_featured("openai", "gpt-4o")

# Provider-specific optimizations
config = AgentConfig.for_openai("gpt-4o")
config = AgentConfig.for_anthropic("claude-3-opus")
config = AgentConfig.for_vllm("Qwen/Qwen3-7B-Instruct")
```

### GASA Configuration

Graph-Aligned Sparse Attention settings:

```python
config = AgentConfig(
    # ... other settings ...
    enable_gasa=True,
    gasa_max_hops=3,                    # Maximum attention hops
    gasa_strategy="binary",             # "binary", "soft", "learned"
    gasa_fallback="prompt_composer",    # "block_diagonal", "prompt_composer"
    gasa_shadow_model=True,             # Use shadow model for tokenization
    gasa_shadow_model_name="Qwen/Qwen3-0.6B",
    gasa_prompt_composer=True           # Enable graph-aware prompting
)
```

### Planning Configuration

Task planning and budget management:

```python
config = AgentConfig(
    # ... other settings ...
    planner_budget_strategy="dynamic",      # "fixed", "proportional", "dynamic"
    planner_total_budget=5.0,              # Total budget in USD
    planner_allow_budget_overflow=True,    # Allow exceeding budget
    planner_budget_overflow_margin=0.1     # 10% overflow margin
)
```

### Retrieval Configuration

Document retrieval settings:

```python
config = AgentConfig(
    # ... other settings ...
    retrieval_entropy_threshold=0.1,    # Entropy threshold for stopping
    retrieval_max_documents=20,         # Maximum documents to retrieve
)
```

## Tools

### Built-in Tools

```python
from saplings.api.tools import (
    PythonInterpreterTool,      # Execute Python code safely
    DuckDuckGoSearchTool,       # Web search via DuckDuckGo
    WikipediaSearchTool,        # Wikipedia search and retrieval
    GoogleSearchTool,           # Google search (requires API key)
    VisitWebpageTool,          # Extract content from web pages
    UserInputTool,             # Interactive user input
    FinalAnswerTool,           # Provide final responses
    SpeechToTextTool           # Speech transcription (beta)
)
```

### Tool Management

```python
from saplings.api.tools import (
    register_tool,              # Register tools globally
    get_all_default_tools,      # Get all built-in tools
    get_default_tool,          # Get specific tool by name
    validate_tool,             # Validate tool implementation
    ToolRegistry,              # Tool registry management
    ToolCollection            # Group related tools
)
```

### Custom Tool Creation

```python
from saplings.api.tools import tool

@tool(name="my_tool", description="Custom tool description")
def my_custom_tool(param1: str, param2: int = 10) -> str:
    """
    Custom tool function.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value
    """
    return f"Processed {param1} with {param2}"

# Register the tool
register_tool(my_custom_tool)
```

### Browser Tools

```python
from saplings.api.browser_tools import (
    BrowserManager,            # Manage browser instances
    is_browser_tools_available # Check availability
)

# Browser tools (when browser extras are installed)
browser_tools = [
    "GoToTool",                # Navigate to URL
    "ClickTool",               # Click elements
    "TypeTool",                # Type text
    "ScrollTool",              # Scroll page
    "ScreenshotTool",          # Take screenshots
    "ExtractTextTool",         # Extract page text
    "FindElementTool"          # Find page elements
]
```

## Services

### Service Interfaces

The framework uses a service-oriented architecture with the following interfaces:

```python
from saplings.api.services import (
    MemoryService,             # Document storage and management
    RetrievalService,          # Document retrieval and search
    PlannerService,            # Task planning and decomposition
    ExecutionService,          # Task execution and orchestration
    ValidationService,         # Output validation and quality control
    ToolService,               # Tool management and execution
    MonitoringService,         # Performance monitoring and logging
    ModelService,              # Model adapter management
    ModalityService,           # Multi-modal input/output handling
    SelfHealingService,        # Error recovery and optimization
    OrchestrationService       # Service coordination
)
```

### Memory Service

```python
class MemoryService:
    def add_document(self, content: str, metadata: dict | None = None) -> str:
        """Add document and return document ID."""
    
    def get_document(self, doc_id: str) -> dict | None:
        """Retrieve document by ID."""
    
    def update_document(self, doc_id: str, content: str, metadata: dict | None = None) -> bool:
        """Update existing document."""
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID."""
    
    def search_documents(self, query: str, limit: int = 10) -> list[dict]:
        """Search documents by content."""
    
    def get_all_documents(self) -> list[dict]:
        """Get all stored documents."""
    
    def clear_memory(self) -> None:
        """Clear all stored documents."""
```

### Retrieval Service

```python
class RetrievalService:
    async def retrieve(self, query: str, limit: int = 10, threshold: float = 0.1) -> list[dict]:
        """Retrieve relevant documents for a query."""
    
    async def semantic_search(self, query: str, limit: int = 10) -> list[dict]:
        """Perform semantic search using embeddings."""
    
    async def keyword_search(self, query: str, limit: int = 10) -> list[dict]:
        """Perform keyword-based search."""
    
    async def hybrid_search(self, query: str, limit: int = 10) -> list[dict]:
        """Combine semantic and keyword search."""
```

### Planning Service

```python
class PlannerService:
    async def create_plan(self, goal: str, context: list[Any] | None = None) -> list[PlanStep]:
        """Create execution plan for a goal."""
    
    async def refine_plan(self, plan: list[PlanStep], feedback: str) -> list[PlanStep]:
        """Refine existing plan based on feedback."""
    
    async def estimate_cost(self, plan: list[PlanStep]) -> float:
        """Estimate execution cost for a plan."""
    
    async def validate_plan(self, plan: list[PlanStep]) -> bool:
        """Validate plan feasibility."""
```

## Model Adapters

### Supported Providers

```python
from saplings.api.model_adapters import (
    OpenAIAdapter,             # OpenAI GPT models
    AnthropicAdapter,          # Anthropic Claude models
    VLLMAdapter,               # vLLM self-hosted models
    HuggingFaceAdapter         # HuggingFace transformers
)
```

### Model Configuration

```python
# OpenAI configuration
config = {
    "provider": "openai",
    "model_name": "gpt-4o",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Anthropic configuration
config = {
    "provider": "anthropic",
    "model_name": "claude-3-opus",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
    "top_k": 40
}

# vLLM configuration
config = {
    "provider": "vllm",
    "model_name": "Qwen/Qwen3-7B-Instruct",
    "base_url": "http://localhost:8000",
    "temperature": 0.7,
    "max_tokens": 2048,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8,
    "trust_remote_code": True
}
```

## Memory and Retrieval

### Vector Store Integration

```python
from saplings.api.vector_store import (
    FaissVectorStore,          # FAISS-based vector storage
    ChromaVectorStore,         # Chroma vector database
    PineconeVectorStore,       # Pinecone cloud vector DB
    WeaviateVectorStore        # Weaviate vector database
)

# Configure vector store
config = AgentConfig(
    # ... other settings ...
    vector_store_type="faiss",
    vector_store_config={
        "dimension": 1536,
        "index_type": "IVF",
        "n_lists": 100
    }
)
```

### Document Processing

```python
from saplings.api.document_node import DocumentNode

# Document structure
class DocumentNode:
    def __init__(
        self,
        content: str,
        metadata: dict | None = None,
        doc_id: str | None = None
    ):
        """
        Initialize document node.
        
        Args:
            content: Document content
            metadata: Optional metadata
            doc_id: Unique document identifier
        """
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentNode":
        """Create from dictionary representation."""
    
    def get_chunks(self, chunk_size: int = 1000, overlap: int = 100) -> list["DocumentNode"]:
        """Split document into chunks."""
```

## Planning and Execution

### Plan Steps

```python
from saplings.planner.plan_step import PlanStep, PlanStepStatus, StepType, StepPriority

class PlanStep:
    def __init__(
        self,
        id: str,
        task_description: str,
        step_type: StepType,
        priority: StepPriority,
        estimated_cost: float,
        estimated_tokens: int,
        dependencies: list[str] = None,
        status: PlanStepStatus = PlanStepStatus.PENDING
    ):
        """
        Initialize plan step.
        
        Args:
            id: Unique step identifier
            task_description: Description of the task
            step_type: Type of step (ANALYSIS, GENERATION, etc.)
            priority: Step priority level
            estimated_cost: Estimated execution cost
            estimated_tokens: Estimated token usage
            dependencies: List of dependent step IDs
            status: Current execution status
        """

# Step types
class StepType(Enum):
    ANALYSIS = "analysis"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    TOOL_USE = "tool_use"
    VALIDATION = "validation"

# Step priorities
class StepPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Step statuses
class PlanStepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
```

### Execution Control

```python
from saplings.api.executor import ExecutionService

class ExecutionService:
    async def execute_step(
        self,
        step: PlanStep,
        context: dict | None = None,
        use_tools: bool = True
    ) -> dict[str, Any]:
        """Execute a single plan step."""
    
    async def execute_plan(
        self,
        plan: list[PlanStep],
        context: dict | None = None,
        use_tools: bool = True,
        parallel: bool = False
    ) -> dict[str, Any]:
        """Execute an entire plan."""
    
    async def validate_execution(
        self,
        step: PlanStep,
        result: dict[str, Any]
    ) -> bool:
        """Validate step execution result."""
```

## Monitoring and Validation

### Monitoring Service

```python
from saplings.api.monitoring import MonitoringService, TraceViewer

class MonitoringService:
    def start_trace(self, trace_id: str) -> None:
        """Start a new execution trace."""
    
    def end_trace(self, trace_id: str) -> None:
        """End an execution trace."""
    
    def record_event(self, event_type: str, data: dict) -> None:
        """Record an event in the current trace."""
    
    def record_model_call(self, prompt: str, response: Any, metadata: dict | None = None) -> None:
        """Record a model API call."""
    
    def record_tool_call(self, tool_name: str, args: dict, result: Any) -> None:
        """Record a tool execution."""
    
    def get_traces(self) -> list[dict]:
        """Get all recorded traces."""
    
    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
```

### Validation

```python
from saplings.api.validator import CodeValidator, FactualValidator, OutputValidator

class CodeValidator:
    def validate(self, code: str, language: str = "python") -> dict[str, Any]:
        """Validate code syntax and structure."""
    
    def check_security(self, code: str) -> list[str]:
        """Check for security issues in code."""
    
    def suggest_improvements(self, code: str) -> list[str]:
        """Suggest code improvements."""

class FactualValidator:
    async def validate_facts(self, text: str) -> dict[str, Any]:
        """Validate factual claims in text."""
    
    async def check_sources(self, text: str, sources: list[str]) -> dict[str, Any]:
        """Check if claims are supported by sources."""

class OutputValidator:
    def validate_format(self, output: Any, expected_format: str) -> bool:
        """Validate output format."""
    
    def validate_completeness(self, output: Any, requirements: list[str]) -> dict[str, Any]:
        """Check if output meets requirements."""
```

## Utilities

### Container and Dependency Injection

```python
from saplings.api.di import (
    Container,
    container,
    configure_container,
    reset_container,
    reset_container_config
)

# Configure services
configure_container(config)

# Access container
service = container.resolve("MemoryService")

# Reset configuration
reset_container_config()
reset_container()
```

### Utility Functions

```python
from saplings.api.utils import (
    setup_logging,             # Configure logging
    get_version,              # Get library version
    check_dependencies,       # Check installed dependencies
    setup_environment        # Setup environment variables
)

# Setup logging
setup_logging(level="INFO", format="structured")

# Check dependencies
missing = check_dependencies(["torch", "transformers"])
if missing:
    print(f"Missing dependencies: {missing}")
```

### Stability Annotations

The library uses stability annotations to indicate API maturity:

- **`@stable`**: Stable API that follows semantic versioning
- **`@beta`**: Beta API that may change in minor versions
- **`@experimental`**: Experimental API that may change significantly

```python
from saplings.api.stability import stable, beta, experimental

@stable
class StableClass:
    """This class has a stable API."""
    pass

@beta
class BetaClass:
    """This class is in beta and may change."""
    pass

@experimental
class ExperimentalClass:
    """This class is experimental and may change significantly."""
    pass
```

## Error Handling

### Common Exceptions

```python
from saplings.core.exceptions import (
    SaplingsError,             # Base exception
    ConfigurationError,        # Configuration issues
    ModelError,               # Model-related errors
    ToolError,                # Tool execution errors
    RetrievalError,           # Retrieval failures
    PlanningError,            # Planning failures
    ValidationError           # Validation failures
)

try:
    agent = Agent(provider="invalid", model_name="test")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Suggestions: {e.suggestions}")
```

### Self-Healing

```python
from saplings.api.self_heal import SelfHealingService

class SelfHealingService:
    async def heal_error(self, error: Exception, context: dict) -> dict[str, Any]:
        """Attempt to heal from an error."""
    
    async def optimize_performance(self, metrics: dict) -> dict[str, Any]:
        """Optimize based on performance metrics."""
    
    async def learn_from_feedback(self, feedback: dict) -> None:
        """Learn from user feedback."""
```

This API reference covers the major components of the Saplings framework. For more detailed information about specific features, refer to the individual module documentation and examples.