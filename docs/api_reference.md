# API Reference

This document provides a comprehensive reference for the Saplings API, including all public classes, methods, and functions.

## Core Components

### Agent

The main entry point for using Saplings.

```python
class Agent:
    def __init__(self, config: AgentConfig):
        """
        Initialize an agent.

        Args:
            config: Agent configuration
        """

    async def run(
        self,
        prompt: str,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Run the agent on a prompt.

        Args:
            prompt: The prompt to run
            input_modalities: List of input modalities
            output_modalities: List of output modalities
            **kwargs: Additional arguments

        Returns:
            The agent's response
        """
```

### AgentConfig

Configuration for the Agent class.

```python
class AgentConfig:
    provider: str  # Model provider (e.g., "openai", "anthropic", "vllm")
    model_name: str  # Model name (e.g., "gpt-4o", "claude-3-opus")

    # Memory configuration
    memory_path: Optional[str] = None  # Path to store memory

    # GASA configuration
    enable_gasa: bool = False  # Whether to enable GASA
    gasa_max_hops: int = 2  # Maximum hops for GASA
    gasa_strategy: str = "binary"  # GASA mask strategy
    gasa_fallback: str = "block_diagonal"  # GASA fallback strategy
    gasa_shadow_model: bool = False  # Whether to use shadow model for tokenization
    gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B"  # Shadow model name
    gasa_prompt_composer: bool = False  # Whether to use prompt composer

    # Tool configuration
    tools: Optional[List[Tool]] = None  # List of tools

    # Modality configuration
    supported_modalities: Optional[List[str]] = None  # List of supported modalities

    # Self-healing configuration
    enable_self_healing: bool = False  # Whether to enable self-healing

    # Model caching configuration
    enable_model_caching: bool = False  # Whether to enable model caching
    cache_ttl: int = 3600  # Cache TTL in seconds
    cache_max_size: int = 1000  # Maximum cache size
    cache_storage_path: Optional[str] = None  # Path to store cache
```

## Model Adapters

### LLM

Abstract base class for all model adapters.

```python
class LLM:
    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs) -> "LLM":
        """
        Create a model instance.

        Args:
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name
            **kwargs: Additional arguments

        Returns:
            LLM instance
        """

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function specifications
            function_call: Function call configuration
            json_mode: Whether to enable JSON mode
            **kwargs: Additional arguments

        Returns:
            The generated response
        """

    async def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate text from the model with streaming.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function specifications
            function_call: Function call configuration
            json_mode: Whether to enable JSON mode
            **kwargs: Additional arguments

        Returns:
            Generator yielding response chunks
        """
```

### LLMResponse

Response from an LLM.

```python
class LLMResponse:
    text: str  # Generated text
    function_call: Optional[Dict[str, Any]]  # Function call information
    usage: Dict[str, int]  # Token usage information
    model: str  # Model name
    finish_reason: Optional[str]  # Reason for finishing generation
    latency_ms: float  # Latency in milliseconds
```

## Memory System

### MemoryStore

Manages document storage and retrieval.

```python
class MemoryStore:
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize a memory store.

        Args:
            config: Memory configuration
        """

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Document:
        """
        Add a document to the memory store.

        Args:
            content: Document content
            metadata: Document metadata
            id: Document ID
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            The added document
        """

    def get_document(self, id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            id: Document ID

        Returns:
            The document, or None if not found
        """

    def get_documents(self) -> List[Document]:
        """
        Get all documents.

        Returns:
            List of documents
        """

    def get_chunks(self, document_id: Optional[str] = None) -> List[DocumentChunk]:
        """
        Get chunks from a document or all documents.

        Args:
            document_id: Document ID, or None for all documents

        Returns:
            List of chunks
        """
```

### Document

Represents a document in the memory store.

```python
class Document:
    id: str  # Document ID
    content: str  # Document content
    metadata: Dict[str, Any]  # Document metadata
    chunks: List[DocumentChunk]  # Document chunks
    embedding: Optional[List[float]]  # Document embedding
```

### DependencyGraph

Represents relationships between documents.

```python
class DependencyGraph:
    def __init__(self):
        """Initialize a dependency graph."""

    async def build_from_memory(self, memory: MemoryStore) -> None:
        """
        Build the graph from a memory store.

        Args:
            memory: Memory store
        """

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
    ) -> None:
        """
        Add a relationship between documents.

        Args:
            source_id: Source document ID
            target_id: Target document ID
            relationship_type: Type of relationship
            weight: Relationship weight
        """

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node ID
            max_hops: Maximum number of hops
            relationship_types: Types of relationships to consider

        Returns:
            Dictionary mapping neighbor IDs to weights
        """
```

## Retrieval System

### CascadeRetriever

Orchestrates the retrieval pipeline.

```python
class CascadeRetriever:
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize a cascade retriever.

        Args:
            config: Retrieval configuration
        """

    async def retrieve(
        self,
        query: str,
        memory: MemoryStore,
        graph: Optional[DependencyGraph] = None,
        max_documents: int = 10,
        min_similarity: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents relevant to a query.

        Args:
            query: Query string
            memory: Memory store
            graph: Dependency graph
            max_documents: Maximum number of documents to retrieve
            min_similarity: Minimum similarity score
            **kwargs: Additional arguments

        Returns:
            List of relevant documents
        """
```

## GASA System

### GASAConfig

Configuration for GASA.

```python
class GASAConfig:
    max_hops: int = 2  # Maximum hops for GASA
    mask_strategy: str = "binary"  # GASA mask strategy
    fallback_strategy: str = "block_diagonal"  # GASA fallback strategy
    cache_masks: bool = True  # Whether to cache masks
    enable_shadow_model: bool = False  # Whether to use shadow model for tokenization
    shadow_model_name: str = "Qwen/Qwen3-0.6B"  # Shadow model name
    shadow_model_device: str = "cpu"  # Shadow model device
    enable_prompt_composer: bool = False  # Whether to use prompt composer
    focus_tags: bool = True  # Whether to use focus tags
```

### GASAService

Provides GASA functionality.

```python
class GASAService:
    def __init__(self, config: GASAConfig):
        """
        Initialize the GASA service.

        Args:
            config: GASA configuration
        """

    def build_mask(
        self,
        documents: List[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> np.ndarray:
        """
        Build a GASA mask.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            format: Mask format
            mask_type: Mask type

        Returns:
            GASA mask
        """

    def apply_gasa(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: Optional[List[int]] = None,
        attention_mask: Optional[np.ndarray] = None,
        model_supports_sparse_attention: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply GASA to a prompt and inputs.

        Args:
            documents: Documents used in the prompt
            prompt: Prompt text
            input_ids: Token IDs
            attention_mask: Attention mask
            model_supports_sparse_attention: Whether the model supports sparse attention
            **kwargs: Additional arguments

        Returns:
            Result containing modified prompt, input_ids, and attention_mask
        """

    def visualize_mask(
        self,
        mask: np.ndarray,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
        output_path: Optional[str] = None,
        title: str = "GASA Mask",
    ) -> None:
        """
        Visualize a GASA mask.

        Args:
            mask: GASA mask
            format: Mask format
            mask_type: Mask type
            output_path: Output path for the visualization
            title: Visualization title
        """
```

## Tools System

### Tool

Base class for all tools.

```python
class Tool:
    name: str  # Tool name
    description: str  # Tool description
    parameters: Dict[str, Any]  # Tool parameters
    output_type: str  # Tool output type

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool result
        """

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward method to be implemented by subclasses.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool result
        """
```

### ToolRegistry

Registry for managing tools.

```python
class ToolRegistry:
    def __init__(self):
        """Initialize a tool registry."""

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register
        """

    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            The tool, or None if not found
        """

    def list(self) -> List[str]:
        """
        List all registered tools.

        Returns:
            List of tool names
        """

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI function specifications.

        Returns:
            List of function specifications
        """
```

### MCPClient

Client for Machine Control Protocol servers.

```python
class MCPClient:
    def __init__(
        self,
        server_parameters: Union[
            "StdioServerParameters",
            Dict[str, Any],
            List[Union["StdioServerParameters", Dict[str, Any]]]
        ],
    ):
        """
        Initialize the MCP client.

        Args:
            server_parameters: MCP server parameters
        """

    def connect(self) -> None:
        """Connect to the MCP server and initialize the tools."""

    def disconnect(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        """
        Disconnect from the MCP server.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised
        """

    def get_tools(self) -> List[Tool]:
        """
        Get the Saplings tools available from the MCP server.

        Returns:
            List of tools
        """

    def __enter__(self) -> List[Tool]:
        """
        Connect to the MCP server and return the tools directly.

        Returns:
            List of tools
        """

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """
        Disconnect from the MCP server.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_traceback: Exception traceback if an exception was raised
        """
```

## Modality System

### ModalityService

Manages modality handlers and orchestrates multimodal operations.

```python
class ModalityService:
    def __init__(
        self,
        model: LLM,
        supported_modalities: Optional[List[str]] = None,
        trace_manager: Optional["TraceManager"] = None,
    ) -> None:
        """
        Initialize the modality service.

        Args:
            model: LLM model to use for processing
            supported_modalities: List of supported modalities
            trace_manager: Optional trace manager for monitoring
        """

    def get_handler(self, modality: str) -> Any:
        """
        Get handler for a specific modality.

        Args:
            modality: Modality name

        Returns:
            ModalityHandler for the specified modality
        """

    def supported_modalities(self) -> List[str]:
        """
        Get list of supported modalities.

        Returns:
            List of supported modality names
        """

    async def process_input(
        self,
        content: Any,
        input_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process input content in the specified modality.

        Args:
            content: The content to process
            input_modality: The modality of the input content
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
            Processed content
        """

    async def format_output(
        self,
        content: str,
        output_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Format output content in the specified modality.

        Args:
            content: The content to format
            output_modality: The desired output modality
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
            Formatted content
        """
```

## Utility Functions

### Token Management

```python
def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count the number of tokens in a text.

    Args:
        text: Text to count tokens for
        model_name: Model name for tokenization

    Returns:
        Number of tokens
    """

def split_text_by_tokens(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Split text into chunks by token count.

    Args:
        text: Text to split
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        model_name: Model name for tokenization

    Returns:
        List of text chunks
    """

def truncate_text_tokens(
    text: str,
    max_tokens: int,
    model_name: Optional[str] = None,
) -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model_name: Model name for tokenization

    Returns:
        Truncated text
    """

def get_tokens_remaining(
    prompt: str,
    max_tokens: int,
    model_name: Optional[str] = None,
) -> int:
    """
    Get the number of tokens remaining after a prompt.

    Args:
        prompt: Prompt text
        max_tokens: Maximum number of tokens
        model_name: Model name for tokenization

    Returns:
        Number of tokens remaining
    """
```

## Plugin System

### Plugin

Base class for all plugins.

```python
class Plugin:
    """Base class for all plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """
        Get the plugin type.

        Returns:
            Plugin type
        """

    @property
    def plugin_id(self) -> str:
        """
        Get the plugin ID.

        Returns:
            Plugin ID
        """
```

### PluginRegistry

Registry for managing plugins.

```python
class PluginRegistry:
    def __init__(self):
        """Initialize a plugin registry."""

    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin.

        Args:
            plugin: Plugin to register
        """

    def get(self, plugin_id: str) -> Optional[Plugin]:
        """
        Get a plugin by ID.

        Args:
            plugin_id: Plugin ID

        Returns:
            The plugin, or None if not found
        """

    def get_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """
        Get plugins by type.

        Args:
            plugin_type: Plugin type

        Returns:
            List of plugins
        """
```

## Dependency Injection

### Container

Central registry for services and their dependencies.

```python
class Container:
    def __init__(self):
        """Initialize a container."""

    def register(
        self,
        service_type: Type,
        instance: Optional[Any] = None,
        factory: Optional[Callable[..., Any]] = None,
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        """
        Register a service.

        Args:
            service_type: Service type
            instance: Service instance
            factory: Factory function
            scope: Service scope
        """

    def resolve(self, service_type: Type) -> Any:
        """
        Resolve a service.

        Args:
            service_type: Service type

        Returns:
            Service instance
        """
```

## Exceptions

```python
class SaplingsError(Exception):
    """Base class for all Saplings exceptions."""

class ConfigurationError(SaplingsError):
    """Error in configuration."""

class ModelError(SaplingsError):
    """Error in model operations."""

class ProviderError(SaplingsError):
    """Error in provider operations."""

class ResourceExhaustedError(SaplingsError):
    """Resource exhausted error."""
```

## Complete API Reference

For a complete API reference, see the source code and docstrings in the Saplings codebase.
