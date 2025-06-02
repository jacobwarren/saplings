from __future__ import annotations

"""
Service interfaces for Saplings.

This module provides interfaces for all services used in the Saplings framework.
These interfaces define the contracts that service implementations must follow,
enabling dependency inversion and more flexible composition.
"""


from abc import ABC, abstractmethod
from typing import Any, Callable

# Forward references for type hints
Document = Any  # From saplings.memory
PlanStep = Any  # From saplings.planner
LLM = Any  # From saplings.core.model_adapter


class IModelService(ABC):
    """
    Interface for model service operations.

    DEPRECATED: Use IModelInitializationService and IModelCachingService instead.
    This interface is kept for backward compatibility only.
    """

    @abstractmethod
    async def get_model(self, timeout: float | None = None) -> LLM:
        """
        Get the configured model instance.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            LLM: The model instance

        """

    @abstractmethod
    async def get_model_metadata(self, timeout: float | None = None) -> dict[str, Any]:
        """
        Get metadata about the model.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict[str, Any]: Model metadata

        """

    @abstractmethod
    async def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, timeout: float | None = None
    ) -> float:
        """
        Estimate the cost of a model call based on token counts.

        Args:
        ----
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            timeout: Optional timeout in seconds

        Returns:
        -------
            float: Estimated cost in USD

        """


class IMemoryManager(ABC):
    """Interface for memory management operations."""

    @abstractmethod
    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Document:
        """
        Add a document to memory.

        Args:
        ----
            content: Document content
            metadata: Optional metadata
            timeout: Optional timeout in seconds

        Returns:
        -------
            Document: The created document

        """

    @abstractmethod
    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt", timeout: float | None = None
    ) -> list[Document]:
        """
        Add documents from a directory.

        Args:
        ----
            directory: Directory path
            extension: File extension
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: The created documents

        """

    @abstractmethod
    async def get_document(self, document_id: str, timeout: float | None = None) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: Document ID
            timeout: Optional timeout in seconds

        Returns:
        -------
            Optional[Document]: The document if found

        """

    @abstractmethod
    async def get_documents(
        self,
        filter_func: Callable[[Document], bool] | None = None,
        timeout: float | None = None,
    ) -> list[Document]:
        """
        Get documents, optionally filtered.

        Args:
        ----
            filter_func: Optional filter function
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: The matching documents

        """


class IRetrievalService(ABC):
    """Interface for retrieval operations."""

    @abstractmethod
    async def retrieve(
        self, query: str, limit: int | None = None, timeout: float | None = None
    ) -> list[Document]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: The search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: Retrieved documents

        """


class IPlannerService(ABC):
    """Interface for planning operations."""

    @abstractmethod
    async def create_plan(
        self, task: str, context: list[Document] | None = None, trace_id: str | None = None
    ) -> list[PlanStep]:
        """Create a plan for a task."""


class IExecutionService(ABC):
    """Interface for execution operations."""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """Execute a prompt with the model."""


class IValidatorService(ABC):
    """Interface for validation operations."""

    @abstractmethod
    async def validate(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: str | None = None,
    ) -> Any:
        """Validate output against input data."""

    @abstractmethod
    async def judge_output(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Judge output quality."""

    @abstractmethod
    async def set_judge(self, judge: Any) -> None:
        """Set the judge for validation."""

    @abstractmethod
    def get_validation_history(self):
        """Get validation history."""


class ISelfHealingService(ABC):
    """Interface for self-healing operations."""

    @abstractmethod
    async def collect_success_pair(
        self,
        input_text: str,
        output_text: str,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> bool:
        """Collect a success pair for future improvements."""

    @abstractmethod
    async def get_all_success_pairs(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Get all collected success pairs."""

    @abstractmethod
    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a patch for a failed execution."""

    @abstractmethod
    async def apply_patch(
        self,
        patch: dict[str, Any],
        code_context: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a patch to fix code."""


class IToolService(ABC):
    """Interface for tool operations."""

    @abstractmethod
    def register_tool(self, tool: Any) -> bool:
        """Register a tool with the service."""

    @abstractmethod
    def prepare_functions_for_model(self):
        """Prepare tool definitions for the model."""

    @abstractmethod
    async def create_tool(
        self, name: str, description: str, code: str, trace_id: str | None = None
    ) -> Any:
        """Create a dynamic tool."""

    @abstractmethod
    def get_registered_tools(self):
        """Get all registered tools."""

    @property
    @abstractmethod
    def tools(self):
        """Get all tools."""


class IModalityService(ABC):
    """Interface for modality operations."""

    @abstractmethod
    def supported_modalities(self):
        """Get supported modalities."""

    @abstractmethod
    async def format_output(
        self, content: Any, output_modality: str, trace_id: str | None = None
    ) -> Any:
        """Format output for a specific modality."""


class IMonitoringService(ABC):
    """Interface for monitoring operations."""

    @property
    @abstractmethod
    def enabled(self):
        """Whether monitoring is enabled."""

    @property
    @abstractmethod
    def trace_manager(self):
        """Get the trace manager if enabled."""

    @abstractmethod
    def create_trace(self):
        """Create a new trace."""

    @abstractmethod
    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Start a new span in the trace."""

    @abstractmethod
    def end_span(self, span_id: str) -> None:
        """End a span in the trace."""

    @abstractmethod
    def process_trace(self, trace_id: str) -> Any:
        """Process a trace for analysis."""

    @abstractmethod
    def identify_bottlenecks(
        self, threshold_ms: float = 100.0, min_call_count: int = 1
    ) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""

    @abstractmethod
    def identify_error_sources(
        self, min_error_rate: float = 0.1, min_call_count: int = 1
    ) -> list[dict[str, Any]]:
        """Identify error sources."""


class IOrchestrationService(ABC):
    """Interface for orchestration operations."""

    @abstractmethod
    async def orchestrate(
        self,
        workflow: dict[str, Any],
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Orchestrate a workflow."""
