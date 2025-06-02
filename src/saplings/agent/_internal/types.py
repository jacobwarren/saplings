from __future__ import annotations

"""
Core types for the agent component.

This module defines the core types used by the agent component to avoid circular imports.
All Protocol classes are defined here to break circular dependencies between implementation files.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union


class AgentFacadeProtocol(Protocol):
    """Protocol for AgentFacade class to avoid circular imports."""

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Add a document to the agent's memory."""
        ...

    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt"
    ) -> List[Any]:
        """Add documents from a directory."""
        ...

    async def execute_plan(
        self, plan: List[Any], context: Optional[List[Any]] = None, use_tools: bool = True
    ) -> Any:
        """Execute a plan."""
        ...

    async def register_tool(self, tool: Any) -> None:
        """Register a tool with the agent."""
        ...

    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """Create a dynamic tool."""
        ...

    async def judge_output(
        self, input_data: Dict[str, Any], output_data: Any, judgment_type: str = "general"
    ) -> Dict[str, Any]:
        """Judge an output."""
        ...

    async def self_improve(self) -> Dict[str, Any]:
        """Improve the agent based on past performance."""
        ...

    async def run(self, task: str, **kwargs) -> Union[Dict[str, Any], str]:
        """Run the agent on a task, handling the full lifecycle."""
        ...

    async def retrieve(
        self, query: str, limit: Optional[int] = None, fast_mode: bool = False
    ) -> List[Any]:
        """Retrieve documents based on a query."""
        ...

    async def plan(self, task: str, context: Optional[List[Any]] = None) -> List[Any]:
        """Create a plan for a task."""
        ...

    async def execute(
        self, prompt: str, context: Optional[List[Any]] = None, use_tools: bool = True
    ) -> Any:
        """Execute a prompt with the agent."""
        ...


class AgentFacadeBuilderProtocol(Protocol):
    """Protocol for AgentFacadeBuilder class to avoid circular imports."""

    def with_config(self, config: Any) -> AgentFacadeBuilderProtocol:
        """Set the agent configuration."""
        ...

    def with_testing(self, testing: bool) -> AgentFacadeBuilderProtocol:
        """Set testing mode."""
        ...

    def with_monitoring_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the monitoring service."""
        ...

    def with_model_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the model service."""
        ...

    def with_memory_manager(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the memory manager."""
        ...

    def with_retrieval_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the retrieval service."""
        ...

    def with_validator_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the validator service."""
        ...

    def with_execution_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the execution service."""
        ...

    def with_planner_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the planner service."""
        ...

    def with_tool_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the tool service."""
        ...

    def with_self_healing_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the self-healing service."""
        ...

    def with_modality_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the modality service."""
        ...

    def with_orchestration_service(self, service: Any) -> AgentFacadeBuilderProtocol:
        """Set the orchestration service."""
        ...

    def build(self) -> Any:
        """Build the agent facade instance."""
        ...


class AgentProtocol(Protocol):
    """Protocol for Agent class to avoid circular imports."""

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Add a document to the agent's memory."""
        ...

    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt"
    ) -> List[Any]:
        """Add documents from a directory."""
        ...

    async def run(self, task: str, **kwargs) -> Any:
        """Run a task with the agent."""
        ...

    async def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent."""
        ...

    async def create_tool(self, name: str, description: str, code: str) -> Any:
        """Create a new tool dynamically."""
        ...

    async def retrieve(
        self, query: str, limit: Optional[int] = None, fast_mode: bool = False
    ) -> List[Any]:
        """Retrieve documents based on a query."""
        ...

    async def plan(self, task: str, context: Optional[List[Any]] = None) -> List[Any]:
        """Create a plan for a task."""
        ...

    async def execute(
        self, prompt: str, context: Optional[List[Any]] = None, use_tools: bool = True
    ) -> Any:
        """Execute a prompt with the agent."""
        ...


class AgentBuilderProtocol(Protocol):
    """Protocol for AgentBuilder class to avoid circular imports."""

    def with_provider(self, provider: str) -> AgentBuilderProtocol:
        """Set the model provider."""
        ...

    def with_model_name(self, model_name: str) -> AgentBuilderProtocol:
        """Set the model name."""
        ...

    def with_memory_path(self, memory_path: str) -> AgentBuilderProtocol:
        """Set the memory path."""
        ...

    def with_output_dir(self, output_dir: str) -> AgentBuilderProtocol:
        """Set the output directory."""
        ...

    def with_gasa_enabled(self, enabled: bool) -> AgentBuilderProtocol:
        """Set whether GASA is enabled."""
        ...

    def with_monitoring_enabled(self, enabled: bool) -> AgentBuilderProtocol:
        """Set whether monitoring is enabled."""
        ...

    def with_self_healing_enabled(self, enabled: bool) -> AgentBuilderProtocol:
        """Set whether self-healing is enabled."""
        ...

    def with_tool_factory_enabled(self, enabled: bool) -> AgentBuilderProtocol:
        """Set whether the tool factory is enabled."""
        ...

    def build(self, use_container: Optional[bool] = None) -> Any:
        """Build the agent instance."""
        ...


class AgentConfigProtocol(Protocol):
    """Protocol for AgentConfig class to avoid circular imports."""

    provider: str
    model_name: str
    memory_path: str
    output_dir: str
    enable_gasa: bool
    enable_monitoring: bool
    enable_self_healing: bool
    self_healing_max_retries: int
    enable_tool_factory: bool
    max_tokens: int
    temperature: float


# Factory function types
AgentFactory = TypeVar("AgentFactory", bound=Callable[..., Any])
AgentFacadeFactory = TypeVar("AgentFacadeFactory", bound=Callable[..., Any])


__all__ = [
    "AgentProtocol",
    "AgentBuilderProtocol",
    "AgentConfigProtocol",
    "AgentFacadeProtocol",
    "AgentFacadeBuilderProtocol",
    "AgentFactory",
    "AgentFacadeFactory",
]
