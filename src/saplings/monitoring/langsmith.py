from __future__ import annotations

"""
LangSmith integration module for Saplings monitoring.

This module provides functionality to export monitoring data to LangSmith
for advanced analysis, visualization, and debugging of agent workflows.
LangSmith integration enables comprehensive tracing of agent execution,
including LLM calls, tool usage, and performance metrics.
"""


import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from saplings.core.config_service import config_service
from saplings.monitoring.config import MonitoringConfig
from saplings.version import __version__

if TYPE_CHECKING:
    import langsmith  # type: ignore[import-not-found]
    from langsmith import Client  # type: ignore[import-not-found]

    from saplings.monitoring.trace import Trace, TraceManager

logger = logging.getLogger(__name__)

# Runtime import
try:
    import langsmith  # type: ignore[import]
    from langsmith import Client  # type: ignore[import]

    LANGSMITH_AVAILABLE = True
except ImportError:
    langsmith = None
    Client = None
    LANGSMITH_AVAILABLE = False
    logger.warning(
        "LangSmith not installed. LangSmith export will not be available. "
        "Install LangSmith with: pip install saplings[langsmith]"
    )


def _require_langsmith() -> Any:
    """
    Ensure LangSmith is available and return the module.

    Raises
    ------
        ImportError: If LangSmith is not installed

    Returns
    -------
        The LangSmith module

    """
    if langsmith is None:
        raise ImportError("LangSmith is optional – install with `pip install saplings[langsmith]`.")
    return langsmith


class LangSmithExporter:
    """
    LangSmith exporter for monitoring data.

    This class provides functionality to export monitoring data to LangSmith
    for advanced analysis, visualization, and debugging of agent workflows.

    LangSmith integration enables:
    - Comprehensive tracing of agent execution
    - Visualization of complex agent workflows
    - Performance analysis and bottleneck identification
    - Sharing and collaboration on agent development
    - Integration with the broader LangChain ecosystem

    The exporter automatically maps Saplings traces and spans to LangSmith's
    run structure, preserving the hierarchical relationships and metadata.
    """

    def __init__(
        self,
        trace_manager: TraceManager | None = None,
        config: MonitoringConfig | None = None,
        api_key: str | None = None,
        project_name: str | None = None,
        auto_export: bool = False,
        export_interval: int = 60,
    ) -> None:
        """
        Initialize the LangSmith exporter.

        Args:
        ----
            trace_manager: Trace manager to use
            config: Monitoring configuration
            api_key: LangSmith API key (overrides config)
            project_name: LangSmith project name (overrides config)
            auto_export: Whether to automatically export traces when completed
            export_interval: Interval in seconds for automatic export (if auto_export is True)

        """
        self.trace_manager = trace_manager
        self.config = config or MonitoringConfig()

        # Initialize LangSmith client
        self.client = None
        self.api_key = (
            api_key
            or self.config.langsmith_api_key
            or config_service.get_value("LANGCHAIN_API_KEY")
        )
        self.project_name = (
            project_name
            or self.config.langsmith_project
            or config_service.get_value("LANGCHAIN_PROJECT")
        )

        # Auto-export settings
        self.auto_export = auto_export
        self.export_interval = export_interval
        self.last_export_time = datetime.now()
        self.exported_trace_ids = set()

        if self.api_key:
            try:
                # Ensure LangSmith is available
                if LANGSMITH_AVAILABLE:
                    # Get the Client class from the langsmith module
                    client_class = Client
                    if client_class is not None:
                        self.client = client_class(api_key=self.api_key)
                        logger.info("LangSmith client initialized successfully")

                        # Create project if it doesn't exist
                        if self.project_name and self.client:
                            try:
                                projects = self.client.list_projects()
                                project_names = [p.name for p in projects]
                                if self.project_name not in project_names:
                                    self.client.create_project(self.project_name)
                                    logger.info(f"Created LangSmith project: {self.project_name}")
                            except Exception as e:
                                logger.warning(f"Failed to create LangSmith project: {e}")
                else:
                    logger.warning(
                        "LangSmith not installed. LangSmith export will not be available. "
                        "Install LangSmith with: pip install saplings[langsmith]"
                    )
            except Exception as e:
                logger.exception(f"Failed to initialize LangSmith client: {e}")

        # Register with trace manager for auto-export if requested
        if self.auto_export and self.trace_manager:
            self.trace_manager.register_trace_callback(self._auto_export_callback)

    def export_trace(
        self,
        trace: str | Trace,
        project_name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Export a trace to LangSmith.

        Args:
        ----
            trace: Trace or trace ID to export
            project_name: LangSmith project name (overrides instance setting)
            tags: Tags to apply to the LangSmith run
            metadata: Additional metadata to include with the run

        Returns:
        -------
            Optional[str]: LangSmith run ID if successful

        """
        try:
            # Ensure LangSmith is available
            _require_langsmith()
        except ImportError as e:
            logger.error(f"Cannot export trace: {e}")
            return None

        if not self.client:
            logger.error("LangSmith client not initialized. Cannot export trace.")
            return None

        # Get trace if ID is provided
        if isinstance(trace, str):
            if not self.trace_manager:
                logger.error("Trace manager not provided, cannot get trace by ID")
                return None

            trace_obj = self.trace_manager.get_trace(trace)
            if not trace_obj:
                logger.error(f"Trace {trace} not found")
                return None
        else:
            trace_obj = trace

        # Use provided project name or default
        project = project_name or self.project_name
        if not project:
            logger.error("Project name not provided. Cannot export trace.")
            return None

        # Add trace ID to exported traces set
        if trace_obj is not None:
            self.exported_trace_ids.add(trace_obj.trace_id)

        try:
            # Convert trace to LangSmith format
            runs = self._convert_trace_to_langsmith(trace_obj)

            # Get root spans (those without parents)
            root_spans = trace_obj.get_root_spans()
            if not root_spans:
                logger.warning(f"No root spans found in trace {trace_obj.trace_id}")
                return None

            # Use the first root span as the main run
            root_span = root_spans[0]

            # Determine run name
            run_name = f"Trace: {trace_obj.trace_id}"
            if "task" in trace_obj.attributes:
                run_name = f"Task: {trace_obj.attributes['task']}"
            elif "name" in trace_obj.attributes:
                run_name = trace_obj.attributes["name"]

            # Determine run type based on root span
            run_type = "chain"
            if "component" in root_span.attributes:
                component = root_span.attributes["component"]
                if component == "llm":
                    run_type = "llm"
                elif component == "tool":
                    run_type = "tool"
                elif component == "agent":
                    run_type = "agent"

            # Prepare inputs and outputs
            inputs = {
                "trace_id": trace_obj.trace_id,
                **{k: v for k, v in trace_obj.attributes.items() if k != "status"},
            }

            outputs = {"status": trace_obj.status}
            if "result" in trace_obj.attributes:
                outputs["result"] = trace_obj.attributes["result"]

            # Prepare error information
            error_info = None
            if trace_obj.status == "ERROR":
                error_info = trace_obj.attributes.get("error", "Error occurred")
                if isinstance(error_info, dict) and "message" in error_info:
                    error_info = error_info["message"]
                # Ensure error is a string
                error_info = str(error_info) if error_info is not None else None

            # Combine provided metadata with trace attributes
            combined_metadata = {
                "saplings_version": __version__,  # Use actual version from package
                "trace_duration_ms": trace_obj.duration_ms(),
                "span_count": len(trace_obj.spans),
                **trace_obj.attributes,
                **(metadata or {}),
            }

            # Combine provided tags with any from trace attributes
            combined_tags = set(tags or [])
            if "tags" in trace_obj.attributes and isinstance(trace_obj.attributes["tags"], list):
                combined_tags.update(trace_obj.attributes["tags"])

            # Export to LangSmith
            run_id = self.client.create_run(
                name=run_name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                error=error_info,
                start_time=trace_obj.start_time,
                end_time=trace_obj.end_time or datetime.now(),
                extra=combined_metadata,
                tags=list(combined_tags),
                project_name=project,
                child_runs=[run for run in runs if run.get("parent_run_id") is None],
            )

            logger.info(f"Exported trace {trace_obj.trace_id} to LangSmith run {run_id}")
            return run_id
        except Exception as e:
            logger.exception(f"Failed to export trace to LangSmith: {e}")
            return None

    def export_traces(
        self,
        trace_ids: list[str] | None = None,
        project_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        skip_exported: bool = True,
    ) -> list[str]:
        """
        Export multiple traces to LangSmith.

        Args:
        ----
            trace_ids: IDs of traces to export (all if not provided)
            project_name: LangSmith project name (overrides instance setting)
            start_time: Start time for filtering traces
            end_time: End time for filtering traces
            tags: Tags to apply to all exported runs
            metadata: Additional metadata to include with all runs
            skip_exported: Whether to skip traces that have already been exported

        Returns:
        -------
            List[str]: List of LangSmith run IDs

        """
        if not self.trace_manager:
            logger.error("Trace manager not provided, cannot get traces")
            return []

        # Get traces to export
        if trace_ids:
            traces = [
                self.trace_manager.get_trace(trace_id)
                for trace_id in trace_ids
                if self.trace_manager.get_trace(trace_id) is not None
            ]
        else:
            traces = self.trace_manager.list_traces(start_time, end_time)

        # Filter out already exported traces if requested
        if skip_exported:
            traces = [
                t for t in traces if t is not None and t.trace_id not in self.exported_trace_ids
            ]

        if not traces:
            logger.info("No new traces to export")
            return []

        # Export each trace
        run_ids = []
        for trace in traces:
            if trace is not None:
                run_id = self.export_trace(
                    trace, project_name=project_name, tags=tags, metadata=metadata
                )
                if run_id:
                    run_ids.append(run_id)

        logger.info(f"Exported {len(run_ids)} traces to LangSmith")
        return run_ids

    def _auto_export_callback(self, trace_id: str, event: str) -> None:
        """
        Callback for automatic trace export.

        Args:
        ----
            trace_id: ID of the trace
            event: Event type (e.g., "created", "completed", "error")

        """
        if not self.auto_export:
            return

        # Only export completed traces
        if event != "completed":
            return

        # Check if it's time to export
        now = datetime.now()
        if (now - self.last_export_time).total_seconds() < self.export_interval:
            return

        # Update last export time
        self.last_export_time = now

        # Export the trace
        try:
            self.export_trace(trace_id)
        except Exception as e:
            logger.exception(f"Auto-export failed for trace {trace_id}: {e}")

    def check_connection(self):
        """
        Check if the LangSmith connection is working.

        Returns
        -------
            bool: True if connection is working, False otherwise

        """
        try:
            # Ensure LangSmith is available
            _require_langsmith()
        except ImportError:
            return False

        if not self.client:
            return False

        try:
            # Try to list projects as a simple API call
            self.client.list_projects()
            return True
        except Exception as e:
            logger.exception(f"LangSmith connection check failed: {e}")
            return False

    def _convert_trace_to_langsmith(self, trace: Trace) -> list[dict[str, Any]]:
        """
        Convert a trace to LangSmith format.

        This method maps Saplings traces and spans to LangSmith's run structure,
        preserving the hierarchical relationships and metadata. It intelligently
        determines the run type based on the span's component and attributes.

        Args:
        ----
            trace: Trace to convert

        Returns:
        -------
            List[Dict[str, Any]]: List of LangSmith run dictionaries

        """
        runs = []

        # Create a run for each span
        for span in trace.spans:
            # Determine run type based on component
            run_type = "chain"
            if "component" in span.attributes:
                component = span.attributes["component"]
                if component == "llm":
                    run_type = "llm"
                elif component == "tool":
                    run_type = "tool"
                elif component == "agent":
                    run_type = "agent"
                elif component == "retriever":
                    run_type = "retriever"

            # Extract inputs and outputs based on run type
            inputs = {}
            outputs = {}

            if run_type == "llm":
                # Extract LLM-specific inputs and outputs
                if "prompt" in span.attributes:
                    inputs["prompt"] = span.attributes["prompt"]
                elif "messages" in span.attributes:
                    inputs["messages"] = span.attributes["messages"]

                if "completion" in span.attributes:
                    outputs["completion"] = span.attributes["completion"]
                elif "response" in span.attributes:
                    outputs["completion"] = span.attributes["response"]

                # Add model information
                if "model" in span.attributes:
                    inputs["model"] = span.attributes["model"]
                elif "model_uri" in span.attributes:
                    inputs["model"] = span.attributes["model_uri"]

                # Add generation parameters
                for param in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                ]:
                    if param in span.attributes:
                        inputs[param] = span.attributes[param]

                # Add token usage
                if "tokens" in span.attributes:
                    outputs["tokens"] = span.attributes["tokens"]
                elif "token_usage" in span.attributes:
                    outputs["token_usage"] = span.attributes["token_usage"]

            elif run_type == "tool":
                # Extract tool-specific inputs and outputs
                if "input" in span.attributes:
                    inputs["input"] = span.attributes["input"]
                elif "args" in span.attributes:
                    inputs["args"] = span.attributes["args"]

                if "output" in span.attributes:
                    outputs["output"] = span.attributes["output"]
                elif "result" in span.attributes:
                    outputs["output"] = span.attributes["result"]

                # Add tool information
                if "tool_name" in span.attributes:
                    inputs["tool_name"] = span.attributes["tool_name"]

            elif run_type == "retriever":
                # Extract retriever-specific inputs and outputs
                if "query" in span.attributes:
                    inputs["query"] = span.attributes["query"]

                if "documents" in span.attributes:
                    outputs["documents"] = span.attributes["documents"]

                # Add retriever information
                if "retriever_type" in span.attributes:
                    inputs["retriever_type"] = span.attributes["retriever_type"]

            # Add any remaining attributes as inputs
            for key, value in span.attributes.items():
                if key not in [
                    "component",
                    "prompt",
                    "messages",
                    "completion",
                    "response",
                    "model",
                    "model_uri",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "tokens",
                    "token_usage",
                    "input",
                    "args",
                    "output",
                    "result",
                    "tool_name",
                    "query",
                    "documents",
                    "retriever_type",
                    "status",
                    "error",
                ]:
                    inputs[key] = value

            # Add status and duration to outputs
            outputs["status"] = span.status
            outputs["duration_ms"] = span.duration_ms()

            # Extract error information
            error_info = None
            if span.status == "ERROR":
                error_info = span.attributes.get("error", "Error occurred")
                if isinstance(error_info, dict) and "message" in error_info:
                    error_info = error_info["message"]
                # Ensure error is a string
                error_info = str(error_info) if error_info is not None else None

            # Create run dictionary
            run = {
                "id": span.span_id,
                "name": span.name,
                "run_type": run_type,
                "inputs": inputs,
                "outputs": outputs,
                "error": error_info,
                "start_time": span.start_time,
                "end_time": span.end_time or datetime.now(),
                "parent_run_id": span.parent_id,
                "extra": {
                    "trace_id": trace.trace_id,
                    "span_id": span.span_id,
                    "component": span.attributes.get("component", "unknown"),
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp.isoformat(),
                            "attributes": event.attributes,
                        }
                        for event in span.events
                    ],
                },
            }

            runs.append(run)

        return runs
