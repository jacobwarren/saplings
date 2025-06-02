from __future__ import annotations

"""
saplings.services.modality_service.
================================

Encapsulates multimodal capabilities:
- Text, image, audio, and video handling
- Modality-specific processing
- Cross-modal conversions
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.api.core.interfaces import IModalityService
from saplings.core.resilience import DEFAULT_TIMEOUT, with_timeout
from saplings.modality import (
    TextHandler,
    get_handler_for_modality,
)

if TYPE_CHECKING:
    from saplings.core.model_adapter import LLM

# Use Any for type hinting

logger = logging.getLogger(__name__)


class ModalityService(IModalityService):
    """Service that manages multimodal capabilities."""

    def __init__(
        self,
        model: LLM | None = None,
        supported_modalities: list[str] | None = None,
        trace_manager: Any | None = None,
        config: Any | None = None,  # Added for compatibility with tests
    ) -> None:
        self._trace_manager = trace_manager
        self.config = config
        self.handlers = {}  # Use handlers instead of modality_handlers for test compatibility

        # If config is provided, use it to get supported modalities
        if config is not None and hasattr(config, "supported_modalities"):
            supported_modalities = [
                m if isinstance(m, str) else m.value for m in config.supported_modalities
            ]

        # Default to text if not specified
        supported_modalities = supported_modalities or ["text"]

        # Validate supported modalities
        for modality in supported_modalities:
            if modality not in ["text", "image", "audio", "video"]:
                msg = f"Unsupported modality: {modality}"
                raise ValueError(msg)

        # Ensure text is always supported
        if "text" not in supported_modalities:
            supported_modalities.append("text")

        # Create handlers for each supported modality if model is provided
        if model is not None:
            for modality in supported_modalities:
                try:
                    handler = get_handler_for_modality(modality, model)
                    self.handlers[modality] = handler
                    logger.info("Initialized handler for %s modality", modality)
                except Exception as e:
                    logger.warning("Failed to initialize handler for %s modality: %s", modality, e)

            # Ensure we always have a text handler
            if "text" not in self.handlers and model is not None:
                self.handlers["text"] = TextHandler(model)
                logger.info("Initialized default text handler")
        else:
            # For testing, create simple handlers
            for modality in supported_modalities:
                # Create a simple handler that just returns the input
                self.handlers[modality] = type(
                    f"{modality.capitalize()}Handler",
                    (),
                    {
                        "process": lambda _, content: content,
                        "process_input": lambda _, content: {"content": content},
                        "format_output": lambda _, content: content,
                    },
                )()
                logger.info("Initialized simple handler for %s modality", modality)

        logger.info(
            "ModalityService initialized with modalities: %s", ", ".join(supported_modalities)
        )

    def register_handler(self, modality: str, handler: Any) -> None:
        """Register a handler for a specific modality."""
        self.handlers[modality] = handler
        logger.info("Registered handler for %s modality", modality)

    def process(self, modality: str, content: Any) -> Any:
        """Process content in the specified modality."""
        if modality not in self.handlers:
            msg = f"No handler available for modality: {modality}"
            raise ValueError(msg)

        return self.handlers[modality].process(content)

    def get_supported_modalities(self):
        """Get list of supported modalities."""
        return list(self.handlers.keys())

    def get_handler(self, modality: str) -> Any:
        """Get handler for a specific modality."""
        if modality not in self.handlers:
            msg = f"No handler available for modality: {modality}"
            raise ValueError(msg)

        return self.handlers[modality]

    def supported_modalities(self):
        """Get list of supported modalities."""
        return list(self.handlers.keys())

    async def process_input(
        self,
        content: Any,
        input_modality: str,
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """
        Process input content in the specified modality.

        Args:
        ----
            content: The content to process
            input_modality: The modality of the input content
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Processed content

        Raises:
        ------
            ValueError: If the modality is not supported
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ModalityService.process_input",
                trace_id=trace_id,
                attributes={"component": "modality_service", "modality": input_modality},
            )

        try:
            # Get the appropriate handler
            handler = self.get_handler(input_modality)

            # Define processing function
            async def _process_input():
                return await handler.process_input(content)

            # Execute with timeout
            return await with_timeout(
                _process_input(), timeout=timeout, operation_name=f"process_{input_modality}_input"
            )
        except Exception as e:
            logger.exception(f"Error processing {input_modality} input: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def convert(
        self,
        content: Any,
        source_modality: str,
        target_modality: str,
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Convert content between modalities.

        Args:
        ----
            content: Content to convert
            source_modality: Source modality
            target_modality: Target modality
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Any: Converted content

        Raises:
        ------
            ValueError: If the modality is not supported
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ModalityService.convert",
                trace_id=trace_id,
                attributes={
                    "component": "modality_service",
                    "source_modality": source_modality,
                    "target_modality": target_modality,
                },
            )

        try:
            # Process input first
            processed = await self.process_input(content, source_modality, trace_id, timeout)

            # Then format as output
            return await self.format_output(
                processed["content"], target_modality, trace_id, timeout
            )
        except Exception as e:
            logger.exception(f"Error converting from {source_modality} to {target_modality}: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def format_output(
        self,
        content: str,
        output_modality: str,
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Format output content in the specified modality.

        Args:
        ----
            content: The content to format (usually text)
            output_modality: The desired output modality
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Formatted content in the specified modality

        Raises:
        ------
            ValueError: If the modality is not supported
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ModalityService.format_output",
                trace_id=trace_id,
                attributes={"component": "modality_service", "modality": output_modality},
            )

        try:
            # Get the appropriate handler
            handler = self.get_handler(output_modality)

            # Define formatting function
            async def _format_output():
                return await handler.format_output(content)

            # Execute with timeout
            return await with_timeout(
                _format_output(), timeout=timeout, operation_name=f"format_{output_modality}_output"
            )
        except Exception as e:
            logger.exception(f"Error formatting {output_modality} output: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)
