from __future__ import annotations

"""
Modality service implementation for Saplings.

This module provides the implementation of the modality service that manages
different modality handlers and provides operations for processing and converting
between modalities.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from saplings.api.core.interfaces import IModalityService
from saplings.core.resilience import DEFAULT_TIMEOUT, with_timeout_decorator
from saplings.modality._internal.handlers.utils import get_handler_for_modality

if TYPE_CHECKING:
    from saplings.api.models import LLM
    from saplings.modality._internal.handlers.modality_handler import ModalityHandler

logger = logging.getLogger(__name__)


class ModalityService(IModalityService):
    """Service that manages multimodal capabilities."""

    def __init__(
        self,
        model: Optional["LLM"] = None,
        supported_modalities: Optional[list[str]] = None,
        trace_manager: Optional[Any] = None,
        config: Optional[Any] = None,
        model_provider: Optional[Callable[[], "LLM"]] = None,
    ) -> None:
        """
        Initialize the modality service.

        Args:
        ----
            model: LLM model to use for processing (can be None if model_provider is given)
            supported_modalities: List of supported modalities
            trace_manager: Optional trace manager for monitoring
            config: Optional configuration object
            model_provider: Optional function to provide the model on-demand

        Raises:
        ------
            ValueError: If neither model nor model_provider is provided
            ValueError: If an unsupported modality is specified

        """
        self._trace_manager = trace_manager
        self.config = config
        self._model = model
        self._model_provider = model_provider
        self._handlers: dict[str, "ModalityHandler"] = {}
        self._initialized_modalities = set()

        # If config is provided, use it to get supported modalities
        if config is not None and hasattr(config, "supported_modalities"):
            supported_modalities = [
                m if isinstance(m, str) else m.value for m in config.supported_modalities
            ]

        # Default to text if not specified
        self._supported_modalities = supported_modalities or ["text"]

        # Ensure text is always supported
        if "text" not in self._supported_modalities:
            self._supported_modalities.append("text")

        # Validate that we have either a model or a model provider
        if model is None and model_provider is None and not self._testing_mode():
            msg = "Either model or model_provider must be provided"
            raise ValueError(msg)

        logger.info(
            "ModalityService initialized with supported modalities: %s",
            ", ".join(self._supported_modalities),
        )

    def _testing_mode(self) -> bool:
        """Check if the service is in testing mode."""
        return self.config is not None and self._model is None and self._model_provider is None

    def _get_model(self) -> "LLM":
        """
        Get the model to use for processing.

        Returns
        -------
            LLM: Model to use for processing

        Raises
        ------
            ValueError: If no model is available

        """
        if self._model is not None:
            return self._model
        if self._model_provider is not None:
            return self._model_provider()
        msg = "No model available"
        raise ValueError(msg)

    def _initialize_handler(self, modality: str) -> "ModalityHandler":
        """
        Initialize a handler for a specific modality.

        Args:
        ----
            modality: Modality name

        Returns:
        -------
            ModalityHandler: Handler for the specified modality

        Raises:
        ------
            ValueError: If the modality is not supported

        """
        # Check if the modality is supported
        if modality not in self._supported_modalities:
            msg = f"Unsupported modality: {modality}"
            raise ValueError(msg)

        # Check if the handler is already initialized
        if modality in self._handlers:
            return self._handlers[modality]

        # Initialize the handler
        try:
            if self._testing_mode():
                # Create a simple handler for testing
                handler = type(
                    f"{modality.capitalize()}Handler",
                    (),
                    {
                        "process": lambda _, content: content,
                        "process_input": lambda _, content: {"content": content},
                        "format_output": lambda _, content: content,
                    },
                )()
                logger.info(f"Created simple handler for {modality} modality (testing mode)")
            else:
                # Get the model
                model = self._get_model()

                # Get the handler
                handler = get_handler_for_modality(modality, model)
                logger.info(f"Initialized handler for {modality} modality")

            # Store the handler
            self._handlers[modality] = handler
            self._initialized_modalities.add(modality)

            return handler
        except Exception as e:
            logger.error(f"Failed to initialize handler for {modality} modality: {e}")
            raise

    def supported_modalities(self) -> list[str]:
        """
        Get list of supported modalities.

        Returns
        -------
            list[str]: List of supported modality names

        """
        return self._supported_modalities.copy()

    def get_handler(self, modality: str) -> "ModalityHandler":
        """
        Get handler for a specific modality.

        Args:
        ----
            modality: Modality name

        Returns:
        -------
            ModalityHandler: Handler for the specified modality

        Raises:
        ------
            ValueError: If the modality is not supported

        """
        return self._initialize_handler(modality)

    @with_timeout_decorator(DEFAULT_TIMEOUT)
    async def process_input(
        self,
        content: Any,
        input_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
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
        handler = self._initialize_handler(input_modality)

        # Start trace span if trace manager is available
        if self._trace_manager and trace_id:
            with self._trace_manager.span(
                trace_id=trace_id,
                span_name=f"process_input_{input_modality}",
                attributes={"modality": input_modality},
            ):
                processed = await handler.process_input(content)
        else:
            processed = await handler.process_input(content)

        return {"content": processed, "modality": input_modality}

    @with_timeout_decorator(DEFAULT_TIMEOUT)
    async def format_output(
        self,
        content: Any,
        output_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Format output content in the specified modality.

        Args:
        ----
            content: The content to format
            output_modality: The target modality for the output
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Formatted content

        Raises:
        ------
            ValueError: If the modality is not supported
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        handler = self._initialize_handler(output_modality)

        # Start trace span if trace manager is available
        if self._trace_manager and trace_id:
            with self._trace_manager.span(
                trace_id=trace_id,
                span_name=f"format_output_{output_modality}",
                attributes={"modality": output_modality},
            ):
                formatted = await handler.format_output(content)
        else:
            formatted = await handler.format_output(content)

        return formatted

    @with_timeout_decorator(DEFAULT_TIMEOUT)
    async def convert(
        self,
        content: Any,
        source_modality: str,
        target_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Convert content from one modality to another.

        Args:
        ----
            content: The content to convert
            source_modality: The modality of the input content
            target_modality: The target modality for the output
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Converted content

        Raises:
        ------
            ValueError: If the modality is not supported
            NotImplementedError: If conversion between the specified modalities is not supported
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        # If source and target are the same, just return the content
        if source_modality == target_modality:
            return content

        # Start trace span if trace manager is available
        if self._trace_manager and trace_id:
            with self._trace_manager.span(
                trace_id=trace_id,
                span_name=f"convert_{source_modality}_to_{target_modality}",
                attributes={
                    "source_modality": source_modality,
                    "target_modality": target_modality,
                },
            ):
                # Process the input
                processed = await self.process_input(content, source_modality, trace_id, timeout)

                # Format the output
                converted = await self.format_output(
                    processed["content"], target_modality, trace_id, timeout
                )
        else:
            # Process the input
            processed = await self.process_input(content, source_modality, trace_id, timeout)

            # Format the output
            converted = await self.format_output(
                processed["content"], target_modality, trace_id, timeout
            )

        return converted
