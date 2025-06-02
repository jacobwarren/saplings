from __future__ import annotations

"""
saplings.services.model_initialization_service.
===========================================

Encapsulates model initialization and configuration:
- Model creation and URI parsing
- Parameter management
- Registry integration
- Model telemetry
"""


import logging
from typing import Any, Dict, Optional

from saplings.api.core.interfaces import IModelInitializationService
from saplings.core._internal.exceptions import ModelError, ProviderError, ResourceExhaustedError
from saplings.core._internal.model_interface import LLM, ModelMetadata
from saplings.core.resilience import CircuitBreaker, with_timeout

logger = logging.getLogger(__name__)


class ModelInitializationService(IModelInitializationService):
    """
    Service that handles model initialization and configuration.

    This service is responsible for model initialization, configuration management,
    and core model operations like metadata access.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        retry_config: Optional[Dict[str, Any]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        lazy_init: bool = True,
        **model_parameters,
    ) -> None:
        """
        Initialize the model initialization service.

        Args:
        ----
            provider: Provider name (e.g., "openai", "anthropic")
            model_name: Name of the model
            retry_config: Configuration for retry mechanism
            circuit_breaker_config: Configuration for circuit breaker
            lazy_init: Whether to initialize the model lazily (on first use)
            **model_parameters: Additional model parameters

        """
        # Set up resilience configuration
        self.retry_config = retry_config or {
            "max_attempts": 3,
            "initial_backoff": 1.0,
            "max_backoff": 30.0,
            "backoff_factor": 2.0,
            "jitter": True,
        }

        self.circuit_breaker_config = circuit_breaker_config or {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
        }

        # Create circuit breaker for model calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker_config["failure_threshold"],
            recovery_timeout=self.circuit_breaker_config["recovery_timeout"],
            expected_exceptions=[
                ProviderError,
                ResourceExhaustedError,
                ConnectionError,
                TimeoutError,
            ],
        )

        self.provider = provider
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model: Optional[LLM] = None
        self.lazy_init = lazy_init
        self._initialized = False

        # Initialize the model if not using lazy initialization
        if not self.lazy_init:
            self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the model.

        This method is called automatically on first use if lazy initialization is enabled,
        or immediately during construction if lazy initialization is disabled.

        Raises
        ------
            ModelError: If model initialization fails

        """
        # Skip if already initialized
        if self._initialized and self.model is not None:
            return

        try:
            # The model registry is handled automatically in LLM.create
            # to ensure that only one instance with the same configuration is created
            self.model = LLM.create(
                provider=self.provider,
                model_name=self.model_name,
                lazy_init=self.lazy_init,  # Pass lazy_init to the adapter
                **self.model_parameters,
            )
            self._initialized = True
            logger.info("Model initialized: %s/%s", self.provider, self.model_name)
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            msg = f"Failed to initialize model: {e!s}"
            raise ModelError(msg, cause=e)

    async def get_model(self, timeout: Optional[float] = None) -> LLM:
        """
        Get the LLM instance.

        If lazy initialization is enabled, this will initialize the model on first use.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            LLM: The initialized model

        Raises:
        ------
            ModelError: If the model initialization fails
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        # Initialize the model if it's not initialized yet and lazy initialization is enabled
        if self.model is None or not self._initialized:
            # Use timeout if provided
            if timeout is not None:

                async def _init_with_timeout():
                    self._init_model()
                    return self.model

                # Execute with timeout
                from saplings.core.resilience import with_timeout

                await with_timeout(
                    _init_with_timeout(), timeout=timeout, operation_name="model_initialization"
                )
            else:
                # Initialize without timeout
                self._init_model()

            # Check if initialization was successful
            if self.model is None:
                msg = "Model initialization failed."
                raise ModelError(msg)

        return self.model

    async def get_model_metadata(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get metadata about the model.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict: Model metadata

        Raises:
        ------
            ModelError: If metadata cannot be retrieved
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        try:
            # Get model first
            model = await self.get_model()

            # Define async function to get metadata with timeout
            async def _get_metadata():
                try:
                    metadata = model.get_metadata()
                    # Convert ModelMetadata to dict if needed
                    if metadata is not None:
                        if isinstance(metadata, ModelMetadata):
                            return metadata.model_dump()
                        if isinstance(metadata, dict):
                            return metadata
                    return metadata or {}
                except Exception as e:
                    # Wrap the original exception to provide context while preserving the trace
                    msg = f"Failed to get model metadata: {e!s}"
                    raise ModelError(msg, cause=e)

            # Execute with timeout
            result = (
                await with_timeout(
                    _get_metadata(), timeout=timeout, operation_name="get_model_metadata"
                )
                if timeout
                else await _get_metadata()
            )

            # Ensure we return a Dict[str, Any]
            if isinstance(result, ModelMetadata):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            # Convert any other type to dict
            return {"value": str(result)}
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to get model metadata: {e!s}"
                raise ModelError(msg, cause=e)
            raise

    def initialize_model(self, model_id: str) -> Any:
        """
        Initialize a model.

        This method implements the IModelInitializationService interface.
        For this implementation, we return the initialized LLM model.

        Args:
        ----
            model_id: The model identifier (not used in this implementation)

        Returns:
        -------
            The initialized LLM model

        Raises:
        ------
            ModelError: If model initialization fails

        """
        # Initialize the model if not already done
        if self.model is None or not self._initialized:
            self._init_model()

        return self.model
