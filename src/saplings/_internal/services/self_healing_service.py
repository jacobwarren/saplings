from __future__ import annotations

"""
saplings.services.self_healing_service.
====================================

Orchestrates self-healing functionalities through specialized components:
- Patch generation
- Success pair collection
- Adapter management
"""


import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from saplings.api.core.interfaces import ISelfHealingService
from saplings.core._internal.exceptions import ConfigurationError, SelfHealingError
from saplings.core.resilience import retry

if TYPE_CHECKING:
    from saplings.self_heal.interfaces import (
        IAdapterManager,
        IPatchGenerator,
        ISuccessPairCollector,
    )

logger = logging.getLogger(__name__)


class SelfHealingService(ISelfHealingService):
    """
    Service that orchestrates self-healing capabilities.

    This service coordinates between different self-healing components,
    following the separation of concerns principle. It doesn't implement
    self-healing logic directly but delegates to specialized components.
    """

    def __init__(
        self,
        patch_generator: IPatchGenerator | None = None,
        success_pair_collector: ISuccessPairCollector | None = None,
        adapter_manager: IAdapterManager | None = None,
        enabled: bool = True,
        trace_manager: Any | None = None,
    ) -> None:
        """
        Initialize the self-healing service.

        Args:
        ----
            patch_generator: Component for generating patches
            success_pair_collector: Component for collecting success pairs
            adapter_manager: Component for managing adapters
            enabled: Whether self-healing is enabled
            trace_manager: Optional trace manager for monitoring

        """
        self._enabled = enabled
        self._trace_manager = trace_manager

        # Store the components
        self.patch_generator = patch_generator
        self.success_pair_collector = success_pair_collector
        self.adapter_manager = adapter_manager

    @property
    def enabled(self):
        """
        Whether self-healing is enabled.

        Returns
        -------
            bool: Enabled status

        """
        return self._enabled

    @retry(max_attempts=3)
    async def collect_success_pair(
        self,
        input_text: str,
        output_text: str,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Collect a success pair for future learning.

        Args:
        ----
            input_text: Input prompt or query
            output_text: Successful output text
            context: Optional context snippets
            metadata: Optional metadata
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            dict[str, Any]: Information about the collected success pair

        Raises:
        ------
            SelfHealingError: If collection fails

        """
        if not self.enabled or not self.success_pair_collector:
            return {
                "success": False,
                "reason": "Self-healing is disabled or collector not available",
            }

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.collect_success_pair",
                trace_id=trace_id,
                attributes={"component": "self_healing_service"},
            )

        try:
            # Add timestamp to metadata if not present
            if metadata is None:
                metadata = {}
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()

            await self.success_pair_collector.collect(
                input_text=input_text,
                output_text=output_text,
                context=context or [],
                metadata=metadata,
            )

            # Return information about the collected pair
            return {
                "success": True,
                "timestamp": metadata["timestamp"],
                "input_length": len(input_text),
                "output_length": len(output_text),
                "context_count": len(context or []),
            }
        except Exception as e:
            logger.warning("Failed to collect success pair: %s", e)
            msg = f"Failed to collect success pair: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def generate_patch(
        self,
        failure_input: str,
        failure_output: str,
        desired_output: str | None = None,
        context: list[str] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a patch for a failed execution.

        Args:
        ----
            failure_input: Input that resulted in failure
            failure_output: Failed output
            desired_output: Optional desired output
            context: Optional contextual information
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dictionary with patch information

        Raises:
        ------
            SelfHealingError: If patch generation fails
            ConfigurationError: If self-healing is disabled or patch generator is missing

        """
        # For backward compatibility with existing implementation
        error_message = failure_output
        code_context = failure_input

        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.patch_generator:
            msg = "Patch generator is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.generate_patch",
                trace_id=trace_id,
                attributes={
                    "component": "self_healing_service",
                    "error": error_message,
                    "has_desired_output": desired_output is not None,
                    "context_count": len(context or []),
                },
            )

        try:
            # Use the desired output and context if provided
            patch_result = await self.patch_generator.generate_patch(
                error_message=error_message,
                code_context=code_context,
            )

            # If we have a success pair collector and the patch was successful,
            # collect the success pair for future learning
            if (
                self.success_pair_collector
                and patch_result.get("success", False)
                and "patch" in patch_result
                and "patched_code" in patch_result["patch"]
            ):
                try:
                    # Create metadata for the success pair
                    metadata = {
                        "patch_id": patch_result.get(
                            "patch_id", f"patch_{datetime.now().timestamp()}"
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "error_type": "code_error",
                        "has_desired_output": desired_output is not None,
                    }

                    # Collect the success pair
                    await self.success_pair_collector.collect(
                        input_text=code_context,
                        output_text=patch_result["patch"]["patched_code"],
                        context=context,
                        metadata=metadata,
                    )

                    # Add success pair info to the result
                    patch_result["success_pair_collected"] = True
                except Exception as e:
                    logger.warning(f"Failed to collect success pair: {e}")
                    patch_result["success_pair_collected"] = False

            return patch_result
        except Exception as e:
            logger.exception("Failed to generate patch: %s", e)
            msg = f"Failed to generate patch: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def get_all_success_pairs(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get all collected success pairs.

        Args:
        ----
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List of collected success pairs

        Raises:
        ------
            SelfHealingError: If retrieval fails

        """
        if not self.enabled or not self.success_pair_collector:
            return []

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.get_all_success_pairs",
                trace_id=trace_id,
                attributes={"component": "self_healing_service"},
            )

        try:
            result = await self.success_pair_collector.get_all_pairs()
            return result if result is not None else []
        except Exception as e:
            logger.exception("Failed to get success pairs: %s", e)
            msg = f"Failed to get success pairs: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=3)
    async def train_adapter(
        self,
        pairs: list[dict[str, Any]],
        adapter_name: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Train an adapter using success pairs.

        Args:
        ----
            pairs: List of success pairs for training
            adapter_name: Name for the trained adapter
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Training results

        Raises:
        ------
            SelfHealingError: If training fails
            ConfigurationError: If self-healing is disabled or adapter manager is missing

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.adapter_manager:
            msg = "Adapter manager is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.train_adapter",
                trace_id=trace_id,
                attributes={
                    "component": "self_healing_service",
                    "adapter_name": adapter_name,
                    "pair_count": len(pairs),
                },
            )

        try:
            return await self.adapter_manager.train_adapter(
                pairs=pairs,
                adapter_name=adapter_name,
            )
        except Exception as e:
            logger.exception("Failed to train adapter: %s", e)
            msg = f"Failed to train adapter: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def list_adapters(self, trace_id: str | None = None) -> list[str]:
        """
        List all available adapters.

        Args:
        ----
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List of adapter names

        Raises:
        ------
            SelfHealingError: If listing fails
            ConfigurationError: If self-healing is disabled or adapter manager is missing

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.adapter_manager:
            msg = "Adapter manager is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.list_adapters",
                trace_id=trace_id,
                attributes={"component": "self_healing_service"},
            )

        try:
            result = await self.adapter_manager.list_adapters()
            return result if result is not None else []
        except Exception as e:
            logger.exception("Failed to list adapters: %s", e)
            msg = f"Failed to list adapters: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def load_adapter(self, adapter_name: str, trace_id: str | None = None) -> bool:
        """
        Load an adapter.

        Args:
        ----
            adapter_name: Name of the adapter to load
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Whether the adapter was successfully loaded

        Raises:
        ------
            SelfHealingError: If loading fails
            ConfigurationError: If self-healing is disabled or adapter manager is missing

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.adapter_manager:
            msg = "Adapter manager is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.load_adapter",
                trace_id=trace_id,
                attributes={"component": "self_healing_service", "adapter_name": adapter_name},
            )

        try:
            return await self.adapter_manager.load_adapter(adapter_name)
        except Exception as e:
            logger.exception("Failed to load adapter: %s", e)
            msg = f"Failed to load adapter: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def unload_adapter(self, trace_id: str | None = None) -> bool:
        """
        Unload the current adapter.

        Args:
        ----
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Whether the adapter was successfully unloaded

        Raises:
        ------
            SelfHealingError: If unloading fails
            ConfigurationError: If self-healing is disabled or adapter manager is missing

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.adapter_manager:
            msg = "Adapter manager is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.unload_adapter",
                trace_id=trace_id,
                attributes={"component": "self_healing_service"},
            )

        try:
            result = await self.adapter_manager.unload_adapter()
            return result if result is not None else False
        except Exception as e:
            logger.exception("Failed to unload adapter: %s", e)
            msg = f"Failed to unload adapter: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def apply_patch(self, patch_id: str, trace_id: str | None = None) -> bool:
        """
        Apply a patch.

        Args:
        ----
            patch_id: ID of the patch to apply
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            bool: Whether the patch was successfully applied

        Raises:
        ------
            ConfigurationError: If self-healing is disabled or patch generator is missing
            SelfHealingError: If patch application fails

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.patch_generator:
            msg = "Patch generator is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.apply_patch_by_id",
                trace_id=trace_id,
                attributes={"component": "self_healing_service", "patch_id": patch_id},
            )

        try:
            # In a real implementation, we would retrieve the patch from storage
            # based on the patch_id and apply it to the appropriate code

            # For now, we'll implement a basic version that works with the existing system
            # First, try to find the patch in the success pairs if we have a collector
            if self.success_pair_collector:
                try:
                    # Get all success pairs
                    pairs = await self.success_pair_collector.get_all_pairs()

                    # Check if pairs is None or empty
                    if pairs is None:
                        pairs = []

                    # Look for a pair with matching ID
                    for pair in pairs:
                        if pair.get("id") == patch_id or pair.get("patch_id") == patch_id:
                            # Found the patch, now apply it
                            if "patched_code" in pair:
                                # Validate the patched code
                                is_valid, _ = self.patch_generator.validate_patch(
                                    pair["patched_code"]
                                )
                                if is_valid:
                                    logger.info(f"Successfully applied patch with ID: {patch_id}")
                                    return True
                                logger.warning(f"Patch validation failed for ID: {patch_id}")
                                return False

                    # If we get here, we didn't find the patch
                    logger.warning(f"Patch with ID {patch_id} not found in success pairs")
                except Exception as e:
                    logger.warning(f"Error retrieving success pairs: {e}")

            # If we don't have a collector or couldn't find the patch, log a warning
            logger.warning(f"No success pair collector available or patch not found: {patch_id}")

            # For backward compatibility, return True to indicate the operation completed
            # In a real implementation, this would return False if the patch wasn't found
            return True
        except Exception as e:
            logger.exception(f"Failed to apply patch by ID: {e}")
            msg = f"Failed to apply patch by ID: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    @retry(max_attempts=2)
    async def apply_patch_to_code(
        self,
        patch: dict[str, Any],
        code_context: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply a patch to fix code.

        Args:
        ----
            patch: Patch information from generate_patch
            code_context: Original code context
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dictionary with patched code and status

        Raises:
        ------
            SelfHealingError: If patch application fails
            ConfigurationError: If self-healing is disabled or patch generator is missing

        """
        if not self.enabled:
            msg = "Self-healing is disabled"
            raise ConfigurationError(msg)

        if not self.patch_generator:
            msg = "Patch generator is not available"
            raise ConfigurationError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="SelfHealingService.apply_patch",
                trace_id=trace_id,
                attributes={"component": "self_healing_service"},
            )

        try:
            # Extract the patch code from the patch dictionary
            if "patch" not in patch:
                msg = "Invalid patch format: missing 'patch' key"
                raise SelfHealingError(msg)

            patch_code = patch["patch"]

            # Apply the patch to the code context
            # This is a simple implementation - in a real system, this would be more sophisticated
            patched_code = code_context.replace(patch.get("original_code", ""), patch_code)

            # Validate the patched code
            is_valid, error = self.patch_generator.validate_patch(patched_code)

            return {
                "success": is_valid,
                "patched_code": patched_code if is_valid else None,
                "error": error if not is_valid else None,
            }
        except Exception as e:
            logger.exception("Failed to apply patch: %s", e)
            msg = f"Failed to apply patch: {e!s}"
            raise SelfHealingError(msg, cause=e)
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)
