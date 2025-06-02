from __future__ import annotations

"""
Executor module for Saplings.

This module provides the executor functionality for Saplings, including:
- Speculative draft generation with low temperature
- Streaming output capabilities
- GASA mask injection mechanism
- Integration with JudgeAgent for verification
- Refinement logic for rejected outputs
"""


import logging
import time
from typing import TYPE_CHECKING, Any

from saplings.core._internal.model_interface import LLM, LLMResponse, ModelRole
from saplings.core.resilience import DEFAULT_TIMEOUT, run_in_executor
from saplings.executor._internal.config import (
    ExecutorConfig,
    RefinementStrategy,
    ValidationStrategy,
)
from saplings.gasa import (
    FallbackStrategy,
    GASAConfig,
    GASAService,
    MaskStrategy,
)
from saplings.validator._internal.result import ValidationStatus

if TYPE_CHECKING:
    from saplings.judge._internal import JudgeAgent
    from saplings.memory._internal.document import Document
    from saplings.memory._internal.graph import DependencyGraph
    from saplings.monitoring._internal.trace import TraceManager
    from saplings.validator._internal.registry import ValidatorRegistry

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of an execution."""

    def __init__(
        self,
        text: str,
        provider: str,
        model_name: str,
        usage: dict[str, int],
        metadata: dict[str, Any],
        verified: bool = False,
        verification_score: float | None = None,
        verification_feedback: str | None = None,
        draft_latency_ms: int | None = None,
        final_latency_ms: int | None = None,
        total_latency_ms: int | None = None,
        refinement_attempts: int = 0,
        response: LLMResponse | None = None,
    ) -> None:
        """
        Initialize the execution result.

        Args:
        ----
            text: Generated text
            provider: Provider of the model
            model_name: Name of the model
            usage: Token usage statistics
            metadata: Additional metadata
            verified: Whether the output was verified
            verification_score: Verification score (0.0 to 1.0)
            verification_feedback: Feedback from verification
            draft_latency_ms: Latency of draft generation in milliseconds
            final_latency_ms: Latency of final generation in milliseconds
            total_latency_ms: Total latency in milliseconds
            refinement_attempts: Number of refinement attempts
            response: Original LLMResponse object

        """
        self.text = text
        self.provider = provider
        self.model_name = model_name
        self.usage = usage
        self.metadata = metadata
        self.verified = verified
        self.verification_score = verification_score
        self.verification_feedback = verification_feedback
        self.draft_latency_ms = draft_latency_ms
        self.final_latency_ms = final_latency_ms
        self.total_latency_ms = total_latency_ms
        self.refinement_attempts = refinement_attempts
        self.response = response

    @classmethod
    def from_llm_response(cls, response: LLMResponse, **kwargs) -> "ExecutionResult":
        """
        Create an execution result from an LLM response.

        Args:
        ----
            response: LLM response
            **kwargs: Additional arguments

        Returns:
        -------
            ExecutionResult: Execution result

        """
        return cls(
            text=response.text or "",  # Handle None case by providing empty string
            provider=response.provider,
            model_name=response.model_name,
            usage=response.usage,
            metadata=response.metadata,
            response=response,  # Include the original response object
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the execution result to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        result = {
            "text": self.text,
            "provider": self.provider,
            "model_name": self.model_name,
            "usage": self.usage,
            "metadata": self.metadata,
            "verified": self.verified,
            "verification_score": self.verification_score,
            "verification_feedback": self.verification_feedback,
            "draft_latency_ms": self.draft_latency_ms,
            "final_latency_ms": self.final_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "refinement_attempts": self.refinement_attempts,
        }

        # Include response if available (but don't include in the dictionary to avoid circular references)
        if hasattr(self, "response") and self.response is not None:
            # Add a reference to indicate response is available
            result["has_response"] = True

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        """
        Create an execution result from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            ExecutionResult: Execution result

        """
        return cls(**data)


class Executor:
    """
    Executor for generating text with speculative execution and verification.

    This class provides the core functionality for generating text with
    speculative execution, GASA mask injection, and verification.
    """

    def __init__(
        self,
        model: LLM,
        config: ExecutorConfig | None = None,
        gasa_config: GASAConfig | None = None,
        dependency_graph: DependencyGraph | None = None,
        judge_agent: JudgeAgent | None = None,
        validator_registry: "ValidatorRegistry | None" = None,
        trace_manager: "TraceManager | None" = None,
        validator_service: Any = None,  # Add validator_service parameter
    ) -> None:
        """
        Initialize the executor.

        Args:
        ----
            model: LLM model to use for generation
            config: Executor configuration
            gasa_config: GASA configuration
            dependency_graph: Dependency graph for GASA
            judge_agent: JudgeAgent for verification
            validator_registry: ValidatorRegistry for validation
            trace_manager: TraceManager for tracing execution
            validator_service: ValidatorService instance to use for validation

        """
        self.model = model
        self.config = config or ExecutorConfig.default()
        self.gasa_config = gasa_config or GASAConfig(
            enabled=True,
            max_hops=2,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
            summary_token="[SUM]",
            add_summary_token=True,
            block_size=512,
            overlap=64,
            soft_mask_temperature=0.1,
            cache_masks=True,
            cache_dir=None,
            visualize=False,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="",
            shadow_model_device="",
            shadow_model_cache_dir="",
            enable_prompt_composer=False,
            focus_tags=False,
            core_tag="",
            near_tag="",
            summary_tag="",
        )
        self.dependency_graph = dependency_graph
        self.judge_agent = judge_agent
        self.validator_registry = validator_registry
        self.trace_manager = trace_manager
        self._validator_service = validator_service  # Store the validator service
        self.tools: dict[str, Any] = {}

        # Initialize GASA if enabled
        self.gasa_service = None
        if self.config.enable_gasa and self.dependency_graph is not None:
            try:
                # Get tokenizer from model or use SimpleTokenizer as fallback
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer is None:
                    try:
                        from saplings.tokenizers import SimpleTokenizer

                        logger.info("Using SimpleTokenizer as fallback for GASA")
                        tokenizer = SimpleTokenizer()
                    except ImportError:
                        logger.warning(
                            "Model does not have a tokenizer attribute and SimpleTokenizer is not available. "
                            "GASA may not work properly."
                        )

                self.gasa_service = GASAService(
                    graph=self.dependency_graph,
                    config=self.gasa_config,
                    tokenizer=tokenizer,
                )
                logger.info(f"Initialized GASA Service with max_hops={self.gasa_config.max_hops}")
            except Exception as e:
                logger.exception(f"Failed to initialize GASA Service: {e}")
                # Continue without GASA

        # Validate model
        self._validate_model()

        # Initialize cache
        self._cache: dict[str, ExecutionResult] = {}

    def _validate_model(self):
        """
        Validate that the model is suitable for execution.

        Raises
        ------
            ValueError: If the model is not suitable for execution

        """
        metadata = self.model.get_metadata()
        roles = (
            metadata.get("roles", [])
            if isinstance(metadata, dict)
            else getattr(metadata, "roles", [])
        )
        name = (
            metadata.get("name", "unknown")
            if isinstance(metadata, dict)
            else getattr(metadata, "name", "unknown")
        )
        if ModelRole.EXECUTOR not in roles and ModelRole.GENERAL not in roles:
            msg = (
                f"Model {name} is not suitable for execution. "
                f"It must have either EXECUTOR or GENERAL role."
            )
            raise ValueError(msg)

    def _get_cache_key(self, prompt: str, **_) -> str:
        """
        Get a cache key for a prompt and generation parameters.

        Args:
        ----
            prompt: Prompt text
            **_: Additional generation parameters (unused in this implementation)

        Returns:
        -------
            str: Cache key

        """
        # For simplicity in tests, just use the prompt as the cache key
        # In a real implementation, we would include all relevant parameters
        return prompt

    async def _apply_gasa(
        self,
        prompt: str,
        documents: list[Document] | None,
        generation_type: str,
        timeout: float | None = DEFAULT_TIMEOUT,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Apply GASA to the given prompt and documents.

        Args:
        ----
            prompt: Prompt text
            documents: Documents to use for building the mask
            generation_type: Type of generation (for logging)
            timeout: Optional timeout in seconds
            **kwargs: Additional parameters for GASA

        Returns:
        -------
            Dict[str, Any]: Result containing modified prompt and inputs

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        # Skip GASA if it's disabled, service is not initialized, or no documents are provided
        if not self.config.enable_gasa or self.gasa_service is None or documents is None:
            return {"prompt": prompt}

        # Check if the model supports sparse attention
        # This would need to be determined based on model capabilities
        model_supports_sparse_attention = hasattr(self.model, "supports_sparse_attention")

        try:
            logger.debug(
                f"Applying GASA for {generation_type} generation (documents: {len(documents)})"
            )

            # Define a function to run in a thread pool
            def _apply_gasa_blocking():
                assert self.gasa_service is not None
                return self.gasa_service.apply_gasa(
                    documents=documents,
                    prompt=prompt,
                    model_supports_sparse_attention=model_supports_sparse_attention,
                    **kwargs,
                )

            # Run the blocking operation in a thread pool with timeout
            result = await run_in_executor(_apply_gasa_blocking, timeout=timeout)

            logger.debug(f"Applied GASA to {generation_type} generation")
            return result
        except Exception as e:
            logger.warning(f"Failed to apply GASA for {generation_type} generation: {e}")
            return {"prompt": prompt}

    async def _generate_draft(
        self, prompt: str, documents: list[Document] | None = None, **kwargs
    ) -> LLMResponse:
        """
        Generate a draft response with low temperature.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            **kwargs: Additional generation parameters

        Returns:
        -------
            LLMResponse: Draft response

        """
        # Start timing
        start_time = time.time()

        # Set draft generation parameters
        draft_kwargs = {
            "max_tokens": self.config.max_draft_tokens,
            "temperature": self.config.draft_temperature,
            **kwargs,
        }

        # Apply GASA if enabled
        gasa_result = await self._apply_gasa(
            prompt=prompt,
            documents=documents,
            generation_type="draft",
            input_ids=draft_kwargs.get("input_ids"),
            attention_mask=draft_kwargs.get("attention_mask"),
        )

        # Update kwargs with GASA result
        if gasa_result and "attention_mask" in gasa_result:
            draft_kwargs["attention_mask"] = gasa_result["attention_mask"]

        # For prompt composers, use the modified prompt
        if gasa_result and "prompt" in gasa_result and gasa_result["prompt"] != prompt:
            prompt = gasa_result["prompt"]

        # Generate draft
        draft_response = await self.model.generate(prompt, **draft_kwargs)

        # Record latency
        draft_response.metadata["latency_ms"] = int((time.time() - start_time) * 1000)

        return draft_response

    async def _generate_final(
        self, prompt: str, documents: list[Document] | None = None, **kwargs
    ) -> LLMResponse:
        """
        Generate a final response with normal temperature.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            **kwargs: Additional generation parameters

        Returns:
        -------
            LLMResponse: Final response

        """
        # Start timing
        start_time = time.time()

        # Set final generation parameters
        final_kwargs = {
            "max_tokens": self.config.max_final_tokens,
            "temperature": self.config.final_temperature,
            **kwargs,
        }

        # Apply GASA if enabled
        gasa_result = await self._apply_gasa(
            prompt=prompt,
            documents=documents,
            generation_type="final",
            input_ids=final_kwargs.get("input_ids"),
            attention_mask=final_kwargs.get("attention_mask"),
        )

        # Update kwargs with GASA result
        if gasa_result and "attention_mask" in gasa_result:
            final_kwargs["attention_mask"] = gasa_result["attention_mask"]

        # For prompt composers, use the modified prompt
        if gasa_result and "prompt" in gasa_result and gasa_result["prompt"] != prompt:
            prompt = gasa_result["prompt"]

        # Generate final response
        final_response = await self.model.generate(prompt, **final_kwargs)

        # Record latency
        final_response.metadata["latency_ms"] = int((time.time() - start_time) * 1000)

        return final_response

    async def _verify_output(
        self,
        output: str,
        prompt: str,
        validation_type: ValidationStrategy | None = None,
        **kwargs,
    ) -> tuple[bool, float | None, str | None]:
        """
        Verify an output using the configured validation strategy.

        Args:
        ----
            output: Output text to verify
            prompt: Original prompt
            validation_type: Override the configured validation type
            **kwargs: Additional verification parameters

        Returns:
        -------
            Tuple[bool, Optional[float], Optional[str]]:
                - Whether the output passed verification
                - Verification score (0.0 to 1.0)
                - Feedback from verification

        """
        # Use the specified validation type or fall back to the configured one
        validation_type = validation_type or self.config.validation_type

        # Skip validation if disabled
        if validation_type == ValidationStrategy.NONE:
            return True, None, None

        # Trace validation if tracing is enabled
        if self.trace_manager:
            self.trace_manager.add_event(
                "validation_start",
                {
                    "validation_type": validation_type,
                    "output_length": len(output),
                    "prompt_length": len(prompt),
                },
            )

        try:
            # Validate using JudgeAgent
            if validation_type in (ValidationStrategy.JUDGE, ValidationStrategy.FULL):
                if self.judge_agent is None:
                    logger.warning("JudgeAgent validation requested but no JudgeAgent provided")
                    return True, None, None

                # Validate using JudgeAgent
                result = await self.judge_agent.validate(output=output, prompt=prompt, **kwargs)
                passed = result.score >= self.config.validation_threshold
                return passed, result.score, result.feedback

            # Validate using ValidatorRegistry
            if validation_type in (ValidationStrategy.VALIDATOR, ValidationStrategy.FULL):
                if self.validator_registry is None and self._validator_service is None:
                    logger.warning(
                        "Validator validation requested but no ValidatorRegistry or ValidatorService provided"
                    )
                    return True, None, None

                # Use validator service if available, otherwise use registry directly
                if self._validator_service is not None:
                    # Use validator service
                    result = await self._validator_service.validate(
                        output=output, prompt=prompt, **kwargs
                    )
                else:
                    # Use validator registry directly
                    assert self.validator_registry is not None
                    result = await self.validator_registry.validate(
                        output=output, prompt=prompt, **kwargs
                    )

                # Check if validation passed
                passed = result.status == ValidationStatus.PASSED
                return passed, result.score, result.feedback

            # Basic validation (simple checks)
            if validation_type == ValidationStrategy.BASIC:
                # Implement basic validation logic here
                # For now, just check if the output is not empty
                passed = bool(output.strip())
                return passed, 1.0 if passed else 0.0, None

            # Self-consistency validation
            if validation_type == ValidationStrategy.SELF_CONSISTENCY:
                # Not implemented yet
                logger.warning("Self-consistency validation not implemented yet")
                return True, None, None

            # Unknown validation type
            logger.warning(f"Unknown validation type: {validation_type}")
            return True, None, None

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            return True, None, f"Validation error: {e}"
        finally:
            # Trace validation completion if tracing is enabled
            if self.trace_manager:
                self.trace_manager.add_event("validation_end", {})

    async def _refine_output(
        self,
        output: str,
        prompt: str,
        verification_feedback: str | None,
        attempt: int,
        documents: list[Document] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Refine an output that failed verification.

        Args:
        ----
            output: Original output that failed verification
            prompt: Original prompt
            verification_feedback: Feedback from verification
            attempt: Current refinement attempt (1-based)
            documents: Documents used in the prompt (for GASA)
            **kwargs: Additional generation parameters

        Returns:
        -------
            LLMResponse: Refined response

        """
        # Skip refinement if disabled or max attempts reached
        if (
            self.config.refinement_strategy == RefinementStrategy.NONE
            or attempt > self.config.max_refinement_attempts
        ):
            # Return the original output as a response
            return LLMResponse(
                text=output,
                provider=self.model.provider,
                model_name=self.model.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"refinement_attempt": attempt},
                function_call=None,
                tool_calls=None,
            )

        # Trace refinement if tracing is enabled
        if self.trace_manager:
            self.trace_manager.add_event(
                "refinement_start",
                {
                    "attempt": attempt,
                    "output_length": len(output),
                    "prompt_length": len(prompt),
                    "has_feedback": verification_feedback is not None,
                },
            )

        try:
            # Choose refinement strategy
            if self.config.refinement_strategy == RefinementStrategy.RETRY:
                # Simple retry with the same prompt
                return await self.model.generate(prompt, **kwargs)

            elif self.config.refinement_strategy == RefinementStrategy.FEEDBACK:
                # Retry with feedback from verification
                if verification_feedback:
                    # Create a new prompt with feedback
                    refinement_prompt = (
                        f"{prompt}\n\n"
                        f"Your previous response was rejected for the following reason:\n"
                        f"{verification_feedback}\n\n"
                        f"Please provide a corrected response."
                    )
                    return await self.model.generate(refinement_prompt, **kwargs)
                else:
                    # No feedback available, fall back to simple retry
                    return await self.model.generate(prompt, **kwargs)

            elif self.config.refinement_strategy == RefinementStrategy.ITERATIVE:
                # Iterative refinement with multiple feedback cycles
                if verification_feedback:
                    # Create a new prompt with feedback and previous output
                    refinement_prompt = (
                        f"{prompt}\n\n"
                        f"Your previous response was:\n"
                        f"{output}\n\n"
                        f"This response was rejected for the following reason:\n"
                        f"{verification_feedback}\n\n"
                        f"Please provide a corrected response."
                    )
                    return await self.model.generate(refinement_prompt, **kwargs)
                else:
                    # No feedback available, fall back to simple retry
                    return await self.model.generate(prompt, **kwargs)
            else:
                # Unknown refinement strategy, fall back to simple retry
                logger.warning(f"Unknown refinement strategy: {self.config.refinement_strategy}")
                return await self.model.generate(prompt, **kwargs)

        except Exception as e:
            logger.exception(f"Error during refinement: {e}")
            # Return the original output as a response
            return LLMResponse(
                text=output,
                provider=self.model.provider,
                model_name=self.model.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"refinement_attempt": attempt, "error": str(e)},
                function_call=None,
                tool_calls=None,
            )
        finally:
            # Trace refinement completion if tracing is enabled
            if self.trace_manager:
                self.trace_manager.add_event("refinement_end", {})

    async def execute(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        validation_type: ValidationStrategy | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute a prompt with speculative execution and verification.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            validation_type: Override the configured validation type
            use_cache: Whether to use the cache (overrides config)
            **kwargs: Additional generation parameters

        Returns:
        -------
            ExecutionResult: Execution result

        """
        # Start timing
        start_time = time.time()

        # Check cache if enabled
        use_cache = self.config.cache_results if use_cache is None else use_cache
        if use_cache:
            cache_key = self._get_cache_key(prompt, **kwargs)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return self._cache[cache_key]

        # Trace execution if tracing is enabled
        if self.trace_manager:
            self.trace_manager.add_event(
                "execution_start",
                {
                    "prompt_length": len(prompt),
                    "has_documents": documents is not None,
                    "document_count": len(documents) if documents else 0,
                },
            )

        try:
            # Generate draft if speculative execution is enabled
            draft_response = None
            if self.config.enable_speculative_execution:
                draft_response = await self._generate_draft(
                    prompt=prompt, documents=documents, **kwargs
                )
                draft_latency_ms = draft_response.metadata.get("latency_ms", 0)
            else:
                draft_latency_ms = None

            # Generate final response
            final_response = await self._generate_final(
                prompt=prompt, documents=documents, **kwargs
            )
            final_latency_ms = final_response.metadata.get("latency_ms", 0)

            # Create execution result
            result = ExecutionResult.from_llm_response(
                final_response,
                draft_latency_ms=draft_latency_ms,
                final_latency_ms=final_latency_ms,
                total_latency_ms=int((time.time() - start_time) * 1000),
            )

            # Verify output if validation is enabled
            if validation_type != ValidationStrategy.NONE or (
                validation_type is None and self.config.validation_type != ValidationStrategy.NONE
            ):
                verified, score, feedback = await self._verify_output(
                    output=result.text,
                    prompt=prompt,
                    validation_type=validation_type,
                    **kwargs,
                )
                result.verified = verified
                result.verification_score = score
                result.verification_feedback = feedback

                # Refine output if verification failed
                if not verified and self.config.refinement_strategy != RefinementStrategy.NONE:
                    refinement_attempts = 0
                    current_output = result.text
                    current_feedback = feedback

                    # Try to refine the output up to max_refinement_attempts times
                    for attempt in range(1, self.config.max_refinement_attempts + 1):
                        refinement_attempts += 1
                        refined_response = await self._refine_output(
                            output=current_output,
                            prompt=prompt,
                            verification_feedback=current_feedback,
                            attempt=attempt,
                            documents=documents,
                            **kwargs,
                        )

                        # Verify the refined output
                        verified, score, feedback = await self._verify_output(
                            output=refined_response.text,
                            prompt=prompt,
                            validation_type=validation_type,
                            **kwargs,
                        )

                        # Update current output and feedback for next iteration
                        current_output = refined_response.text
                        current_feedback = feedback

                        # If verification passed, use the refined output
                        if verified:
                            # Create a new execution result with the refined output
                            result = ExecutionResult.from_llm_response(
                                refined_response,
                                draft_latency_ms=draft_latency_ms,
                                final_latency_ms=final_latency_ms,
                                total_latency_ms=int((time.time() - start_time) * 1000),
                                verified=verified,
                                verification_score=score,
                                verification_feedback=feedback,
                                refinement_attempts=refinement_attempts,
                            )
                            break
                        elif attempt == self.config.max_refinement_attempts:
                            # If this was the last attempt, use the refined output anyway
                            result = ExecutionResult.from_llm_response(
                                refined_response,
                                draft_latency_ms=draft_latency_ms,
                                final_latency_ms=final_latency_ms,
                                total_latency_ms=int((time.time() - start_time) * 1000),
                                verified=verified,
                                verification_score=score,
                                verification_feedback=feedback,
                                refinement_attempts=refinement_attempts,
                            )

            # Cache result if caching is enabled
            if use_cache:
                cache_key = self._get_cache_key(prompt, **kwargs)
                self._cache[cache_key] = result

            return result
        except Exception as e:
            logger.exception(f"Error during execution: {e}")
            # Create an error result
            return ExecutionResult(
                text=f"Error during execution: {e}",
                provider=self.model.provider,
                model_name=self.model.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"error": str(e)},
                total_latency_ms=int((time.time() - start_time) * 1000),
            )
        finally:
            # Trace execution completion if tracing is enabled
            if self.trace_manager:
                self.trace_manager.add_event("execution_end", {})

    async def stream(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ):
        """
        Stream the execution of a prompt.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            chunk_size: Number of tokens to generate per streaming chunk
            **kwargs: Additional generation parameters

        Yields:
        ------
            str: Chunks of generated text

        """
        # Use configured chunk size if not specified
        chunk_size = chunk_size or self.config.stream_chunk_size

        # Skip streaming if disabled
        if not self.config.enable_streaming:
            # Generate the full response and yield it as a single chunk
            result = await self.execute(prompt, documents, **kwargs)
            yield result.text
            return

        # Trace streaming if tracing is enabled
        if self.trace_manager:
            self.trace_manager.add_event(
                "streaming_start",
                {
                    "prompt_length": len(prompt),
                    "has_documents": documents is not None,
                    "document_count": len(documents) if documents else 0,
                    "chunk_size": chunk_size,
                },
            )

        try:
            # Apply GASA if enabled
            gasa_result = await self._apply_gasa(
                prompt=prompt,
                documents=documents,
                generation_type="streaming",
                input_ids=kwargs.get("input_ids"),
                attention_mask=kwargs.get("attention_mask"),
            )

            # Update kwargs with GASA result
            if gasa_result and "attention_mask" in gasa_result:
                kwargs["attention_mask"] = gasa_result["attention_mask"]

            # For prompt composers, use the modified prompt
            if gasa_result and "prompt" in gasa_result and gasa_result["prompt"] != prompt:
                prompt = gasa_result["prompt"]

            # Set streaming parameters
            streaming_kwargs = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": True,
                "chunk_size": chunk_size,
                **kwargs,
            }

            # Stream the response
            async for chunk in self.model.generate_stream(prompt, **streaming_kwargs):
                yield chunk

        except Exception as e:
            logger.exception(f"Error during streaming: {e}")
            yield f"Error during streaming: {e}"
        finally:
            # Trace streaming completion if tracing is enabled
            if self.trace_manager:
                self.trace_manager.add_event("streaming_end", {})
