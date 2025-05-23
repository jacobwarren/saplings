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
from typing import TYPE_CHECKING, Any, Callable

from saplings.core.model_adapter import LLM, LLMResponse, ModelRole
from saplings.core.resilience import DEFAULT_TIMEOUT, run_in_executor
from saplings.executor.config import ExecutorConfig, RefinementStrategy, VerificationStrategy
from saplings.gasa import (
    FallbackStrategy,
    GASAConfig,
    GASAService,
    MaskStrategy,
)
from saplings.validator.result import ValidationStatus

if TYPE_CHECKING:
    from saplings.judge import JudgeAgent
    from saplings.memory.document import Document
    from saplings.memory.graph import DependencyGraph
    from saplings.monitoring.trace import TraceManager
    from saplings.validator.registry import ValidatorRegistry

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
        verification_strategy: VerificationStrategy | None = None,
        **kwargs,
    ) -> tuple[bool, float | None, str | None]:
        """
        Verify an output using the configured verification strategy.

        Args:
        ----
            output: Output text to verify
            prompt: Original prompt
            verification_strategy: Override the configured verification strategy
            **kwargs: Additional verification parameters

        Returns:
        -------
            Tuple[bool, Optional[float], Optional[str]]:
                - Whether the output passed verification
                - Verification score (0.0 to 1.0)
                - Feedback from verification

        """
        strategy = verification_strategy or self.config.verification_strategy

        # No verification
        if strategy == VerificationStrategy.NONE:
            return True, None, None

        # Basic verification
        if strategy == VerificationStrategy.BASIC:
            # Simple length check
            if len(output.strip()) == 0:
                return False, 0.0, "Output is empty"

            # Check if output is too short
            if len(output.split()) < 5:
                return False, 0.3, "Output is too short"

            # Check if output is just repeating the prompt
            if output.strip() == prompt.strip():
                return False, 0.0, "Output is identical to prompt"

            # Basic verification passed
            return True, 1.0, None

        # Judge verification
        if strategy == VerificationStrategy.JUDGE:
            if self.judge_agent is None:
                logger.warning("JudgeAgent not provided, falling back to basic verification")
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

            # Use JudgeAgent to verify the output
            try:
                # Judge the output
                judge_result = await self.judge_agent.judge(output=output, prompt=prompt)

                # Get the verification result
                passed = judge_result.passed
                score = judge_result.overall_score

                # Format the feedback
                feedback = self.judge_agent.format_critique(judge_result)

                # Log the verification result
                if passed:
                    logger.info(f"Output passed verification with score {score:.2f}")
                else:
                    logger.warning(f"Output failed verification with score {score:.2f}")

                return passed, score, feedback
            except Exception as e:
                logger.exception(f"Error during JudgeAgent verification: {e}")
                logger.warning("Falling back to basic verification")
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

        # Validator verification
        if strategy == VerificationStrategy.VALIDATOR:
            if self.validator_registry is None:
                logger.warning("ValidatorRegistry not provided, falling back to basic verification")
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

            # Use ValidatorRegistry to verify the output
            try:
                # Get validator type from kwargs or use runtime validators by default
                validator_type = kwargs.get("validator_type")
                validator_ids = kwargs.get("validator_ids")

                # Validate the output
                assert self.validator_registry is not None
                validation_results = await self.validator_registry.validate(
                    output=output,
                    prompt=prompt,
                    validator_type=validator_type,
                    validator_ids=validator_ids,
                    **kwargs,
                )

                # Check if we have any results
                if not validation_results:
                    logger.warning(
                        "No validation results returned, falling back to basic verification"
                    )
                    return await self._verify_output(
                        output=output,
                        prompt=prompt,
                        verification_strategy=VerificationStrategy.BASIC,
                        **kwargs,
                    )

                # Calculate the overall score and status
                total_score = 0.0
                failed_validations = []

                for result in validation_results:
                    # Add to total score
                    score = result.metadata.get("score", 0.0)
                    total_score += score

                    # Check if validation failed
                    if result.status == ValidationStatus.FAILED:
                        failed_validations.append(result)

                # Calculate average score
                avg_score = total_score / len(validation_results) if validation_results else 0.0

                # Determine if validation passed
                passed = len(failed_validations) == 0

                # Format feedback
                if failed_validations:
                    feedback = "Validation failed:\n" + "\n".join(
                        [
                            f"- {result.validator_id}: {result.message}"
                            for result in failed_validations
                        ]
                    )
                else:
                    feedback = "All validations passed"

                # Log the verification result
                if passed:
                    logger.info(f"Output passed validation with score {avg_score:.2f}")
                else:
                    logger.warning(f"Output failed validation with score {avg_score:.2f}")

                return passed, avg_score, feedback
            except Exception as e:
                logger.exception(f"Error during ValidatorRegistry verification: {e}")
                logger.warning("Falling back to basic verification")
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

        # Full verification
        if strategy == VerificationStrategy.FULL:
            # Check if we have both JudgeAgent and ValidatorRegistry
            if self.judge_agent is None and self.validator_registry is None:
                logger.warning(
                    "Neither JudgeAgent nor ValidatorRegistry provided for FULL verification, falling back to BASIC"
                )
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

            # If we only have JudgeAgent, use that
            if self.judge_agent is not None and self.validator_registry is None:
                logger.warning(
                    "ValidatorRegistry not provided for FULL verification, using JUDGE strategy"
                )
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.JUDGE,
                    **kwargs,
                )

            # If we only have ValidatorRegistry, use that
            if self.judge_agent is None and self.validator_registry is not None:
                logger.warning(
                    "JudgeAgent not provided for FULL verification, using VALIDATOR strategy"
                )
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.VALIDATOR,
                    **kwargs,
                )

            # We have both JudgeAgent and ValidatorRegistry, use both
            try:
                # First, run validation with ValidatorRegistry
                validator_type = kwargs.get("validator_type")
                validator_ids = kwargs.get("validator_ids")

                assert self.validator_registry is not None
                validation_results = await self.validator_registry.validate(
                    output=output,
                    prompt=prompt,
                    validator_type=validator_type,
                    validator_ids=validator_ids,
                    **kwargs,
                )

                # Check if validation passed
                validator_passed = True
                validator_score = 0.0
                validator_feedback = ""

                if validation_results:
                    # Calculate the overall score and status
                    total_score = 0.0
                    failed_validations = []

                    for result in validation_results:
                        # Add to total score
                        score = result.metadata.get("score", 0.0)
                        total_score += score

                        # Check if validation failed
                        if result.status == ValidationStatus.FAILED:
                            failed_validations.append(result)

                    # Calculate average score
                    validator_score = (
                        total_score / len(validation_results) if validation_results else 0.0
                    )

                    # Determine if validation passed
                    validator_passed = len(failed_validations) == 0

                    # Format feedback
                    if failed_validations:
                        validator_feedback = "Validation failed:\n" + "\n".join(
                            [
                                f"- {result.validator_id}: {result.message}"
                                for result in failed_validations
                            ]
                        )
                    else:
                        validator_feedback = "All validations passed"

                # Next, run judgment with JudgeAgent
                assert self.judge_agent is not None
                judge_result = await self.judge_agent.judge(output=output, prompt=prompt)

                # Get the judgment result
                judge_passed = judge_result.passed
                judge_score = judge_result.overall_score
                judge_feedback = self.judge_agent.format_critique(judge_result)

                # Combine the results
                passed = validator_passed and judge_passed

                # Weight the scores (50% validator, 50% judge)
                score = (validator_score + judge_score) / 2.0

                # Combine the feedback
                feedback = f"Validator: {validator_feedback}\n\nJudge: {judge_feedback}"

                # Log the verification result
                if passed:
                    logger.info(f"Output passed FULL verification with score {score:.2f}")
                else:
                    logger.warning(f"Output failed FULL verification with score {score:.2f}")

                return passed, score, feedback
            except Exception as e:
                logger.exception(f"Error during FULL verification: {e}")
                logger.warning("Falling back to basic verification")
                return await self._verify_output(
                    output=output,
                    prompt=prompt,
                    verification_strategy=VerificationStrategy.BASIC,
                    **kwargs,
                )

        # Unknown strategy
        logger.warning(f"Unknown verification strategy: {strategy}")
        return True, None, None

    async def _refine_output(
        self,
        prompt: str,
        rejected_output: str,
        verification_feedback: str | None,
        documents: list[Document] | None = None,
        attempt: int = 1,
        **kwargs,
    ) -> LLMResponse:
        """
        Refine a rejected output using the configured refinement strategy.

        Args:
        ----
            prompt: Original prompt
            rejected_output: Rejected output
            verification_feedback: Feedback from verification
            documents: Documents used in the prompt (for GASA)
            attempt: Current refinement attempt (used for iterative refinement)
            **kwargs: Additional generation parameters

        Returns:
        -------
            LLMResponse: Refined response

        """
        strategy = self.config.refinement_strategy

        # No refinement
        if strategy == RefinementStrategy.NONE:
            # Just retry with the original prompt
            return await self._generate_final(prompt, documents, **kwargs)

        # Simple retry
        if strategy == RefinementStrategy.RETRY:
            # Retry with the original prompt
            return await self._generate_final(prompt, documents, **kwargs)

        # Feedback refinement
        if strategy == RefinementStrategy.FEEDBACK:
            # Create a new prompt with feedback
            feedback_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt was rejected for the following reason:\n"
                f"{verification_feedback or 'Unknown reason'}\n\n"
                f"Please try again, addressing the feedback. "
                f"This is refinement attempt #{attempt} out of {self.config.max_refinement_attempts}."
            )

            # Generate with the feedback prompt
            return await self._generate_final(feedback_prompt, documents, **kwargs)

        # Iterative refinement
        if strategy == RefinementStrategy.ITERATIVE:
            # Create a new prompt with feedback and the rejected output
            iterative_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt (#{attempt}):\n{rejected_output}\n\n"
                f"Feedback:\n{verification_feedback or 'Unknown reason'}\n\n"
                f"Please improve the previous attempt based on the feedback. "
                f"This is refinement attempt #{attempt} out of {self.config.max_refinement_attempts}."
            )

            # Generate with the iterative prompt
            return await self._generate_final(iterative_prompt, documents, **kwargs)

        # Unknown strategy
        logger.warning(f"Unknown refinement strategy: {strategy}")
        return await self._generate_final(prompt, documents, **kwargs)

    async def execute(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        stream: bool | None = None,
        on_draft: Callable[[str], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute a prompt to generate text.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            stream: Whether to stream the output (overrides config)
            on_draft: Callback for draft completion
            on_chunk: Callback for streaming chunks
            **kwargs: Additional generation parameters

        Returns:
        -------
            ExecutionResult: Result of the execution

        """
        # Determine whether to stream
        should_stream = self.config.enable_streaming if stream is None else stream

        # Check cache if enabled
        if self.config.cache_results:
            cache_key = self._get_cache_key(prompt, **kwargs)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                return self._cache[cache_key]

        # Start timing
        start_time = time.time()

        # If streaming is enabled, use the streaming implementation
        if should_stream:
            return await self._execute_streaming(
                prompt=prompt,
                documents=documents,
                on_draft=on_draft,
                on_chunk=on_chunk,
                **kwargs,
            )

        # Otherwise, execute normally and return the result

        # Generate draft if speculative execution is enabled
        draft_response = None
        if self.config.enable_speculative_execution:
            draft_response = await self._generate_draft(prompt, documents, **kwargs)
            draft_latency_ms = draft_response.metadata.get("latency_ms", 0)

            # Call draft callback if provided
            if on_draft is not None:
                on_draft(draft_response.text or "")
        else:
            draft_latency_ms = 0

        # Generate final response
        final_response = await self._generate_final(prompt, documents, **kwargs)
        final_latency_ms = final_response.metadata.get("latency_ms", 0)

        # Verify the output
        verified, verification_score, verification_feedback = await self._verify_output(
            output=final_response.text or "",
            prompt=prompt,
            **kwargs,
        )

        # Refine the output if verification failed
        refinement_attempts = 0
        while (
            not verified
            and refinement_attempts < self.config.max_refinement_attempts
            and self.config.refinement_strategy != RefinementStrategy.NONE
        ):
            refinement_attempts += 1
            logger.info(
                f"Output verification failed (score: {verification_score}). "
                f"Refinement attempt {refinement_attempts}/{self.config.max_refinement_attempts}"
            )

            # Refine the output
            refined_response = await self._refine_output(
                prompt=prompt,
                rejected_output=final_response.text or "",
                verification_feedback=verification_feedback,
                documents=documents,
                attempt=refinement_attempts,
                **kwargs,
            )

            # Update final response
            final_response = refined_response

            # Verify the refined output
            verified, verification_score, verification_feedback = await self._verify_output(
                output=final_response.text or "",
                prompt=prompt,
                **kwargs,
            )

            # Update final latency
            final_latency_ms += final_response.metadata.get("latency_ms", 0)

        # Calculate total latency
        total_latency_ms = int((time.time() - start_time) * 1000)

        # Create execution result
        result = ExecutionResult.from_llm_response(
            response=final_response,
            verified=verified,
            verification_score=verification_score,
            verification_feedback=verification_feedback,
            draft_latency_ms=draft_latency_ms,
            final_latency_ms=final_latency_ms,
            total_latency_ms=total_latency_ms,
            refinement_attempts=refinement_attempts,
        )

        # Cache the result if enabled
        if self.config.cache_results:
            cache_key = self._get_cache_key(prompt, **kwargs)
            self._cache[cache_key] = result

        return result

    async def _generate_streaming_draft(
        self, prompt: str, documents: list[Document] | None = None, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a draft response with streaming output.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            **kwargs: Additional generation parameters

        Returns:
        -------
            Tuple[str, Dict[str, Any]]: Draft text and metadata

        """
        # Start timing
        start_time = time.time()

        # Set draft generation parameters
        draft_kwargs = {
            "max_tokens": self.config.max_draft_tokens,
            "temperature": self.config.draft_temperature,
            "chunk_size": self.config.stream_chunk_size,
            **kwargs,
        }

        # Apply GASA if enabled
        gasa_result = await self._apply_gasa(
            prompt=prompt,
            documents=documents,
            generation_type="streaming_draft",
            input_ids=draft_kwargs.get("input_ids"),
            attention_mask=draft_kwargs.get("attention_mask"),
        )

        # Update kwargs with GASA result
        if gasa_result and "attention_mask" in gasa_result:
            draft_kwargs["attention_mask"] = gasa_result["attention_mask"]

        # For prompt composers, use the modified prompt
        if gasa_result and "prompt" in gasa_result and gasa_result["prompt"] != prompt:
            prompt = gasa_result["prompt"]

        # Generate draft with streaming
        draft_text = ""
        stream_generator = self.model.generate_streaming(prompt, **draft_kwargs)
        async for chunk in stream_generator:
            draft_text += str(chunk)

        # Record latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Create metadata
        metadata = {
            "latency_ms": latency_ms,
            "temperature": self.config.draft_temperature,
            "max_tokens": self.config.max_draft_tokens,
        }

        return draft_text, metadata

    async def _generate_streaming_final(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a final response with streaming output.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            on_chunk: Callback for streaming chunks
            **kwargs: Additional generation parameters

        Returns:
        -------
            Tuple[str, Dict[str, Any]]: Final text and metadata

        """
        # Start timing
        start_time = time.time()

        # Set final generation parameters
        final_kwargs = {
            "max_tokens": self.config.max_final_tokens,
            "temperature": self.config.final_temperature,
            "chunk_size": self.config.stream_chunk_size,
            **kwargs,
        }

        # Apply GASA if enabled
        gasa_result = await self._apply_gasa(
            prompt=prompt,
            documents=documents,
            generation_type="streaming_final",
            input_ids=final_kwargs.get("input_ids"),
            attention_mask=final_kwargs.get("attention_mask"),
        )

        # Update kwargs with GASA result
        if gasa_result and "attention_mask" in gasa_result:
            final_kwargs["attention_mask"] = gasa_result["attention_mask"]

        # For prompt composers, use the modified prompt
        if gasa_result and "prompt" in gasa_result and gasa_result["prompt"] != prompt:
            prompt = gasa_result["prompt"]

        # Generate final response with streaming
        final_text = ""
        stream_generator = self.model.generate_streaming(prompt, **final_kwargs)
        async for chunk in stream_generator:
            final_text += str(chunk)

            # Call chunk callback if provided
            if on_chunk is not None:
                on_chunk(str(chunk))

        # Record latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Create metadata
        metadata = {
            "latency_ms": latency_ms,
            "temperature": self.config.final_temperature,
            "max_tokens": self.config.max_final_tokens,
        }

        return final_text, metadata

    async def _execute_streaming(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        on_draft: Callable[[str], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute a prompt with streaming output.

        Args:
        ----
            prompt: Prompt text
            documents: Documents used in the prompt (for GASA)
            on_draft: Callback for draft completion
            on_chunk: Callback for streaming chunks
            **kwargs: Additional generation parameters

        Returns:
        -------
            ExecutionResult: Result of the execution

        """
        # Start timing
        start_time = time.time()

        # Generate draft if speculative execution is enabled
        draft_text = None
        draft_metadata = None
        if self.config.enable_speculative_execution:
            draft_text, draft_metadata = await self._generate_streaming_draft(
                prompt, documents, **kwargs
            )
            draft_latency_ms = draft_metadata.get("latency_ms", 0)

            # Call draft callback if provided
            if on_draft is not None:
                on_draft(draft_text or "")
        else:
            draft_latency_ms = 0

        # Generate final response with streaming
        final_text, final_metadata = await self._generate_streaming_final(
            prompt, documents, on_chunk, **kwargs
        )
        final_latency_ms = final_metadata.get("latency_ms", 0)

        # Create a simulated LLMResponse for verification
        # Ensure token counts are integers
        prompt_tokens = int(self.model.estimate_tokens(prompt))
        completion_tokens = int(self.model.estimate_tokens(final_text))
        total_tokens = prompt_tokens + completion_tokens

        # Extract tool calls from kwargs if present
        tool_calls = kwargs.get("tool_calls")
        function_call = kwargs.get("function_call")

        # Convert function_call to a dictionary if it's a string
        if isinstance(function_call, str):
            if function_call == "auto":
                function_call = {"name": "auto"}
            elif function_call == "none":
                function_call = None
        elif not isinstance(function_call, dict) and function_call is not None:
            function_call = None

        # Ensure function_call is a dict or None
        if not isinstance(function_call, dict):
            function_call = None
        final_response = LLMResponse(
            text=final_text,
            provider=getattr(self.model, "provider", "unknown"),
            model_name=getattr(self.model, "model_name", "unknown"),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata=final_metadata,
            function_call=function_call,
            tool_calls=tool_calls,
        )

        # Verify the output
        verified, verification_score, verification_feedback = await self._verify_output(
            output=final_text or "",
            prompt=prompt,
            **kwargs,
        )

        # Refine the output if verification failed
        refinement_attempts = 0
        while (
            not verified
            and refinement_attempts < self.config.max_refinement_attempts
            and self.config.refinement_strategy != RefinementStrategy.NONE
        ):
            refinement_attempts += 1
            logger.info(
                f"Output verification failed (score: {verification_score}). "
                f"Refinement attempt {refinement_attempts}/{self.config.max_refinement_attempts}"
            )

            # Refine the output
            refined_response = await self._refine_output(
                prompt=prompt,
                rejected_output=final_text or "",
                verification_feedback=verification_feedback,
                documents=documents,
                attempt=refinement_attempts,
                **kwargs,
            )

            # Update final response
            final_response = refined_response
            final_text = refined_response.text or ""

            # Call chunk callback if provided
            if on_chunk is not None:
                on_chunk(f"\n[Refinement {refinement_attempts}]\n{final_text}")

            # Verify the refined output
            verified, verification_score, verification_feedback = await self._verify_output(
                output=final_text or "",
                prompt=prompt,
                **kwargs,
            )

            # Update final latency
            final_latency_ms += refined_response.metadata.get("latency_ms", 0)

        # Calculate total latency
        total_latency_ms = int((time.time() - start_time) * 1000)

        # Create execution result
        return ExecutionResult.from_llm_response(
            response=final_response,
            verified=verified,
            verification_score=verification_score,
            verification_feedback=verification_feedback,
            draft_latency_ms=draft_latency_ms,
            final_latency_ms=final_latency_ms,
            total_latency_ms=total_latency_ms,
            refinement_attempts=refinement_attempts,
        )
