from __future__ import annotations

"""
JudgeAgent module for Saplings.

This module provides the JudgeAgent class for evaluating outputs based on scoring criteria
and providing structured feedback.
"""


import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from saplings.core.model_adapter import LLM, ModelRole
from saplings.judge.config import CritiqueFormat, JudgeConfig, Rubric, ScoringDimension

logger = logging.getLogger(__name__)


class DimensionScore(BaseModel):
    """Score for a single dimension."""

    dimension: ScoringDimension = Field(..., description="Dimension being scored")
    score: float = Field(..., description="Score for this dimension (0.0 to 1.0)")
    explanation: str = Field("", description="Explanation for the score")


class JudgeResult(BaseModel):
    """Result of a judgment."""

    output: str = Field(..., description="Output that was judged")
    prompt: str = Field(..., description="Prompt that generated the output")
    passed: bool = Field(..., description="Whether the output passed verification")
    overall_score: float = Field(..., description="Overall score (0.0 to 1.0)")
    dimension_scores: list[DimensionScore] = Field(
        default_factory=list, description="Scores for individual dimensions"
    )
    critique: str = Field("", description="Critique of the output")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the judgment result to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "output": self.output,
            "prompt": self.prompt,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "dimension_scores": [score.model_dump() for score in self.dimension_scores],
            "critique": self.critique,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeResult":
        """
        Create a judgment result from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            JudgeResult: Judgment result

        """
        # Convert dimension scores from dicts to DimensionScore objects
        if "dimension_scores" in data:
            data["dimension_scores"] = [
                DimensionScore(**score) for score in data["dimension_scores"]
            ]

        return cls(**data)


class JudgeAgent:
    """
    Agent for evaluating outputs based on scoring criteria and providing structured feedback.

    This class provides the core functionality for judging outputs, including:
    - Scoring outputs based on rubrics
    - Generating structured critiques
    - Enforcing budget constraints
    - Tracking statistics
    """

    def __init__(
        self,
        model: LLM | None = None,
        config: JudgeConfig | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        **model_parameters,
    ) -> None:
        """
        Initialize the judge agent.

        Args:
        ----
            model: LLM model to use for judgment
            config: Judge configuration
            provider: Model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: Model name
            **model_parameters: Additional model parameters

        """
        # Handle the new approach for specifying models
        if model is None and provider is not None and model_name is not None:
            try:
                # Create a model using the new approach
                model = LLM.create(provider=provider, model_name=model_name, **model_parameters)
            except Exception as e:
                # If vLLM fails due to Triton issues or other initialization problems
                if provider == "vllm" and (
                    "triton" in str(e).lower()
                    or "failed to be inspected" in str(e).lower()
                    or "vllm not installed" in str(e).lower()
                ):
                    logger.warning(f"vLLM initialization failed: {e}. Trying fallback providers.")

                    # Check if we're on Apple Silicon
                    import platform

                    IS_APPLE_SILICON = (
                        platform.system() == "Darwin" and platform.machine().startswith("arm")
                    )

                    # List of fallback providers to try in order
                    fallback_providers = []

                    # If we're on Apple Silicon, prioritize providers known to work well on Mac
                    if IS_APPLE_SILICON:
                        # These providers are known to work better on Apple Silicon
                        fallback_providers = [
                            (
                                "transformers",
                                "facebook/opt-125m",
                            ),  # Direct Transformers is often most reliable on Apple Silicon
                            ("transformers", "distilgpt2"),
                            ("transformers", "gpt2"),
                            ("huggingface", "distilgpt2"),
                            ("huggingface", "gpt2"),
                            ("huggingface", "facebook/opt-125m"),
                            ("openai", "gpt-3.5-turbo"),
                            ("anthropic", "claude-instant-1"),
                        ]
                    else:
                        # For other platforms
                        fallback_providers = [
                            ("openai", "gpt-3.5-turbo"),
                            ("anthropic", "claude-instant-1"),
                            ("transformers", "facebook/opt-125m"),
                            ("transformers", "distilgpt2"),
                            ("huggingface", "distilgpt2"),
                            ("huggingface", "gpt2"),
                        ]

                    # Try each fallback provider in order
                    last_error = e
                    for fallback_provider, fallback_model in fallback_providers:
                        try:
                            logger.info(
                                f"Trying {fallback_provider}/{fallback_model} as a fallback"
                            )

                            # Filter out parameters that might not be compatible with the fallback provider
                            filtered_params = {
                                k: v
                                for k, v in model_parameters.items()
                                if k not in ["enable_tool_choice"]
                                and not (fallback_provider != "vllm" and k.startswith("gasa_"))
                            }

                            model = LLM.create(
                                provider=fallback_provider,
                                model_name=fallback_model,
                                **filtered_params,
                            )

                            # If we got here, we successfully created a model
                            logger.info(
                                f"Successfully created fallback model with {fallback_provider}/{fallback_model}"
                            )
                            break
                        except Exception as fallback_error:
                            logger.warning(
                                f"{fallback_provider}/{fallback_model} fallback failed: {fallback_error}"
                            )
                            last_error = fallback_error
                    else:
                        # If we've tried all fallbacks and none worked, raise the last error
                        logger.exception("All fallback providers failed")
                        raise last_error
                else:
                    # Re-raise other errors
                    raise
        elif model is None:
            msg = "Either 'model' or both 'provider' and 'model_name' must be provided"
            raise ValueError(msg)

        self.model = model
        self.config = config or JudgeConfig.default()

        # Initialize statistics
        self.total_judgments: int = 0
        self.passed_judgments: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

        # Validate model
        self._validate_model()

    def _validate_model(self):
        """
        Validate that the model is suitable for judgment.

        Raises
        ------
            ValueError: If the model is not suitable for judgment

        """
        # For now, we'll skip strict validation since model metadata might not be fully implemented
        # This is a more permissive approach that will work with any model
        try:
            metadata = self.model.get_metadata()
            # Check if metadata is a ModelMetadata object with roles attribute
            # or if it's a dict with a 'roles' key
            has_judge_role = False
            has_general_role = False

            if hasattr(metadata, "roles"):
                roles = getattr(metadata, "roles", [])
                if isinstance(roles, list):
                    has_judge_role = ModelRole.JUDGE in roles
                    has_general_role = ModelRole.GENERAL in roles
            elif isinstance(metadata, dict) and "roles" in metadata:
                roles = metadata["roles"]
                if isinstance(roles, list):
                    has_judge_role = ModelRole.JUDGE.value in roles
                    has_general_role = ModelRole.GENERAL.value in roles

            if not (has_judge_role or has_general_role):
                logger.warning(
                    f"Model {self.model.model_name} does not have JUDGE or GENERAL role. "
                    f"It may not be suitable for judgment tasks."
                )
        except Exception as e:
            # If we can't validate the model, log a warning but continue
            logger.warning(f"Could not validate model for judgment: {e}")

    def _create_judgment_prompt(
        self, output: str, prompt: str, rubric: Rubric | None = None
    ) -> str:
        """
        Create a prompt for judging an output.

        Args:
        ----
            output: Output to judge
            prompt: Prompt that generated the output
            rubric: Rubric to use for judgment

        Returns:
        -------
            str: Judgment prompt

        """
        rubric = rubric or self.config.rubric

        # Create the prompt
        judgment_prompt = (
            "You are a judge evaluating the quality of an AI assistant's response. "
            "Your task is to provide a fair and detailed assessment based on the given criteria.\n\n"
            "# Original Prompt\n"
            f"{prompt}\n\n"
            "# AI Assistant's Response\n"
            f"{output}\n\n"
            "# Evaluation Criteria\n"
        )

        # Add rubric items
        for item in rubric.items:
            judgment_prompt += f"## {item.dimension.value.capitalize()}\n"
            judgment_prompt += f"{item.description}\n"
            judgment_prompt += "Score levels:\n"

            # Add criteria for different score levels
            for score, description in sorted(item.criteria.items()):
                judgment_prompt += f"- {score}: {description}\n"

            judgment_prompt += "\n"

        # Add instructions for the response format
        judgment_prompt += (
            "# Instructions\n"
            "Please evaluate the response according to the criteria above. For each dimension:\n"
            "1. Assign a score between 0.0 and 1.0\n"
            "2. Provide a brief explanation for your score\n"
            "3. Calculate an overall score as a weighted average of the dimension scores\n"
            "4. Provide a detailed critique of the response\n"
            "5. Suggest specific improvements\n\n"
            "# Response Format\n"
        )

        # Add response format based on the configured format
        if self.config.critique_format == CritiqueFormat.JSON:
            judgment_prompt += (
                "Provide your evaluation in JSON format:\n"
                "```json\n"
                "{\n"
                '  "dimension_scores": [\n'
                '    {"dimension": "relevance", "score": 0.8, "explanation": "..."}, \n'
                '    {"dimension": "correctness", "score": 0.9, "explanation": "..."}, \n'
                "    ...\n"
                "  ],\n"
                '  "overall_score": 0.85,\n'
                '  "critique": "Detailed critique here...",\n'
                '  "suggestions": ["Suggestion 1", "Suggestion 2", ...]\n'
                "}\n"
                "```\n"
            )
        elif self.config.critique_format == CritiqueFormat.MARKDOWN:
            judgment_prompt += (
                "Provide your evaluation in Markdown format:\n\n"
                "## Dimension Scores\n\n"
                "- **Relevance**: 0.8\n"
                "  - Explanation: ...\n"
                "- **Correctness**: 0.9\n"
                "  - Explanation: ...\n"
                "...\n\n"
                "## Overall Score\n\n"
                "0.85\n\n"
                "## Critique\n\n"
                "Detailed critique here...\n\n"
                "## Suggestions\n\n"
                "1. Suggestion 1\n"
                "2. Suggestion 2\n"
                "...\n"
            )
        else:  # SIMPLE or STRUCTURED
            judgment_prompt += (
                "Provide your evaluation in the following format:\n\n"
                "DIMENSION SCORES:\n"
                "- Relevance: 0.8\n"
                "  Explanation: ...\n"
                "- Correctness: 0.9\n"
                "  Explanation: ...\n"
                "...\n\n"
                "OVERALL SCORE: 0.85\n\n"
                "CRITIQUE:\n"
                "Detailed critique here...\n\n"
                "SUGGESTIONS:\n"
                "1. Suggestion 1\n"
                "2. Suggestion 2\n"
                "...\n"
            )

        return judgment_prompt

    def _parse_judgment_response(self, response: str | None) -> dict[str, Any]:
        """
        Parse a judgment response.

        Args:
        ----
            response: Judgment response, can be None

        Returns:
        -------
            Dict[str, Any]: Parsed judgment

        """
        if response is None:
            logger.warning("Received None response from model")
            return {
                "dimension_scores": [],
                "overall_score": 0.0,
                "critique": "No response received from model.",
                "suggestions": ["Try again with a different model or prompt."],
            }
        # Try to parse as JSON first
        if self.config.critique_format == CritiqueFormat.JSON:
            try:
                # Extract JSON from the response if it's wrapped in ```json ... ```
                if "```json" in response and "```" in response.split("```json", 1)[1]:
                    json_str = response.split("```json", 1)[1].split("```", 1)[0]
                    return json.loads(json_str)
                # Try to parse the whole response as JSON
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse judgment response as JSON, falling back to text parsing"
                )

        # Parse as text
        result = {
            "dimension_scores": [],
            "overall_score": 0.0,
            "critique": "",
            "suggestions": [],
        }

        # Extract dimension scores
        if "DIMENSION SCORES:" in response or "## Dimension Scores" in response:
            # Split the response into sections
            sections = response.split("\n\n")

            # Find the dimension scores section
            for _i, section in enumerate(sections):
                if "DIMENSION SCORES:" in section or "## Dimension Scores" in section:
                    # Extract dimension scores
                    lines = section.split("\n")
                    current_dimension = None
                    current_score = None
                    current_explanation = ""

                    for line in lines[1:]:  # Skip the header
                        if line.strip() == "":
                            continue

                        # Check if this is a new dimension
                        if line.strip().startswith("-") or line.strip().startswith("*"):
                            # Save the previous dimension if there is one
                            if current_dimension is not None and current_score is not None:
                                result["dimension_scores"].append(
                                    {
                                        "dimension": current_dimension,
                                        "score": current_score,
                                        "explanation": current_explanation.strip(),
                                    }
                                )

                            # Parse the new dimension
                            parts = line.strip()[1:].strip().split(":", 1)
                            if len(parts) == 2:
                                current_dimension = parts[0].strip().lower()
                                try:
                                    current_score = float(parts[1].strip())
                                    current_explanation = ""
                                except ValueError:
                                    current_score = None
                        elif line.strip().startswith("Explanation:") or line.strip().startswith(
                            "  -"
                        ):
                            # This is an explanation
                            explanation = line.strip()
                            if explanation.startswith("Explanation:"):
                                explanation = explanation[len("Explanation:") :].strip()
                            elif explanation.startswith("  -"):
                                explanation = explanation[len("  -") :].strip()
                            current_explanation += " " + explanation

                    # Save the last dimension
                    if current_dimension is not None and current_score is not None:
                        result["dimension_scores"].append(
                            {
                                "dimension": current_dimension,
                                "score": current_score,
                                "explanation": current_explanation.strip(),
                            }
                        )

                    break

        # Extract overall score
        if "OVERALL SCORE:" in response:
            overall_score_line = response.split("OVERALL SCORE:", 1)[1].split("\n", 1)[0].strip()
            try:
                result["overall_score"] = float(overall_score_line)
            except ValueError:
                logger.warning(f"Failed to parse overall score: {overall_score_line}")
        elif "## Overall Score" in response:
            sections = response.split("## Overall Score", 1)[1].split("##", 1)[0].strip()
            try:
                result["overall_score"] = float(sections)
            except ValueError:
                logger.warning(f"Failed to parse overall score: {sections}")

        # Extract critique
        if "CRITIQUE:" in response:
            critique_section = response.split("CRITIQUE:", 1)[1].split("SUGGESTIONS:", 1)[0].strip()
            result["critique"] = critique_section
        elif "## Critique" in response:
            sections = response.split("## Critique", 1)[1]
            if "##" in sections:
                critique_section = sections.split("##", 1)[0].strip()
            else:
                critique_section = sections.strip()
            result["critique"] = critique_section

        # Extract suggestions
        if "SUGGESTIONS:" in response:
            suggestions_section = response.split("SUGGESTIONS:", 1)[1].strip()
            suggestions = []
            for line in suggestions_section.split("\n"):
                if line.strip() == "":
                    continue
                if line.strip()[0].isdigit() and ". " in line:
                    suggestions.append(line.split(". ", 1)[1].strip())
            result["suggestions"] = suggestions
        elif "## Suggestions" in response:
            sections = response.split("## Suggestions", 1)[1]
            if "##" in sections:
                suggestions_section = sections.split("##", 1)[0].strip()
            else:
                suggestions_section = sections.strip()
            suggestions = []
            for line in suggestions_section.split("\n"):
                if line.strip() == "":
                    continue
                if line.strip()[0].isdigit() and ". " in line:
                    suggestions.append(line.split(". ", 1)[1].strip())
            result["suggestions"] = suggestions

        return result

    def _calculate_overall_score(
        self, dimension_scores: list[dict[str, Any]], rubric: Rubric | None = None
    ) -> float:
        """
        Calculate the overall score from dimension scores.

        Args:
        ----
            dimension_scores: Scores for individual dimensions
            rubric: Rubric to use for weighting

        Returns:
        -------
            float: Overall score

        """
        rubric = rubric or self.config.rubric

        # Create a mapping of dimension to weight
        weights = {item.dimension.value: item.weight for item in rubric.items}

        # Calculate the weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for score in dimension_scores:
            dimension = score["dimension"]
            if isinstance(dimension, ScoringDimension):
                dimension = dimension.value

            weight = weights.get(dimension, 1.0)
            total_weight += weight
            weighted_sum += weight * score["score"]

        # Return the weighted average, or 0.0 if there are no scores
        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight

    async def judge(
        self,
        output: str,
        prompt: str,
        rubric: Rubric | None = None,
        template: str | None = None,
    ) -> JudgeResult:
        """
        Judge an output.

        Args:
        ----
            output: Output to judge
            prompt: Prompt that generated the output
            rubric: Rubric to use for judgment
            template: Rubric template to use (ignored if rubric is provided)

        Returns:
        -------
            JudgeResult: Judgment result

        """
        # Start timing
        start_time = time.time()

        # Load rubric from template if specified and no rubric is provided
        if rubric is None and template is not None:
            try:
                from saplings.judge.rubric import RubricLoader

                rubric = RubricLoader.load_from_template(template)
                logger.info(f"Loaded rubric template: {template}")
            except (ImportError, ValueError) as e:
                logger.warning(f"Failed to load rubric template: {e}")

        # Create the judgment prompt
        judgment_prompt = self._create_judgment_prompt(output, prompt, rubric)

        # Generate the judgment
        response = await self.model.generate(
            prompt=judgment_prompt,
            max_tokens=self.config.max_tokens_per_judgment,
        )

        # Parse the judgment response
        parsed_judgment = self._parse_judgment_response(response.text)

        # Calculate the overall score if not provided
        if "overall_score" not in parsed_judgment or parsed_judgment["overall_score"] == 0.0:
            parsed_judgment["overall_score"] = self._calculate_overall_score(
                parsed_judgment["dimension_scores"], rubric
            )

        # Determine if the output passed verification
        passed = parsed_judgment["overall_score"] >= self.config.threshold

        # Create dimension scores
        dimension_scores = []
        for score in parsed_judgment["dimension_scores"]:
            dimension = score["dimension"]
            if isinstance(dimension, str):
                try:
                    dimension = ScoringDimension(dimension)
                except ValueError:
                    dimension = ScoringDimension.CUSTOM

            dimension_scores.append(
                DimensionScore(
                    dimension=dimension,
                    score=score["score"],
                    explanation=score.get("explanation", ""),
                )
            )

        # Create the judgment result
        result = JudgeResult(
            output=output,
            prompt=prompt,
            passed=passed,
            overall_score=parsed_judgment["overall_score"],
            dimension_scores=dimension_scores,
            critique=parsed_judgment.get("critique", ""),
            suggestions=parsed_judgment.get("suggestions", []),
            metadata={
                "latency_ms": int((time.time() - start_time) * 1000),
                "provider": self.model.provider,
                "model_name": self.model.model_name,
                "usage": response.usage,
                "rubric_name": rubric.name if rubric else self.config.rubric.name,
            },
        )

        # Update statistics
        self.total_judgments += 1
        if passed:
            self.passed_judgments += 1

        self.total_tokens += response.usage.get("total_tokens", 0)
        self.total_cost += self.model.estimate_cost(
            response.usage.get("prompt_tokens", 0),
            response.usage.get("completion_tokens", 0),
        )

        return result

    async def judge_with_template(self, output: str, prompt: str, template: str) -> JudgeResult:
        """
        Judge an output using a predefined rubric template.

        Args:
        ----
            output: Output to judge
            prompt: Prompt that generated the output
            template: Rubric template to use

        Returns:
        -------
            JudgeResult: Judgment result

        """
        return await self.judge(output, prompt, template=template)

    def get_statistics(self):
        """
        Get statistics about the judge agent.

        Returns
        -------
            Dict[str, Any]: Statistics

        """
        return {
            "total_judgments": self.total_judgments,
            "passed_judgments": self.passed_judgments,
            "pass_rate": self.passed_judgments / self.total_judgments
            if self.total_judgments > 0
            else 0.0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    def format_critique(self, result: JudgeResult) -> str:
        """
        Format a critique for display.

        Args:
        ----
            result: Judgment result

        Returns:
        -------
            str: Formatted critique

        """
        if self.config.critique_format == CritiqueFormat.JSON:
            return json.dumps(result.to_dict(), indent=2)

        if self.config.critique_format == CritiqueFormat.MARKDOWN:
            critique = "# Judgment Result\n\n"

            if self.config.include_scores:
                critique += f"## Overall Score: {result.overall_score:.2f}\n\n"
                critique += f"**Passed**: {'Yes' if result.passed else 'No'}\n\n"

                critique += "## Dimension Scores\n\n"
                for score in result.dimension_scores:
                    critique += f"### {score.dimension.value.capitalize()}: {score.score:.2f}\n\n"
                    critique += f"{score.explanation}\n\n"

            critique += f"## Critique\n\n{result.critique}\n\n"

            if self.config.include_suggestions and result.suggestions:
                critique += "## Suggestions\n\n"
                for i, suggestion in enumerate(result.suggestions):
                    critique += f"{i + 1}. {suggestion}\n"

            return critique

        if self.config.critique_format == CritiqueFormat.STRUCTURED:
            critique = "JUDGMENT RESULT\n\n"

            if self.config.include_scores:
                critique += f"OVERALL SCORE: {result.overall_score:.2f}\n"
                critique += f"PASSED: {'Yes' if result.passed else 'No'}\n\n"

                critique += "DIMENSION SCORES:\n"
                for score in result.dimension_scores:
                    critique += f"- {score.dimension.value.capitalize()}: {score.score:.2f}\n"
                    critique += f"  {score.explanation}\n"
                critique += "\n"

            critique += f"CRITIQUE:\n{result.critique}\n\n"

            if self.config.include_suggestions and result.suggestions:
                critique += "SUGGESTIONS:\n"
                for i, suggestion in enumerate(result.suggestions):
                    critique += f"{i + 1}. {suggestion}\n"

            return critique

        # SIMPLE
        critique = (
            f"Score: {result.overall_score:.2f} ({'Passed' if result.passed else 'Failed'})\n\n"
        )
        critique += f"{result.critique}\n\n"

        if self.config.include_suggestions and result.suggestions:
            critique += "Suggestions:\n"
            for suggestion in result.suggestions:
                critique += f"- {suggestion}\n"

        return critique
