from __future__ import annotations

"""
Configuration module for the Judge.

This module defines the configuration classes for the Judge module.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScoringDimension(str, Enum):
    """Dimensions for scoring outputs."""

    RELEVANCE = "relevance"  # How relevant the output is to the prompt
    CORRECTNESS = "correctness"  # How factually correct the output is
    COHERENCE = "coherence"  # How coherent and well-structured the output is
    CONCISENESS = "conciseness"  # How concise the output is
    HELPFULNESS = "helpfulness"  # How helpful the output is
    CREATIVITY = "creativity"  # How creative the output is
    SAFETY = "safety"  # How safe and appropriate the output is
    CUSTOM = "custom"  # Custom dimension defined by the user


class CritiqueFormat(str, Enum):
    """Format for critique output."""

    SIMPLE = "simple"  # Simple text feedback
    STRUCTURED = "structured"  # Structured feedback with sections
    JSON = "json"  # JSON-formatted feedback
    MARKDOWN = "markdown"  # Markdown-formatted feedback


class RubricItem(BaseModel):
    """A single item in a rubric."""

    dimension: ScoringDimension = Field(..., description="Dimension to score")
    weight: float = Field(1.0, description="Weight of this dimension in the overall score")
    description: str = Field("", description="Description of what this dimension measures")
    criteria: dict[str, str] = Field(
        default_factory=dict, description="Criteria for different score levels"
    )


class Rubric(BaseModel):
    """A rubric for evaluating outputs."""

    name: str = Field(..., description="Name of the rubric")
    description: str = Field("", description="Description of the rubric")
    items: list[RubricItem] = Field(default_factory=list, description="Items in the rubric")

    @classmethod
    def default(cls):
        """
        Create a default rubric.

        Returns
        -------
            Rubric: Default rubric

        """
        return cls(
            name="Default Rubric",
            description="Default rubric for evaluating outputs",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is to the prompt",
                    criteria={
                        "0.0": "Completely irrelevant",
                        "0.5": "Somewhat relevant",
                        "1.0": "Highly relevant",
                    },
                ),
                RubricItem(
                    dimension=ScoringDimension.CORRECTNESS,
                    weight=1.0,
                    description="How factually correct the output is",
                    criteria={
                        "0.0": "Contains major factual errors",
                        "0.5": "Contains minor factual errors",
                        "1.0": "Factually correct",
                    },
                ),
                RubricItem(
                    dimension=ScoringDimension.COHERENCE,
                    weight=1.0,
                    description="How coherent and well-structured the output is",
                    criteria={
                        "0.0": "Incoherent or poorly structured",
                        "0.5": "Somewhat coherent and structured",
                        "1.0": "Very coherent and well-structured",
                    },
                ),
                RubricItem(
                    dimension=ScoringDimension.HELPFULNESS,
                    weight=1.0,
                    description="How helpful the output is",
                    criteria={
                        "0.0": "Not helpful",
                        "0.5": "Somewhat helpful",
                        "1.0": "Very helpful",
                    },
                ),
            ],
        )


class JudgeConfig(BaseModel):
    """Configuration for the Judge module."""

    # Scoring settings
    rubric: Rubric = Field(default_factory=Rubric.default, description="Rubric for evaluation")
    threshold: float = Field(0.7, description="Threshold for passing verification (0.0 to 1.0)")

    # Critique settings
    critique_format: CritiqueFormat = Field(
        CritiqueFormat.STRUCTURED, description="Format for critique output"
    )
    include_scores: bool = Field(True, description="Whether to include scores in the critique")
    include_suggestions: bool = Field(
        True, description="Whether to include improvement suggestions in the critique"
    )

    # Budget settings
    enforce_budget: bool = Field(True, description="Whether to enforce budget constraints")
    max_tokens_per_judgment: int | None = Field(None, description="Maximum tokens per judgment")
    max_cost_per_judgment: float | None = Field(
        None, description="Maximum cost per judgment in USD"
    )

    # Model settings
    model_uri: str | None = Field(None, description="URI of the model to use for judgment")

    @classmethod
    def default(cls):
        """
        Create a default configuration.

        Returns
        -------
            JudgeConfig: Default configuration

        """
        return cls(
            threshold=0.7,
            critique_format=CritiqueFormat.STRUCTURED,
            include_scores=True,
            include_suggestions=True,
            enforce_budget=True,
            max_tokens_per_judgment=None,
            max_cost_per_judgment=None,
            model_uri=None,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JudgeConfig":
        """
        Create a configuration from a dictionary.

        Args:
        ----
            config_dict: Configuration dictionary

        Returns:
        -------
            JudgeConfig: Configuration

        """
        return cls(**config_dict)
