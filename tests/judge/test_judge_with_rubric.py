"""
Tests for the judge module with rubrics.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.judge.config import CritiqueFormat, JudgeConfig, Rubric, RubricItem, ScoringDimension
from saplings.judge.judge_agent import JudgeAgent
from saplings.judge.rubric import RubricLoader, RubricTemplate


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, response_text: str = "", metadata: Optional[ModelMetadata] = None):
        """Initialize the mock LLM."""
        self.response_text = response_text
        self._metadata = metadata or ModelMetadata(
            name="mock_model",
            provider="mock",
            model_uri="mock://model",
            roles=[ModelRole.JUDGE],
            version="1.0.0",
            context_window=4096,
            max_tokens_per_request=1024,
        )
        self.generate_called = False
        self.prompt = ""
        self.model_uri = "mock://model"

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response."""
        self.generate_called = True
        self.prompt = prompt

        return LLMResponse(
            text=self.response_text,
            model_uri="mock://model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    async def generate_streaming(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        self.generate_called = True
        self.prompt = prompt

        # Just yield the entire response at once for simplicity
        yield self.response_text

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request."""
        return 0.0

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        # Simple estimation: 1 token per 4 characters
        return len(text) // 4


class TestJudgeWithRubric:
    """Tests for the JudgeAgent class with rubrics."""

    @pytest.fixture
    def mock_llm(self) -> MockLLM:
        """Create a mock LLM."""
        return MockLLM(
            response_text="""
            DIMENSION SCORES:
            - relevance: 0.8
              Explanation: The response is mostly relevant to the prompt.
            - correctness: 0.9
              Explanation: The response is factually correct.
            - coherence: 0.7
              Explanation: The response is somewhat coherent.
            - helpfulness: 0.85
              Explanation: The response is helpful.

            OVERALL SCORE: 0.8125

            CRITIQUE:
            The response is good but could be more coherent.

            SUGGESTIONS:
            1. Improve the structure of the response.
            2. Add more details to support the main points.
            """
        )

    @pytest.fixture
    def judge_agent(self, mock_llm: MockLLM) -> JudgeAgent:
        """Create a judge agent."""
        config = JudgeConfig(
            critique_format=CritiqueFormat.STRUCTURED,
            threshold=0.7,
            max_tokens_per_judgment=500,
        )
        return JudgeAgent(model=mock_llm, config=config)

    @pytest.mark.asyncio
    async def test_judge_with_default_rubric(self, judge_agent: JudgeAgent):
        """Test judging with the default rubric."""
        # Judge an output
        result = await judge_agent.judge(
            output="This is a test output.",
            prompt="This is a test prompt.",
        )

        # Check the result
        assert result.passed
        assert result.overall_score == 0.8125
        assert len(result.dimension_scores) == 4
        assert result.critique == "The response is good but could be more coherent."
        assert len(result.suggestions) == 2
        assert "Improve the structure of the response." in result.suggestions

    @pytest.mark.asyncio
    async def test_judge_with_custom_rubric(self, judge_agent: JudgeAgent):
        """Test judging with a custom rubric."""
        # Create a custom rubric
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=2.0,  # Higher weight for relevance
                    description="How relevant the output is",
                    criteria={
                        "0.0": "Not relevant",
                        "1.0": "Very relevant",
                    },
                ),
                RubricItem(
                    dimension=ScoringDimension.CORRECTNESS,
                    weight=1.0,
                    description="How correct the output is",
                    criteria={
                        "0.0": "Not correct",
                        "1.0": "Very correct",
                    },
                ),
            ],
        )

        # Judge an output with the custom rubric
        result = await judge_agent.judge(
            output="This is a test output.",
            prompt="This is a test prompt.",
            rubric=rubric,
        )

        # Check the result
        assert result.passed
        assert result.overall_score == 0.8125
        assert len(result.dimension_scores) == 4
        assert "Test Rubric" in result.metadata["rubric_name"]

    @pytest.mark.asyncio
    async def test_judge_with_template(self, judge_agent: JudgeAgent):
        """Test judging with a rubric template."""
        # Mock the RubricLoader.load_from_template method
        with patch("saplings.judge.rubric.RubricLoader.load_from_template") as mock_load:
            # Create a mock rubric
            mock_rubric = Rubric(
                name="Code Evaluation Rubric",
                description="Rubric for evaluating code outputs",
                items=[
                    RubricItem(
                        dimension=ScoringDimension.CORRECTNESS,
                        weight=2.0,
                        description="How correct the code is",
                        criteria={
                            "0.0": "Not correct",
                            "1.0": "Very correct",
                        },
                    ),
                ],
            )

            # Configure the mock to return the mock rubric
            mock_load.return_value = mock_rubric

            # Judge an output with the template
            result = await judge_agent.judge_with_template(
                output="This is a test output.",
                prompt="This is a test prompt.",
                template="code",
            )

            # Check that the template was loaded
            mock_load.assert_called_once_with("code")

            # Check the result
            assert result.passed
            assert result.overall_score == 0.8125
            assert len(result.dimension_scores) == 4
            assert "Code Evaluation Rubric" in result.metadata["rubric_name"]

    @pytest.mark.asyncio
    async def test_judge_with_all_templates(self, judge_agent: JudgeAgent):
        """Test judging with all rubric templates."""
        # Test each template
        for template in RubricTemplate:
            # Judge an output with the template
            result = await judge_agent.judge_with_template(
                output="This is a test output.",
                prompt="This is a test prompt.",
                template=template.value,
            )

            # Check the result
            assert result.passed
            assert result.overall_score == 0.8125
            assert len(result.dimension_scores) == 4

    @pytest.mark.asyncio
    async def test_rubric_based_scoring_weights(self):
        """Test that rubric weights affect the overall score."""
        # Create dimension scores directly
        dimension_scores = [
            {
                "dimension": "relevance",
                "score": 1.0,
                "explanation": "The response is completely relevant to the prompt.",
            },
            {
                "dimension": "correctness",
                "score": 0.5,
                "explanation": "The response has some factual errors.",
            },
            {
                "dimension": "coherence",
                "score": 0.8,
                "explanation": "The response is mostly coherent.",
            },
            {
                "dimension": "helpfulness",
                "score": 0.7,
                "explanation": "The response is somewhat helpful.",
            },
        ]

        # Create two different rubrics with different weights
        # Rubric 1: Equal weights
        equal_weights_rubric = Rubric(
            name="Equal Weights Rubric",
            items=[
                RubricItem(dimension=ScoringDimension.RELEVANCE, weight=1.0),
                RubricItem(dimension=ScoringDimension.CORRECTNESS, weight=1.0),
                RubricItem(dimension=ScoringDimension.COHERENCE, weight=1.0),
                RubricItem(dimension=ScoringDimension.HELPFULNESS, weight=1.0),
            ],
        )

        # Rubric 2: Emphasize correctness
        correctness_weighted_rubric = Rubric(
            name="Correctness Weighted Rubric",
            items=[
                RubricItem(dimension=ScoringDimension.RELEVANCE, weight=1.0),
                RubricItem(dimension=ScoringDimension.CORRECTNESS, weight=3.0),  # 3x weight
                RubricItem(dimension=ScoringDimension.COHERENCE, weight=1.0),
                RubricItem(dimension=ScoringDimension.HELPFULNESS, weight=1.0),
            ],
        )

        # Create a judge agent to test the calculation method directly
        mock_llm = MockLLM()
        judge_agent = JudgeAgent(model=mock_llm)

        # Calculate scores with different rubrics
        equal_weights_score = judge_agent._calculate_overall_score(
            dimension_scores, equal_weights_rubric
        )
        correctness_weighted_score = judge_agent._calculate_overall_score(
            dimension_scores, correctness_weighted_rubric
        )

        # The correctness_weighted_score should be lower because correctness has a higher weight and a lower score (0.5)
        assert equal_weights_score > correctness_weighted_score

        # Calculate the expected scores
        # Equal weights: (1.0 + 0.5 + 0.8 + 0.7) / 4 = 0.75
        expected_equal_score = (1.0 + 0.5 + 0.8 + 0.7) / 4

        # Correctness weighted: (1.0*1.0 + 0.5*3.0 + 0.8*1.0 + 0.7*1.0) / (1.0 + 3.0 + 1.0 + 1.0) = 3.5/6.0 = 0.583
        expected_weighted_score = (1.0 * 1.0 + 0.5 * 3.0 + 0.8 * 1.0 + 0.7 * 1.0) / (
            1.0 + 3.0 + 1.0 + 1.0
        )

        # Check that the calculated scores match the expected scores
        assert abs(equal_weights_score - expected_equal_score) < 0.01
        assert abs(correctness_weighted_score - expected_weighted_score) < 0.01
