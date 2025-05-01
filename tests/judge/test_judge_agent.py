"""
Tests for the judge module.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.judge import CritiqueFormat, JudgeAgent, JudgeConfig, JudgeResult, ScoringDimension


class MockJudgeLLM(LLM):
    """Mock LLM for testing the JudgeAgent."""

    def __init__(self, model_uri, **kwargs):
        """Initialize the mock LLM."""
        self.model_uri = model_uri
        self.kwargs = kwargs
        self.tokenizer = None

    async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs) -> LLMResponse:
        """Generate text from the model."""
        # Check if this is a judgment prompt
        if "You are a judge evaluating the quality of an AI assistant's response" in prompt:
            # Create a mock judgment response
            if "DIMENSION SCORES:" in prompt or "## Dimension Scores" in prompt:
                # Structured or Markdown format
                response_text = (
                    "DIMENSION SCORES:\n"
                    "- Relevance: 0.8\n"
                    "  Explanation: The response is highly relevant to the prompt.\n"
                    "- Correctness: 0.9\n"
                    "  Explanation: The response is factually correct.\n"
                    "- Coherence: 0.7\n"
                    "  Explanation: The response is well-structured but could be more coherent.\n"
                    "- Helpfulness: 0.85\n"
                    "  Explanation: The response is very helpful.\n\n"
                    "OVERALL SCORE: 0.81\n\n"
                    "CRITIQUE:\n"
                    "The response is generally good, addressing the prompt effectively. "
                    "It provides accurate information and is helpful to the user. "
                    "However, it could be more coherent in some places.\n\n"
                    "SUGGESTIONS:\n"
                    "1. Improve the flow between paragraphs.\n"
                    "2. Add more specific examples.\n"
                    "3. Consider providing additional context."
                )
            else:
                # JSON format
                response_text = (
                    "```json\n"
                    "{\n"
                    '  "dimension_scores": [\n'
                    '    {"dimension": "relevance", "score": 0.8, "explanation": "The response is highly relevant to the prompt."}, \n'
                    '    {"dimension": "correctness", "score": 0.9, "explanation": "The response is factually correct."}, \n'
                    '    {"dimension": "coherence", "score": 0.7, "explanation": "The response is well-structured but could be more coherent."}, \n'
                    '    {"dimension": "helpfulness", "score": 0.85, "explanation": "The response is very helpful."}\n'
                    "  ],\n"
                    '  "overall_score": 0.81,\n'
                    '  "critique": "The response is generally good, addressing the prompt effectively. It provides accurate information and is helpful to the user. However, it could be more coherent in some places.",\n'
                    '  "suggestions": ["Improve the flow between paragraphs.", "Add more specific examples.", "Consider providing additional context."]\n'
                    "}\n"
                    "```\n"
                )
        else:
            # Simple mock implementation for non-judgment prompts
            response_text = f"Response to: {prompt[:50]}..."

        return LLMResponse(
            text=response_text,
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def generate_streaming(
        self, prompt, max_tokens=None, temperature=None, chunk_size=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from the model with streaming output."""
        # Generate the full response first
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)

        # Split the response into chunks
        words = response.text.split()
        chunk_size = chunk_size or 5  # Default to 5 words per chunk

        # Yield chunks with simulated delay
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            # Simulate some processing time
            await asyncio.sleep(0.01)
            yield chunk

    def get_metadata(self) -> ModelMetadata:
        """Get metadata about the model."""
        return ModelMetadata(
            name="mock-judge-model",
            provider="mock-provider",
            version="latest",
            roles=[ModelRole.JUDGE, ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=2048,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        return len(text.split())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request."""
        return (prompt_tokens + completion_tokens) * 0.0001


class TestJudgeAgent:
    """Tests for the JudgeAgent class."""

    @pytest.fixture
    def mock_judge_llm(self):
        """Create a mock LLM for testing."""
        return MockJudgeLLM("mock://judge-model/latest")

    @pytest.fixture
    def judge_agent(self, mock_judge_llm):
        """Create a judge agent with default configuration."""
        return JudgeAgent(model=mock_judge_llm)

    @pytest.fixture
    def judge_agent_with_json_format(self, mock_judge_llm):
        """Create a judge agent with JSON critique format."""
        config = JudgeConfig(critique_format=CritiqueFormat.JSON)
        return JudgeAgent(model=mock_judge_llm, config=config)

    @pytest.mark.asyncio
    async def test_initialization(self, mock_judge_llm):
        """Test judge agent initialization."""
        # Test with default config
        judge_agent = JudgeAgent(model=mock_judge_llm)
        assert judge_agent.model == mock_judge_llm
        assert judge_agent.config is not None
        assert judge_agent.total_judgments == 0
        assert judge_agent.passed_judgments == 0

        # Test with custom config
        config = JudgeConfig(
            threshold=0.8,
            critique_format=CritiqueFormat.JSON,
        )
        judge_agent = JudgeAgent(model=mock_judge_llm, config=config)
        assert judge_agent.config == config

    @pytest.mark.asyncio
    async def test_judge(self, judge_agent):
        """Test judging an output."""
        # Create a prompt and output
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."

        # Judge the output
        result = await judge_agent.judge(output=output, prompt=prompt)

        # Check the result
        assert isinstance(result, JudgeResult)
        assert result.output == output
        assert result.prompt == prompt
        assert result.overall_score > 0.0
        assert len(result.dimension_scores) > 0
        assert result.critique != ""
        assert len(result.suggestions) > 0

        # Check that statistics were updated
        assert judge_agent.total_judgments == 1
        assert judge_agent.passed_judgments == 1 if result.passed else 0

    @pytest.mark.asyncio
    async def test_judge_with_json_format(self, judge_agent_with_json_format):
        """Test judging an output with JSON critique format."""
        # Create a prompt and output
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."

        # Judge the output
        result = await judge_agent_with_json_format.judge(output=output, prompt=prompt)

        # Check the result
        assert isinstance(result, JudgeResult)
        assert result.output == output
        assert result.prompt == prompt
        assert result.overall_score > 0.0
        assert len(result.dimension_scores) > 0
        assert result.critique != ""
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_format_critique(self, judge_agent):
        """Test formatting a critique."""
        # Create a judgment result
        result = JudgeResult(
            output="The capital of France is Paris.",
            prompt="What is the capital of France?",
            passed=True,
            overall_score=0.85,
            dimension_scores=[
                {
                    "dimension": ScoringDimension.RELEVANCE,
                    "score": 0.9,
                    "explanation": "The response is highly relevant to the prompt.",
                },
                {
                    "dimension": ScoringDimension.CORRECTNESS,
                    "score": 0.95,
                    "explanation": "The response is factually correct.",
                },
            ],
            critique="The response is excellent, providing a concise and accurate answer.",
            suggestions=["Add more context about Paris."],
        )

        # Format the critique
        critique = judge_agent.format_critique(result)

        # Check the critique
        assert isinstance(critique, str)
        assert "0.85" in critique
        assert "RELEVANCE" in critique.upper() or "Relevance" in critique
        assert "CORRECTNESS" in critique.upper() or "Correctness" in critique
        assert "excellent" in critique.lower()
        assert "Add more context" in critique

    @pytest.mark.asyncio
    async def test_calculate_overall_score(self, judge_agent):
        """Test calculating the overall score."""
        # Create dimension scores
        dimension_scores = [
            {"dimension": "relevance", "score": 0.8},
            {"dimension": "correctness", "score": 0.9},
            {"dimension": "coherence", "score": 0.7},
        ]

        # Calculate the overall score
        score = judge_agent._calculate_overall_score(dimension_scores)

        # Check the score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.8) < 1e-6  # (0.8 + 0.9 + 0.7) / 3

    @pytest.mark.asyncio
    async def test_parse_judgment_response(self, judge_agent):
        """Test parsing a judgment response."""
        # Test parsing a structured response
        structured_response = (
            "DIMENSION SCORES:\n"
            "- Relevance: 0.8\n"
            "  Explanation: The response is highly relevant to the prompt.\n"
            "- Correctness: 0.9\n"
            "  Explanation: The response is factually correct.\n\n"
            "OVERALL SCORE: 0.85\n\n"
            "CRITIQUE:\n"
            "The response is excellent.\n\n"
            "SUGGESTIONS:\n"
            "1. Add more context.\n"
            "2. Provide examples."
        )

        parsed = judge_agent._parse_judgment_response(structured_response)

        assert "dimension_scores" in parsed
        assert len(parsed["dimension_scores"]) == 2
        assert parsed["overall_score"] == 0.85
        assert "excellent" in parsed["critique"].lower()
        assert len(parsed["suggestions"]) == 2

        # Test parsing a JSON response
        json_response = (
            "```json\n"
            "{\n"
            '  "dimension_scores": [\n'
            '    {"dimension": "relevance", "score": 0.8, "explanation": "The response is highly relevant to the prompt."}, \n'
            '    {"dimension": "correctness", "score": 0.9, "explanation": "The response is factually correct."}\n'
            "  ],\n"
            '  "overall_score": 0.85,\n'
            '  "critique": "The response is excellent.",\n'
            '  "suggestions": ["Add more context.", "Provide examples."]\n'
            "}\n"
            "```\n"
        )

        judge_agent.config.critique_format = CritiqueFormat.JSON
        parsed = judge_agent._parse_judgment_response(json_response)

        assert "dimension_scores" in parsed
        assert len(parsed["dimension_scores"]) == 2
        assert parsed["overall_score"] == 0.85
        assert "excellent" in parsed["critique"].lower()
        assert len(parsed["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_scoring_accuracy(self, mock_judge_llm):
        """Test the accuracy of scoring against expected values."""
        # Create a judge agent with a specific threshold
        config = JudgeConfig(threshold=0.8)
        judge_agent = JudgeAgent(model=mock_judge_llm, config=config)

        # Test case 1: High-quality output (should pass)
        prompt = "What is the capital of France?"
        good_output = "The capital of France is Paris, which is located in the north-central part of the country."

        result = await judge_agent.judge(output=good_output, prompt=prompt)

        # Check that the result passes the threshold
        assert result.passed is True
        assert result.overall_score >= config.threshold

        # Test case 2: Low-quality output (should fail)
        # Create a custom mock LLM that returns a lower score
        class LowScoreMockLLM(MockJudgeLLM):
            async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
                response = await super().generate(prompt, max_tokens, temperature, **kwargs)
                if "You are a judge evaluating" in prompt:
                    # Override with a low score response
                    response.text = (
                        response.text.replace("0.8", "0.5")
                        .replace("0.9", "0.6")
                        .replace("0.7", "0.4")
                        .replace("0.85", "0.5")
                        .replace("0.81", "0.5")
                    )
                return response

        low_score_llm = LowScoreMockLLM("mock://judge-model/latest")
        low_score_judge = JudgeAgent(model=low_score_llm, config=config)

        bad_output = "Paris is in France."
        result = await low_score_judge.judge(output=bad_output, prompt=prompt)

        # Check that the result fails the threshold
        assert result.passed is False
        assert result.overall_score < config.threshold

        # Verify dimension scores are within expected ranges
        for dimension_score in result.dimension_scores:
            assert 0.0 <= dimension_score.score <= 1.0
            assert isinstance(dimension_score.dimension, ScoringDimension)

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, mock_judge_llm):
        """Test that budget constraints are enforced."""
        # Create a judge agent with budget constraints
        config = JudgeConfig(
            enforce_budget=True, max_tokens_per_judgment=1000, max_cost_per_judgment=0.05
        )
        judge_agent = JudgeAgent(model=mock_judge_llm, config=config)

        # Test with normal usage (should work)
        prompt = "What is the capital of France?"
        output = "The capital of France is Paris."

        result = await judge_agent.judge(output=output, prompt=prompt)
        assert isinstance(result, JudgeResult)

        # Check if the JudgeAgent implementation has budget enforcement
        # If it doesn't, we'll need to modify the implementation

        # Let's check if the _create_judgment_prompt method exists and is used in the judge method
        # This is a more direct test of the implementation

        # Create a judge agent with a mock model that tracks calls
        class TrackingMockLLM(MockJudgeLLM):
            def __init__(self, model_uri):
                super().__init__(model_uri)
                self.generate_calls = []

            async def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
                self.generate_calls.append((prompt, max_tokens))
                return await super().generate(prompt, max_tokens, temperature, **kwargs)

        tracking_llm = TrackingMockLLM("mock://judge-model/latest")
        tracking_judge = JudgeAgent(model=tracking_llm, config=config)

        # Judge an output
        await tracking_judge.judge(output=output, prompt=prompt)

        # Verify that max_tokens was passed to the model
        assert len(tracking_llm.generate_calls) > 0
        _, max_tokens = tracking_llm.generate_calls[0]
        assert max_tokens == config.max_tokens_per_judgment

        # Test with a very large output that would exceed token limits
        large_output = "The capital of France is Paris. " * 100

        # The judge should still work with the large output
        result = await judge_agent.judge(output=large_output, prompt=prompt)
        assert isinstance(result, JudgeResult)
