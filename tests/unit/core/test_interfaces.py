from __future__ import annotations

"""
Tests for core interfaces.

This module tests the core interfaces to ensure they are properly defined.
"""


from saplings.api.core.interfaces import (
    JudgeConfig,
    JudgeResult,
    ValidationConfig,
    ValidationContext,
    ValidationResult,
)


def test_validation_config():
    """Test ValidationConfig class."""
    # Test default values
    config = ValidationConfig()
    assert config.validation_type == "general"
    assert config.criteria == []
    assert config.model_name is None
    assert config.provider is None
    assert config.threshold == 0.7

    # Test custom values
    config = ValidationConfig(
        validation_type="custom",
        criteria=["criteria1", "criteria2"],
        model_name="gpt-4",
        provider="openai",
        threshold=0.8,
    )
    assert config.validation_type == "custom"
    assert config.criteria == ["criteria1", "criteria2"]
    assert config.model_name == "gpt-4"
    assert config.provider == "openai"
    assert config.threshold == 0.8


def test_validation_result():
    """Test ValidationResult class."""
    # Test without details
    result = ValidationResult(
        is_valid=True,
        score=0.9,
        feedback="Good job!",
    )
    assert result.is_valid is True
    assert result.score == 0.9
    assert result.feedback == "Good job!"
    assert result.details is None

    # Test with details
    result = ValidationResult(
        is_valid=False,
        score=0.5,
        feedback="Needs improvement",
        details={"error_type": "format", "suggestions": ["Fix indentation"]},
    )
    assert result.is_valid is False
    assert result.score == 0.5
    assert result.feedback == "Needs improvement"
    assert result.details == {"error_type": "format", "suggestions": ["Fix indentation"]}


def test_validation_context():
    """Test ValidationContext class."""
    # Test default values
    context = ValidationContext()
    assert context.input_data == {}
    assert context.output_data is None
    assert context.trace_id is None

    # Test custom values
    context = ValidationContext(
        input_data={"prompt": "Hello"},
        output_data="Hi there!",
        trace_id="trace-123",
    )
    assert context.input_data == {"prompt": "Hello"}
    assert context.output_data == "Hi there!"
    assert context.trace_id == "trace-123"


def test_judge_config():
    """Test JudgeConfig class."""
    # Test default values
    config = JudgeConfig(model_name="gpt-4")
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.timeout is None
    assert config.additional_params is None

    # Test custom values
    config = JudgeConfig(
        model_name="claude-3",
        temperature=0.2,
        max_tokens=500,
        timeout=30.0,
        additional_params={"top_p": 0.95},
    )
    assert config.model_name == "claude-3"
    assert config.temperature == 0.2
    assert config.max_tokens == 500
    assert config.timeout == 30.0
    assert config.additional_params == {"top_p": 0.95}


def test_judge_result():
    """Test JudgeResult class."""
    # Test without details
    result = JudgeResult(
        score=0.85,
        feedback="Good response",
        strengths=["Clear", "Concise"],
        weaknesses=["Missing examples"],
    )
    assert result.score == 0.85
    assert result.feedback == "Good response"
    assert result.strengths == ["Clear", "Concise"]
    assert result.weaknesses == ["Missing examples"]
    assert result.details is None

    # Test with details
    result = JudgeResult(
        score=0.7,
        feedback="Acceptable response",
        strengths=["Accurate"],
        weaknesses=["Verbose", "Disorganized"],
        details={"dimension_scores": {"clarity": 0.8, "accuracy": 0.9}},
    )
    assert result.score == 0.7
    assert result.feedback == "Acceptable response"
    assert result.strengths == ["Accurate"]
    assert result.weaknesses == ["Verbose", "Disorganized"]
    assert result.details == {"dimension_scores": {"clarity": 0.8, "accuracy": 0.9}}
