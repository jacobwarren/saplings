from __future__ import annotations

"""
Unit tests for the validator service.
"""


from unittest.mock import MagicMock

from saplings.core.interfaces import IModelService, IValidatorService
from saplings.services.validator_service import ValidatorService
from saplings.validator.config import ValidatorConfig
from saplings.validator.result import ValidationResult, ValidationStatus


class TestValidatorService:
    THRESHOLD_1 = 0.3
    THRESHOLD_2 = 0.65
    THRESHOLD_3 = 0.7
    THRESHOLD_4 = 0.95

    """Test the validator service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock model service
        self.mock_model_service = MagicMock(spec=IModelService)

        # Mock the model response
        mock_response = MagicMock()
        mock_response.text = '{"valid": true, "score": 0.95, "feedback": "Good response"}'
        self.mock_model_service.generate.return_value = mock_response

        # Create validator service
        self.config = ValidatorConfig(
            threshold=0.7, validation_model_provider="test", validation_model_name="test-validator"
        )
        self.service = ValidatorService(model_service=self.mock_model_service, config=self.config)

    def test_initialization(self) -> None:
        """Test validator service initialization."""
        assert self.service.model_service is self.mock_model_service
        assert self.service.config is self.config
        assert self.service.config.threshold == self.THRESHOLD_3
        assert self.service.config.validation_model_provider == "test"
        assert self.service.config.validation_model_name == "test-validator"

    def test_validate_response(self) -> None:
        """Test validate_response method."""
        # Validate a response
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            context=["France is a country in Europe."],
        )

        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.score == self.THRESHOLD_4
        assert result.feedback == "Good response"
        assert result.status == ValidationStatus.PASSED

        # Verify model service was called
        self.mock_model_service.generate.assert_called_once()

    def test_validate_response_invalid(self) -> None:
        """Test validate_response method with invalid response."""
        # Mock an invalid response
        mock_response = MagicMock()
        mock_response.text = '{"valid": false, "score": 0.3, "feedback": "Incorrect information"}'
        self.mock_model_service.generate.return_value = mock_response

        # Validate a response
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is London.",
            context=["France is a country in Europe."],
        )

        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.valid is False
        assert result.score == self.THRESHOLD_1
        assert result.feedback == "Incorrect information"
        assert result.status == ValidationStatus.FAILED

    def test_validate_response_with_criteria(self) -> None:
        """Test validate_response method with custom criteria."""
        # Validate a response with custom criteria
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            context=["France is a country in Europe."],
            criteria=["Accuracy", "Completeness", "Relevance"],
        )

        # Verify result
        assert isinstance(result, ValidationResult)

        # Verify model service was called with criteria
        call_args = self.mock_model_service.generate.call_args[0][0]
        assert "Accuracy" in call_args
        assert "Completeness" in call_args
        assert "Relevance" in call_args

    def test_validate_response_with_threshold(self) -> None:
        """Test validate_response method with custom threshold."""
        # Mock a borderline response
        mock_response = MagicMock()
        mock_response.text = '{"valid": true, "score": 0.65, "feedback": "Acceptable response"}'
        self.mock_model_service.generate.return_value = mock_response

        # Validate with default threshold (0.7)
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            context=["France is a country in Europe."],
        )

        # Should fail with default threshold
        assert result.valid is True  # From the mock
        assert result.score == self.THRESHOLD_2
        assert result.status == ValidationStatus.FAILED  # Below threshold

        # Validate with custom threshold (0.6)
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            context=["France is a country in Europe."],
            threshold=0.6,
        )

        # Should pass with lower threshold
        assert result.valid is True
        assert result.score == self.THRESHOLD_2
        assert result.status == ValidationStatus.PASSED  # Above custom threshold

    def test_validate_response_with_invalid_json(self) -> None:
        """Test validate_response method with invalid JSON response."""
        # Mock an invalid JSON response
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        self.mock_model_service.generate.return_value = mock_response

        # Validate a response
        result = self.service.validate_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            context=["France is a country in Europe."],
        )

        # Verify result indicates error
        assert isinstance(result, ValidationResult)
        assert result.valid is False
        assert result.status == ValidationStatus.ERROR
        assert "Error parsing validation response" in result.feedback

    def test_interface_compliance(self) -> None:
        """Test that ValidatorService implements IValidatorService."""
        assert isinstance(self.service, IValidatorService)

        # Check required methods
        assert hasattr(self.service, "validate_response")
