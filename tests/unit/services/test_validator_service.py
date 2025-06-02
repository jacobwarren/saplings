from __future__ import annotations

"""
Unit tests for the validator service.
"""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saplings.core.exceptions import ValidationError

# Import necessary components
from saplings.services.validator_service import ValidatorService


class TestValidatorService:
    """Test the validator service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock model
        self.mock_model = MagicMock()

        # Create mock validator registry
        self.mock_validator_registry = MagicMock()
        self.mock_validator = MagicMock()
        self.mock_validator.validate = AsyncMock()
        self.mock_validator_registry.get_validator.return_value = self.mock_validator

        # Create validator service with patch for validator registry
        with patch(
            "saplings.services.validator_service.get_validator_registry",
            return_value=self.mock_validator_registry,
        ):
            self.service = ValidatorService(model=self.mock_model)

        # Manually set the validator_registry to our mock for testing
        self.service.validator_registry = self.mock_validator_registry
        self.service.validator = self.mock_validator

    def test_initialization(self) -> None:
        """Test validator service initialization."""
        assert self.service._model is self.mock_model
        assert self.service.validator_registry is self.mock_validator_registry
        assert self.service.validator is self.mock_validator
        assert self.service._judge_service is None

    @pytest.mark.asyncio()
    async def test_validate(self) -> None:
        """Test validate method."""
        # Setup mock validation strategy
        mock_validation_result = MagicMock()
        mock_validation_result.is_valid = True
        mock_validation_result.score = 0.9
        mock_validation_result.feedback = "Good response"
        mock_validation_result.details = {"key": "value"}

        # Create a mock validation strategy
        mock_strategy = MagicMock()
        mock_strategy.validate = AsyncMock(return_value=mock_validation_result)

        # Set the validation strategy
        self.service._validation_strategy = mock_strategy

        # Call validate
        input_data = {"prompt": "What is the capital of France?"}
        output_data = "The capital of France is Paris."

        result = await self.service.validate(
            input_data=input_data, output_data=output_data, validation_type="execution"
        )

        # Verify result structure
        assert result["is_valid"] == True
        assert result["score"] == 0.9
        assert result["feedback"] == "Good response"
        assert result["details"] == {"key": "value"}

        # Verify validation strategy was called
        mock_strategy.validate.assert_called_once_with(
            input_data=input_data,
            output_data=output_data,
            validation_type="execution",
            trace_id=None,
        )

    @pytest.mark.asyncio()
    async def test_judge_output_without_judge_service(self) -> None:
        """Test judge_output method when judge_service is not set."""
        # Call judge_output without setting judge_service
        input_data = {"prompt": "What is the capital of France?"}
        output_data = "The capital of France is Paris."

        # Mock the validation strategy to raise an exception
        self.service._validation_strategy = MagicMock()
        self.service._validation_strategy.validate = AsyncMock(
            side_effect=Exception("Strategy failed")
        )

        # Ensure judge_service is None
        self.service._judge_service = None

        # Should raise ValidationError since there's no judge_service and strategy fails
        with pytest.raises(ValidationError):
            await self.service.judge_output(input_data=input_data, output_data=output_data)

    @pytest.mark.asyncio()
    async def test_judge_output_with_judge_service(self) -> None:
        """Test judge_output method with judge_service set."""
        # Setup mock judge service
        mock_judge_service = MagicMock()
        mock_judgment = MagicMock()
        mock_judgment.score = 0.95
        mock_judgment.feedback = "Good response"
        mock_judgment.strengths = ["Accurate", "Clear"]
        mock_judgment.weaknesses = ["Could be more detailed"]
        mock_judge_service.judge = AsyncMock(return_value=mock_judgment)

        # Set judge_service
        self.service._judge_service = mock_judge_service

        # Mock the validation strategy to raise an exception to force using judge_service
        self.service._validation_strategy = MagicMock()
        self.service._validation_strategy.validate = AsyncMock(
            side_effect=Exception("Strategy failed")
        )

        # Call judge_output
        input_data = {"prompt": "What is the capital of France?"}
        output_data = "The capital of France is Paris."

        result = await self.service.judge_output(
            input_data=input_data, output_data=output_data, judgment_type="general"
        )

        # Verify result
        assert result["score"] == 0.95
        assert result["feedback"] == "Good response"
        assert result["strengths"] == ["Accurate", "Clear"]
        assert result["weaknesses"] == ["Could be more detailed"]

        # Verify judge_service was called
        mock_judge_service.judge.assert_called_once_with(
            input_data=input_data, output_data=output_data, judgment_type="general", trace_id=None
        )

    def test_get_validation_history(self) -> None:
        """Test get_validation_history method."""
        # Call get_validation_history (not async)
        history = self.service.get_validation_history()

        # Verify result (empty list in current implementation)
        assert history == []

    def test_interface_compliance(self) -> None:
        """Test that ValidatorService has the required methods."""
        # Check required methods
        assert hasattr(self.service, "validate")
        assert hasattr(self.service, "judge_output")
        assert hasattr(self.service, "get_validation_history")
        assert hasattr(self.service, "validate_with_rubric")

        # Verify judge_service attribute exists
        assert hasattr(self.service, "_judge_service")
