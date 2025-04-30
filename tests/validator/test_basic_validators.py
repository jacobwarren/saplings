"""
Tests for the basic validators.
"""

import pytest

from saplings.validator.validator import ValidationStatus
from saplings.validator.validators.basic import (
    LengthValidator,
    KeywordValidator,
    SentimentValidator,
)


class TestLengthValidator:
    """Tests for the LengthValidator."""
    
    @pytest.mark.asyncio
    async def test_character_length_validation(self):
        """Test validating character length."""
        # Create a validator with min and max length
        validator = LengthValidator(min_length=10, max_length=20, unit="characters")
        
        # Test with a valid output
        result = await validator.validate_output(
            output="This is valid",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that's too short
        result = await validator.validate_output(
            output="Too short",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "less than the minimum" in result.message
        
        # Test with an output that's too long
        result = await validator.validate_output(
            output="This output is way too long for the validator",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "greater than the maximum" in result.message
    
    @pytest.mark.asyncio
    async def test_word_length_validation(self):
        """Test validating word length."""
        # Create a validator with min and max length
        validator = LengthValidator(min_length=3, max_length=5, unit="words")
        
        # Test with a valid output
        result = await validator.validate_output(
            output="This is a valid output",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that's too short
        result = await validator.validate_output(
            output="Too short",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "less than the minimum" in result.message
        
        # Test with an output that's too long
        result = await validator.validate_output(
            output="This output is way too long for the validator to accept",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "greater than the maximum" in result.message
    
    @pytest.mark.asyncio
    async def test_sentence_length_validation(self):
        """Test validating sentence length."""
        # Create a validator with min and max length
        validator = LengthValidator(min_length=2, max_length=3, unit="sentences")
        
        # Test with a valid output
        result = await validator.validate_output(
            output="This is the first sentence. This is the second sentence.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that's too short
        result = await validator.validate_output(
            output="This is the only sentence.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "less than the minimum" in result.message
        
        # Test with an output that's too long
        result = await validator.validate_output(
            output="This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "greater than the maximum" in result.message
    
    @pytest.mark.asyncio
    async def test_unknown_unit(self):
        """Test validating with an unknown unit."""
        # Create a validator with an unknown unit
        validator = LengthValidator(min_length=10, max_length=20, unit="paragraphs")
        
        # Test with any output
        result = await validator.validate_output(
            output="This is a test output",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.ERROR
        assert "Unknown unit" in result.message


class TestKeywordValidator:
    """Tests for the KeywordValidator."""
    
    @pytest.mark.asyncio
    async def test_required_keywords(self):
        """Test validating required keywords."""
        # Create a validator with required keywords
        validator = KeywordValidator(
            required_keywords=["important", "critical", "essential"],
        )
        
        # Test with an output that contains all required keywords
        result = await validator.validate_output(
            output="This is an important and critical output with essential information.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that's missing some required keywords
        result = await validator.validate_output(
            output="This is an important output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "Missing required keywords" in result.message
        assert "critical" in result.details["missing_keywords"]
        assert "essential" in result.details["missing_keywords"]
    
    @pytest.mark.asyncio
    async def test_forbidden_keywords(self):
        """Test validating forbidden keywords."""
        # Create a validator with forbidden keywords
        validator = KeywordValidator(
            forbidden_keywords=["bad", "inappropriate", "offensive"],
        )
        
        # Test with an output that doesn't contain any forbidden keywords
        result = await validator.validate_output(
            output="This is a good and appropriate output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that contains forbidden keywords
        result = await validator.validate_output(
            output="This is a bad and inappropriate output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "Found forbidden keywords" in result.message
        assert "bad" in result.details["found_forbidden_keywords"]
        assert "inappropriate" in result.details["found_forbidden_keywords"]
    
    @pytest.mark.asyncio
    async def test_case_sensitivity(self):
        """Test case sensitivity in keyword validation."""
        # Create a case-sensitive validator
        validator = KeywordValidator(
            required_keywords=["Important", "Critical"],
            forbidden_keywords=["Bad", "Inappropriate"],
            case_sensitive=True,
        )
        
        # Test with an output that matches the case
        result = await validator.validate_output(
            output="This is an Important and Critical output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        
        # Test with an output that doesn't match the case
        result = await validator.validate_output(
            output="This is an important and critical output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "Missing required keywords" in result.message
        
        # Test with an output that contains forbidden keywords with different case
        result = await validator.validate_output(
            output="This is an Important output but it's bad.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "Missing required keywords" in result.message
        assert "Found forbidden keywords" not in result.message


class TestSentimentValidator:
    """Tests for the SentimentValidator."""
    
    @pytest.mark.asyncio
    async def test_positive_sentiment(self):
        """Test validating positive sentiment."""
        # Create a validator for positive sentiment
        validator = SentimentValidator(desired_sentiment="positive", threshold=0.1)
        
        # Test with a positive output
        result = await validator.validate_output(
            output="This is a great and wonderful product. I love it!",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        assert result.details["actual_sentiment"] == "positive"
        
        # Test with a negative output
        result = await validator.validate_output(
            output="This is a terrible product. I hate it!",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert result.details["actual_sentiment"] == "negative"
    
    @pytest.mark.asyncio
    async def test_negative_sentiment(self):
        """Test validating negative sentiment."""
        # Create a validator for negative sentiment
        validator = SentimentValidator(desired_sentiment="negative", threshold=0.1)
        
        # Test with a negative output
        result = await validator.validate_output(
            output="This is a terrible product. I hate it!",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        assert result.details["actual_sentiment"] == "negative"
        
        # Test with a positive output
        result = await validator.validate_output(
            output="This is a great and wonderful product. I love it!",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert result.details["actual_sentiment"] == "positive"
    
    @pytest.mark.asyncio
    async def test_neutral_sentiment(self):
        """Test validating neutral sentiment."""
        # Create a validator for neutral sentiment
        validator = SentimentValidator(desired_sentiment="neutral", threshold=0.5)
        
        # Test with a neutral output
        result = await validator.validate_output(
            output="This is a product with some features.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        assert result.details["actual_sentiment"] == "neutral"
        
        # Test with a positive output
        result = await validator.validate_output(
            output="This is a great and wonderful product. I love it!",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert result.details["actual_sentiment"] == "positive"
