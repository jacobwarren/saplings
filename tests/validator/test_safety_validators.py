"""
Tests for the safety validators.
"""

import pytest

from saplings.validator.validator import ValidationStatus
from saplings.validator.validators.safety import PiiValidator, ProfanityValidator


class TestProfanityValidator:
    """Tests for the ProfanityValidator."""

    @pytest.mark.asyncio
    async def test_no_profanity(self):
        """Test validating output with no profanity."""
        # Create a validator
        validator = ProfanityValidator()

        # Test with a clean output
        result = await validator.validate_output(
            output="This is a clean and appropriate output.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        assert result.details["profanity_score"] == 0.0

    @pytest.mark.asyncio
    async def test_with_profanity(self):
        """Test validating output with profanity."""
        # Create a validator with a custom profanity list
        validator = ProfanityValidator(
            custom_profanity_list=["inappropriate", "offensive", "rude"],
        )

        # Test with an output containing profanity
        result = await validator.validate_output(
            output="This output contains inappropriate and offensive language.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert result.details["profanity_score"] > 0.0
        assert "inappropriate" in result.details["profanity_words"]
        assert "offensive" in result.details["profanity_words"]

    @pytest.mark.asyncio
    async def test_with_threshold(self):
        """Test validating output with a profanity threshold."""
        # Create a validator with a threshold
        validator = ProfanityValidator(
            custom_profanity_list=["inappropriate", "offensive", "rude"],
            threshold=0.2,  # Allow up to 20% profanity
        )

        # Test with an output containing a small amount of profanity
        result = await validator.validate_output(
            output="This output has one inappropriate word but is otherwise fine.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED
        assert result.details["profanity_score"] <= 0.2

        # Test with an output containing a large amount of profanity
        result = await validator.validate_output(
            output="This inappropriate output is offensive and rude.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert result.details["profanity_score"] > 0.2


class TestPiiValidator:
    """Tests for the PiiValidator."""

    @pytest.mark.asyncio
    async def test_no_pii(self):
        """Test validating output with no PII."""
        # Create a validator
        validator = PiiValidator()

        # Test with a clean output
        result = await validator.validate_output(
            output="This is a clean output with no personal information.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_with_email(self):
        """Test validating output with an email address."""
        # Create a validator that checks emails
        validator = PiiValidator(check_emails=True)

        # Test with an output containing an email
        result = await validator.validate_output(
            output="Contact me at user@example.com for more information.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "email" in result.details["pii_types"]
        assert "user@example.com" in result.details["found_pii"]["email"]

    @pytest.mark.asyncio
    async def test_with_phone_number(self):
        """Test validating output with a phone number."""
        # Create a validator that checks phone numbers
        validator = PiiValidator(check_phone_numbers=True)

        # Test with an output containing a phone number
        result = await validator.validate_output(
            output="Call me at (123) 456-7890 for more information.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "phone" in result.details["pii_types"]

        # Test with different phone number formats
        result = await validator.validate_output(
            output="Call me at 123-456-7890 or +1 987-654-3210.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "phone" in result.details["pii_types"]
        assert len(result.details["found_pii"]["phone"]) == 2

    @pytest.mark.asyncio
    async def test_with_credit_card(self):
        """Test validating output with a credit card number."""
        # Create a validator that checks credit card numbers
        validator = PiiValidator(check_credit_cards=True)

        # Test with an output containing a credit card number
        result = await validator.validate_output(
            output="My credit card number is 1234-5678-9012-3456.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "credit_card" in result.details["pii_types"]

    @pytest.mark.asyncio
    async def test_with_ssn(self):
        """Test validating output with a Social Security Number."""
        # Create a validator that checks SSNs
        validator = PiiValidator(check_ssns=True)

        # Test with an output containing an SSN
        result = await validator.validate_output(
            output="My SSN is 123-45-6789.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "ssn" in result.details["pii_types"]

    @pytest.mark.asyncio
    async def test_with_custom_pattern(self):
        """Test validating output with a custom pattern."""
        # Create a validator with a custom pattern for passport numbers
        validator = PiiValidator(
            custom_patterns={
                "passport": r"\b[A-Z]{1,2}\d{6,9}\b",  # Simple passport number pattern
            },
        )

        # Test with an output containing a passport number
        result = await validator.validate_output(
            output="My passport number is AB123456.",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert "passport" in result.details["pii_types"]

    @pytest.mark.asyncio
    async def test_with_multiple_pii_types(self):
        """Test validating output with multiple PII types."""
        # Create a validator that checks multiple PII types
        validator = PiiValidator(
            check_emails=True,
            check_phone_numbers=True,
            check_credit_cards=True,
            check_ssns=True,
        )

        # Test with an output containing multiple PII types
        result = await validator.validate_output(
            output="Contact: user@example.com, Phone: 123-456-7890, CC: 1234-5678-9012-3456, SSN: 123-45-6789",
            prompt="Test prompt",
        )
        assert result.status == ValidationStatus.FAILED
        assert len(result.details["pii_types"]) == 4
        assert "email" in result.details["pii_types"]
        assert "phone" in result.details["pii_types"]
        assert "credit_card" in result.details["pii_types"]
        assert "ssn" in result.details["pii_types"]
