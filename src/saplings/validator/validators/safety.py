"""
Safety validators for Saplings.

This module provides safety validators for Saplings.
"""

import re
from typing import Dict, List, Optional, Set, Union

from saplings.validator.validator import (
    RuntimeValidator,
    StaticValidator,
    ValidationResult,
    ValidationStatus,
)


class ProfanityValidator(RuntimeValidator):
    """
    Validator that checks for profanity in the output.

    This validator checks if the output contains profanity.
    """

    def __init__(
        self,
        custom_profanity_list: Optional[List[str]] = None,
        threshold: float = 0.0,
    ):
        """
        Initialize the profanity validator.

        Args:
            custom_profanity_list: Custom list of profanity words
            threshold: Threshold for profanity detection (0.0 = any profanity fails)
        """
        self._threshold = threshold

        # Default profanity list (very limited for demonstration purposes)
        self._profanity_list = {
            "badword1", "badword2", "badword3",
        }

        # Add custom profanity words
        if custom_profanity_list:
            self._profanity_list.update(custom_profanity_list)

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "profanity_validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "Profanity Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "Validates that the output does not contain profanity"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate the output for profanity.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        # Simple profanity detection based on word matching
        # In a real implementation, you would use a more sophisticated approach
        words = re.findall(r'\b\w+\b', output.lower())
        profanity_words = [word for word in words if word in self._profanity_list]

        # Calculate profanity score
        if words:
            profanity_score = len(profanity_words) / len(words)
        else:
            profanity_score = 0.0

        # Check if the profanity score is below the threshold
        if profanity_score <= self._threshold:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message=f"Output does not contain significant profanity (score: {profanity_score:.2f})",
                details={
                    "profanity_score": profanity_score,
                    "threshold": self._threshold,
                    "profanity_words": profanity_words,
                },
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message=f"Output contains profanity (score: {profanity_score:.2f})",
                details={
                    "profanity_score": profanity_score,
                    "threshold": self._threshold,
                    "profanity_words": profanity_words,
                },
            )


class PiiValidator(RuntimeValidator):
    """
    Validator that checks for personally identifiable information (PII) in the output.

    This validator checks if the output contains PII.
    """

    def __init__(
        self,
        check_emails: bool = True,
        check_phone_numbers: bool = True,
        check_credit_cards: bool = True,
        check_ssns: bool = True,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the PII validator.

        Args:
            check_emails: Whether to check for email addresses
            check_phone_numbers: Whether to check for phone numbers
            check_credit_cards: Whether to check for credit card numbers
            check_ssns: Whether to check for Social Security Numbers
            custom_patterns: Custom regex patterns to check for
        """
        self._check_emails = check_emails
        self._check_phone_numbers = check_phone_numbers
        self._check_credit_cards = check_credit_cards
        self._check_ssns = check_ssns
        self._custom_patterns = custom_patterns or {}

        # Compile regex patterns
        self._patterns = {}

        if check_emails:
            self._patterns["email"] = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        if check_phone_numbers:
            self._patterns["phone"] = re.compile(r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')

        if check_credit_cards:
            self._patterns["credit_card"] = re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')

        if check_ssns:
            self._patterns["ssn"] = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

        # Add custom patterns
        for name, pattern in self._custom_patterns.items():
            self._patterns[name] = re.compile(pattern)

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "pii_validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "PII Validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the validator."""
        return "Validates that the output does not contain personally identifiable information (PII)"

    async def validate_output(self, output: str, prompt: str, **kwargs) -> ValidationResult:
        """
        Validate the output for PII.

        Args:
            output: Output to validate
            prompt: Prompt that generated the output
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation result
        """
        # Check for PII using regex patterns
        found_pii = {}

        for name, pattern in self._patterns.items():
            matches = pattern.findall(output)
            if matches:
                found_pii[name] = matches

        # Check if any PII was found
        if found_pii:
            pii_types = list(found_pii.keys())
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message=f"Output contains PII: {', '.join(pii_types)}",
                details={
                    "found_pii": found_pii,
                    "pii_types": pii_types,
                },
            )
        else:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.PASSED,
                message="Output does not contain PII",
                details={
                    "checked_types": list(self._patterns.keys()),
                },
            )
