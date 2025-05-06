from __future__ import annotations

"""
FactualValidator plugin for Saplings.

This module provides a validator for checking factual accuracy of outputs
against reference documents.
"""


import logging
import re
from typing import TYPE_CHECKING

from saplings.core.plugin import PluginType
from saplings.retrieval import CascadeRetriever, RetrievalConfig
from saplings.validator.result import ValidationResult, ValidationStatus
from saplings.validator.validator import RuntimeValidator

if TYPE_CHECKING:
    from saplings.memory.document import Document

logger = logging.getLogger(__name__)


class FactualValidator(RuntimeValidator):
    """
    Validator for factual accuracy.

    This validator checks outputs for factual accuracy against reference documents.
    """

    def __init__(self) -> None:
        """Initialize the factual validator."""
        self.retriever = None

    @property
    def id(self) -> str:
        """ID of the validator."""
        return "factual_validator"

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "factual_validator"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "Validates outputs for factual accuracy against reference documents"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR

    async def validate_output(self, output: str, _prompt: str, **kwargs) -> ValidationResult:
        """
        Validate an output for factual accuracy.

        Args:
        ----
            output: Output to validate
            _prompt: Prompt that generated the output (unused)
            **kwargs: Additional validation parameters
                - memory_store: MemoryStore to use for retrieval
                - threshold: Similarity threshold for factual validation
                - max_statements: Maximum number of statements to check

        Returns:
        -------
            ValidationResult: Validation result

        """
        # Get memory store from kwargs
        memory_store = kwargs.get("memory_store")
        if memory_store is None:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.ERROR,
                message="No memory store provided for factual validation",
                metadata={"error": "missing_memory_store"},
            )

        # Initialize retriever if needed
        if self.retriever is None:
            config = RetrievalConfig()
            # Configure graph expansion
            config.graph.max_hops = 2
            # Configure max documents
            config.entropy.max_documents = 5
            self.retriever = CascadeRetriever(memory_store=memory_store, config=config)

        # Extract statements from the output
        statements = self._extract_statements(output)

        if not statements:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.WARNING,
                message="No statements found to validate",
                metadata={"statements_found": 0},
            )

        # Limit the number of statements to check
        max_statements = kwargs.get("max_statements", 10)
        if len(statements) > max_statements:
            statements = statements[:max_statements]

        # Check each statement
        threshold = kwargs.get("threshold", 0.7)
        issues = []

        for i, statement in enumerate(statements):
            # Retrieve relevant documents
            retrieval_result = self.retriever.retrieve(statement)
            relevant_docs = retrieval_result.documents

            if not relevant_docs:
                issues.append(
                    {
                        "statement_index": i,
                        "statement": statement,
                        "issue": "No relevant documents found to validate this statement",
                        "confidence": 0.0,
                    }
                )
                continue

            # Check if the statement is supported by the documents
            is_supported, confidence, supporting_doc = self._check_statement_support(
                statement, relevant_docs
            )

            if not is_supported or confidence < threshold:
                issues.append(
                    {
                        "statement_index": i,
                        "statement": statement,
                        "issue": "Statement not sufficiently supported by reference documents",
                        "confidence": confidence,
                        "best_match": supporting_doc.content if supporting_doc else None,
                    }
                )

        if issues:
            return ValidationResult(
                validator_id=self.id,
                status=ValidationStatus.FAILED,
                message=f"Found {len(issues)} statements with factual issues",
                metadata={
                    "statements_checked": len(statements),
                    "statements_with_issues": len(issues),
                    "issues": issues,
                    "threshold": threshold,
                },
            )

        return ValidationResult(
            validator_id=self.id,
            status=ValidationStatus.PASSED,
            message=f"All {len(statements)} statements passed factual validation",
            metadata={
                "statements_checked": len(statements),
                "threshold": threshold,
            },
        )

    def _extract_statements(self, text: str) -> list[str]:
        """
        Extract statements from text.

        Args:
        ----
            text: Text to extract statements from

        Returns:
        -------
            List[str]: List of statements

        """
        # Split text into sentences
        sentence_pattern = r"[.!?]\s+"
        sentences = re.split(sentence_pattern, text)

        # Filter out short sentences and clean up
        statements = []
        for sentence in sentences:
            # Clean up the sentence
            sentence = sentence.strip()

            # Skip short sentences
            if len(sentence) < 10:
                continue

            # Skip sentences that don't make factual claims
            if sentence.startswith(("I think", "In my opinion")):
                continue

            statements.append(sentence)

        return statements

    def _check_statement_support(
        self, statement: str, documents: list[Document]
    ) -> tuple[bool, float, Document | None]:
        """
        Check if a statement is supported by documents.

        Args:
        ----
            statement: Statement to check
            documents: Documents to check against

        Returns:
        -------
            Tuple[bool, float, Optional[Document]]: (is_supported, confidence, supporting_doc)

        """
        # Simple implementation: check for keyword overlap
        statement_words = set(statement.lower().split())

        best_match = None
        best_score = 0.0

        for doc in documents:
            doc_words = set(doc.content.lower().split())

            # Calculate Jaccard similarity
            intersection = len(statement_words.intersection(doc_words))
            union = len(statement_words.union(doc_words))

            score = 0.0 if union == 0 else intersection / union

            if score > best_score:
                best_score = score
                best_match = doc

        # Consider the statement supported if the best score is above 0.3
        is_supported = best_score > 0.3

        return is_supported, best_score, best_match
