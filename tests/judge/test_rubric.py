"""
Tests for the rubric module.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from saplings.judge.config import Rubric, RubricItem, ScoringDimension
from saplings.judge.rubric import RubricLoader, RubricTemplate, RubricValidator


class TestRubricLoader:
    """Tests for the RubricLoader class."""

    def test_load_from_template_general(self):
        """Test loading the general template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.GENERAL)

        assert rubric.name == "General Evaluation Rubric"
        assert len(rubric.items) == 4
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.CORRECTNESS for item in rubric.items)
        assert any(item.dimension == ScoringDimension.COHERENCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.HELPFULNESS for item in rubric.items)

    def test_load_from_template_code(self):
        """Test loading the code template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.CODE)

        assert rubric.name == "Code Evaluation Rubric"
        assert len(rubric.items) == 4
        assert any(item.dimension == ScoringDimension.CORRECTNESS for item in rubric.items)
        assert any(item.dimension == ScoringDimension.COHERENCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.SAFETY for item in rubric.items)

        # Check that correctness has a higher weight
        correctness_item = next(
            item for item in rubric.items if item.dimension == ScoringDimension.CORRECTNESS
        )
        assert correctness_item.weight > 1.0

    def test_load_from_template_creative(self):
        """Test loading the creative template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.CREATIVE)

        assert rubric.name == "Creative Writing Rubric"
        assert len(rubric.items) == 4
        assert any(item.dimension == ScoringDimension.CREATIVITY for item in rubric.items)
        assert any(item.dimension == ScoringDimension.COHERENCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.SAFETY for item in rubric.items)

        # Check that creativity has a higher weight
        creativity_item = next(
            item for item in rubric.items if item.dimension == ScoringDimension.CREATIVITY
        )
        assert creativity_item.weight > 1.0

    def test_load_from_template_educational(self):
        """Test loading the educational template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.EDUCATIONAL)

        assert rubric.name == "Educational Content Rubric"
        assert len(rubric.items) == 4
        assert any(item.dimension == ScoringDimension.CORRECTNESS for item in rubric.items)
        assert any(item.dimension == ScoringDimension.HELPFULNESS for item in rubric.items)
        assert any(item.dimension == ScoringDimension.COHERENCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)

        # Check that correctness has a higher weight
        correctness_item = next(
            item for item in rubric.items if item.dimension == ScoringDimension.CORRECTNESS
        )
        assert correctness_item.weight > 1.0

    def test_load_from_template_factual(self):
        """Test loading the factual template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.FACTUAL)

        assert rubric.name == "Factual Content Rubric"
        assert len(rubric.items) == 4
        assert any(item.dimension == ScoringDimension.CORRECTNESS for item in rubric.items)
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.COHERENCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.CONCISENESS for item in rubric.items)

        # Check that correctness has a much higher weight
        correctness_item = next(
            item for item in rubric.items if item.dimension == ScoringDimension.CORRECTNESS
        )
        assert correctness_item.weight > 2.0

    def test_load_from_template_safety(self):
        """Test loading the safety template."""
        rubric = RubricLoader.load_from_template(RubricTemplate.SAFETY)

        assert rubric.name == "Safety Evaluation Rubric"
        assert len(rubric.items) == 3
        assert any(item.dimension == ScoringDimension.SAFETY for item in rubric.items)
        assert any(item.dimension == ScoringDimension.RELEVANCE for item in rubric.items)
        assert any(item.dimension == ScoringDimension.HELPFULNESS for item in rubric.items)

        # Check that safety has a much higher weight
        safety_item = next(
            item for item in rubric.items if item.dimension == ScoringDimension.SAFETY
        )
        assert safety_item.weight > 2.0

    def test_load_from_template_string(self):
        """Test loading a template from a string."""
        rubric = RubricLoader.load_from_template("code")

        assert rubric.name == "Code Evaluation Rubric"
        assert len(rubric.items) == 4

    def test_load_from_template_invalid(self):
        """Test loading an invalid template."""
        with pytest.raises(ValueError):
            RubricLoader.load_from_template("invalid_template")

    def test_load_from_file_yaml(self):
        """Test loading a rubric from a YAML file."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(
                {
                    "name": "Test Rubric",
                    "description": "Test rubric for testing",
                    "items": [
                        {
                            "dimension": "relevance",
                            "weight": 1.0,
                            "description": "How relevant the output is",
                            "criteria": {
                                "0.0": "Not relevant",
                                "1.0": "Very relevant",
                            },
                        },
                        {
                            "dimension": "correctness",
                            "weight": 2.0,
                            "description": "How correct the output is",
                            "criteria": {
                                "0.0": "Not correct",
                                "1.0": "Very correct",
                            },
                        },
                    ],
                },
                f,
            )

        try:
            # Load the rubric
            rubric = RubricLoader.load_from_file(f.name)

            # Check the rubric
            assert rubric.name == "Test Rubric"
            assert rubric.description == "Test rubric for testing"
            assert len(rubric.items) == 2
            assert rubric.items[0].dimension == ScoringDimension.RELEVANCE
            assert rubric.items[0].weight == 1.0
            assert rubric.items[1].dimension == ScoringDimension.CORRECTNESS
            assert rubric.items[1].weight == 2.0
        finally:
            # Clean up
            os.unlink(f.name)

    def test_load_from_file_json(self):
        """Test loading a rubric from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                {
                    "name": "Test Rubric",
                    "description": "Test rubric for testing",
                    "items": [
                        {
                            "dimension": "relevance",
                            "weight": 1.0,
                            "description": "How relevant the output is",
                            "criteria": {
                                "0.0": "Not relevant",
                                "1.0": "Very relevant",
                            },
                        },
                        {
                            "dimension": "correctness",
                            "weight": 2.0,
                            "description": "How correct the output is",
                            "criteria": {
                                "0.0": "Not correct",
                                "1.0": "Very correct",
                            },
                        },
                    ],
                },
                f,
            )

        try:
            # Load the rubric
            rubric = RubricLoader.load_from_file(f.name)

            # Check the rubric
            assert rubric.name == "Test Rubric"
            assert rubric.description == "Test rubric for testing"
            assert len(rubric.items) == 2
            assert rubric.items[0].dimension == ScoringDimension.RELEVANCE
            assert rubric.items[0].weight == 1.0
            assert rubric.items[1].dimension == ScoringDimension.CORRECTNESS
            assert rubric.items[1].weight == 2.0
        finally:
            # Clean up
            os.unlink(f.name)

    def test_load_from_file_not_found(self):
        """Test loading a rubric from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            RubricLoader.load_from_file("non_existent_file.yaml")

    def test_load_from_file_invalid_format(self):
        """Test loading a rubric from a file with an invalid format."""
        # Create a temporary file with an invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("This is not a valid rubric file")

        try:
            # Try to load the rubric
            with pytest.raises(ValueError):
                RubricLoader.load_from_file(f.name)
        finally:
            # Clean up
            os.unlink(f.name)

    def test_load_from_file_invalid_yaml(self):
        """Test loading a rubric from an invalid YAML file."""
        # Create a temporary YAML file with invalid YAML
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("This is not valid YAML: :")

        try:
            # Try to load the rubric
            with pytest.raises(ValueError):
                RubricLoader.load_from_file(f.name)
        finally:
            # Clean up
            os.unlink(f.name)

    def test_load_from_file_invalid_json(self):
        """Test loading a rubric from an invalid JSON file."""
        # Create a temporary JSON file with invalid JSON
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("This is not valid JSON: {")

        try:
            # Try to load the rubric
            with pytest.raises(ValueError):
                RubricLoader.load_from_file(f.name)
        finally:
            # Clean up
            os.unlink(f.name)

    def test_load_from_file_invalid_rubric(self):
        """Test loading an invalid rubric from a file."""
        # Create a temporary YAML file with an invalid rubric
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(
                {
                    # Missing name
                    "description": "Test rubric for testing",
                    "items": [
                        {
                            "dimension": "relevance",
                            "weight": 1.0,
                            "description": "How relevant the output is",
                            "criteria": {
                                "0.0": "Not relevant",
                                "1.0": "Very relevant",
                            },
                        },
                    ],
                },
                f,
            )

        try:
            # Try to load the rubric
            with pytest.raises(ValueError):
                RubricLoader.load_from_file(f.name)
        finally:
            # Clean up
            os.unlink(f.name)


class TestRubricValidator:
    """Tests for the RubricValidator class."""

    def test_validate_valid_rubric(self):
        """Test validating a valid rubric."""
        # Create a valid rubric
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is",
                    criteria={
                        "0.0": "Not relevant",
                        "1.0": "Very relevant",
                    },
                ),
                RubricItem(
                    dimension=ScoringDimension.CORRECTNESS,
                    weight=2.0,
                    description="How correct the output is",
                    criteria={
                        "0.0": "Not correct",
                        "1.0": "Very correct",
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there are no errors
        assert len(errors) == 0

    def test_validate_no_name(self):
        """Test validating a rubric with no name."""
        # Create a rubric with no name
        rubric = Rubric(
            name="",  # Empty name
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is",
                    criteria={
                        "0.0": "Not relevant",
                        "1.0": "Very relevant",
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the name
        assert len(errors) == 1
        assert "name" in errors[0].lower()

    def test_validate_no_items(self):
        """Test validating a rubric with no items."""
        # Create a rubric with no items
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[],  # Empty items
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the items
        assert len(errors) == 1
        assert "item" in errors[0].lower()

    def test_validate_invalid_weight(self):
        """Test validating a rubric with an invalid weight."""
        # Create a rubric with an item with an invalid weight
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=0.0,  # Invalid weight (must be positive)
                    description="How relevant the output is",
                    criteria={
                        "0.0": "Not relevant",
                        "1.0": "Very relevant",
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the weight
        assert len(errors) == 1
        assert "weight" in errors[0].lower()

    def test_validate_no_criteria(self):
        """Test validating a rubric with an item with no criteria."""
        # Create a rubric with an item with no criteria
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is",
                    criteria={},  # Empty criteria
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the criteria
        assert len(errors) == 1
        assert "criterion" in errors[0].lower() or "criteria" in errors[0].lower()

    def test_validate_invalid_score(self):
        """Test validating a rubric with an item with an invalid score."""
        # Create a rubric with an item with an invalid score
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is",
                    criteria={
                        "invalid": "Not relevant",  # Invalid score (not a number)
                        "1.0": "Very relevant",
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the score
        assert len(errors) == 1
        assert "score" in errors[0].lower()

        # Create a rubric with an item with a score out of range
        rubric = Rubric(
            name="Test Rubric",
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=1.0,
                    description="How relevant the output is",
                    criteria={
                        "2.0": "Not relevant",  # Invalid score (out of range)
                        "1.0": "Very relevant",
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there is an error about the score
        assert len(errors) == 1
        assert "score" in errors[0].lower()

    def test_validate_multiple_errors(self):
        """Test validating a rubric with multiple errors."""
        # Create a rubric with multiple errors
        rubric = Rubric(
            name="",  # Empty name
            description="Test rubric for testing",
            items=[
                RubricItem(
                    dimension=ScoringDimension.RELEVANCE,
                    weight=0.0,  # Invalid weight
                    description="How relevant the output is",
                    criteria={
                        "invalid": "Not relevant",  # Invalid score
                    },
                ),
            ],
        )

        # Validate the rubric
        errors = RubricValidator.validate(rubric)

        # Check that there are multiple errors
        assert len(errors) == 3
        assert any("name" in error.lower() for error in errors)
        assert any("weight" in error.lower() for error in errors)
        assert any("score" in error.lower() for error in errors)
