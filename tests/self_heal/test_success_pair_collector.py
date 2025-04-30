"""
Tests for the SuccessPairCollector class.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock

from saplings.self_heal.patch_generator import Patch, PatchStatus
from saplings.self_heal.success_pair_collector import SuccessPairCollector


class TestSuccessPairCollector:
    """Tests for the SuccessPairCollector class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def collector(self, temp_dir):
        """Create a SuccessPairCollector instance for testing."""
        return SuccessPairCollector(storage_dir=temp_dir, max_pairs=5)

    @pytest.fixture
    def sample_patch(self):
        """Create a sample patch for testing."""
        return Patch(
            original_code="def foo():\n    print(bar)\n",
            patched_code="def foo():\n    bar = None  # TODO: Replace with appropriate value\n    print(bar)\n",
            error="NameError: name 'bar' is not defined",
            error_info={
                "type": "NameError",
                "message": "name 'bar' is not defined",
                "patterns": ["undefined_variable"],
                "variable": "bar",
            },
            status=PatchStatus.VALIDATED,
        )

    def test_initialization(self, collector, temp_dir):
        """Test initialization of SuccessPairCollector."""
        assert collector.storage_dir == temp_dir
        assert collector.max_pairs == 5
        assert collector.pairs == []
        assert os.path.exists(temp_dir)

    def test_collect(self, collector, sample_patch):
        """Test collecting a successful patch."""
        collector.collect(sample_patch)

        assert len(collector.pairs) == 1
        assert collector.pairs[0]["original_code"] == sample_patch.original_code
        assert collector.pairs[0]["patched_code"] == sample_patch.patched_code
        assert collector.pairs[0]["error"] == sample_patch.error
        assert collector.pairs[0]["error_info"] == sample_patch.error_info
        assert "timestamp" in collector.pairs[0]

    def test_get_pairs(self, collector, sample_patch):
        """Test getting collected pairs."""
        collector.collect(sample_patch)
        pairs = collector.get_pairs()

        assert len(pairs) == 1
        assert pairs[0]["original_code"] == sample_patch.original_code

    def test_clear(self, collector, sample_patch):
        """Test clearing collected pairs."""
        collector.collect(sample_patch)
        assert len(collector.pairs) == 1

        collector.clear()
        assert len(collector.pairs) == 0

    def test_max_pairs(self, collector, sample_patch):
        """Test that max_pairs is enforced."""
        # Collect 6 pairs (max is 5)
        for i in range(6):
            collector.collect(sample_patch)

        assert len(collector.pairs) == 5

    def test_save_and_load_pairs(self, collector, sample_patch, temp_dir):
        """Test saving and loading pairs."""
        collector.collect(sample_patch)
        
        # Create a new collector to load the pairs
        new_collector = SuccessPairCollector(storage_dir=temp_dir, max_pairs=5)
        
        assert len(new_collector.pairs) == 1
        assert new_collector.pairs[0]["original_code"] == sample_patch.original_code

    def test_export_to_jsonl(self, collector, sample_patch, temp_dir):
        """Test exporting pairs to a JSONL file."""
        collector.collect(sample_patch)
        
        output_file = os.path.join(temp_dir, "export.jsonl")
        collector.export_to_jsonl(output_file)
        
        assert os.path.exists(output_file)
        
        # Check the content of the file
        with open(output_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            # Parse the JSON line
            pair = json.loads(lines[0])
            assert pair["original_code"] == sample_patch.original_code

    def test_import_from_jsonl(self, collector, temp_dir):
        """Test importing pairs from a JSONL file."""
        # Create a JSONL file
        input_file = os.path.join(temp_dir, "import.jsonl")
        with open(input_file, "w") as f:
            f.write(json.dumps({
                "original_code": "def test():\n    print(x)\n",
                "patched_code": "def test():\n    x = None\n    print(x)\n",
                "error": "NameError: name 'x' is not defined",
                "error_info": {
                    "type": "NameError",
                    "message": "name 'x' is not defined",
                    "patterns": ["undefined_variable"],
                    "variable": "x",
                },
                "timestamp": "2023-01-01T00:00:00",
            }) + "\n")
        
        # Import the pairs
        collector.import_from_jsonl(input_file)
        
        assert len(collector.pairs) == 1
        assert collector.pairs[0]["original_code"] == "def test():\n    print(x)\n"

    def test_import_from_jsonl_append(self, collector, sample_patch, temp_dir):
        """Test importing pairs from a JSONL file with append=True."""
        # Collect a pair
        collector.collect(sample_patch)
        
        # Create a JSONL file
        input_file = os.path.join(temp_dir, "import.jsonl")
        with open(input_file, "w") as f:
            f.write(json.dumps({
                "original_code": "def test():\n    print(x)\n",
                "patched_code": "def test():\n    x = None\n    print(x)\n",
                "error": "NameError: name 'x' is not defined",
                "error_info": {
                    "type": "NameError",
                    "message": "name 'x' is not defined",
                    "patterns": ["undefined_variable"],
                    "variable": "x",
                },
                "timestamp": "2023-01-01T00:00:00",
            }) + "\n")
        
        # Import the pairs with append=True
        collector.import_from_jsonl(input_file, append=True)
        
        assert len(collector.pairs) == 2
        assert collector.pairs[0]["original_code"] == sample_patch.original_code
        assert collector.pairs[1]["original_code"] == "def test():\n    print(x)\n"

    def test_get_statistics(self, collector, sample_patch):
        """Test getting statistics about collected pairs."""
        collector.collect(sample_patch)
        
        stats = collector.get_statistics()
        
        assert stats["total_pairs"] == 1
        assert stats["error_types"]["NameError"] == 1
        assert stats["error_patterns"]["undefined_variable"] == 1
