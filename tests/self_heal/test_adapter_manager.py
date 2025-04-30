"""
Tests for the AdapterManager class.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from saplings.self_heal.adapter_manager import AdapterManager, AdapterMetadata, AdapterPriority


class TestAdapterManager:
    """Tests for the AdapterManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def adapter_manager(self, temp_dir):
        """Create an AdapterManager instance for testing."""
        return AdapterManager(
            model_name="gpt2",
            adapters_dir=os.path.join(temp_dir, "adapters"),
        )

    @pytest.fixture
    def sample_metadata(self):
        """Create sample adapter metadata for testing."""
        return AdapterMetadata(
            adapter_id="test_adapter",
            model_name="gpt2",
            description="Test adapter",
            version="1.0.0",
            created_at="2023-01-01T00:00:00",
            success_rate=0.85,
            priority=AdapterPriority.MEDIUM,
            error_types=["NameError", "SyntaxError"],
            tags=["test", "python"],
        )

    def test_initialization(self, adapter_manager, temp_dir):
        """Test initialization of AdapterManager."""
        assert adapter_manager.model_name == "gpt2"
        assert adapter_manager.adapters_dir == os.path.join(temp_dir, "adapters")
        assert os.path.exists(adapter_manager.adapters_dir)
        assert adapter_manager.adapters == {}
        assert adapter_manager.active_adapter is None

    def test_register_adapter(self, adapter_manager, sample_metadata):
        """Test registering an adapter."""
        # Mock the adapter path
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        
        # Register the adapter
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Check that the adapter was registered
        assert sample_metadata.adapter_id in adapter_manager.adapters
        assert adapter_manager.adapters[sample_metadata.adapter_id].metadata == sample_metadata
        
        # Check that the metadata was saved
        metadata_path = os.path.join(adapter_path, "metadata.json")
        assert os.path.exists(metadata_path)
        
        # Check the content of the metadata file
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)
            assert loaded_metadata["adapter_id"] == sample_metadata.adapter_id
            assert loaded_metadata["model_name"] == sample_metadata.model_name
            assert loaded_metadata["success_rate"] == sample_metadata.success_rate

    def test_load_adapters(self, adapter_manager, sample_metadata):
        """Test loading adapters from the adapters directory."""
        # Create an adapter directory with metadata
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        
        # Save the metadata
        metadata_path = os.path.join(adapter_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata.__dict__, f)
        
        # Load the adapters
        adapter_manager.load_adapters()
        
        # Check that the adapter was loaded
        assert sample_metadata.adapter_id in adapter_manager.adapters
        assert adapter_manager.adapters[sample_metadata.adapter_id].metadata.adapter_id == sample_metadata.adapter_id

    def test_get_adapter(self, adapter_manager, sample_metadata):
        """Test getting an adapter by ID."""
        # Register an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Get the adapter
        adapter = adapter_manager.get_adapter(sample_metadata.adapter_id)
        
        # Check that the correct adapter was returned
        assert adapter.metadata.adapter_id == sample_metadata.adapter_id

    def test_activate_adapter(self, adapter_manager, sample_metadata):
        """Test activating an adapter."""
        # Register an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Mock the load_model method
        with patch("saplings.self_heal.adapter_manager.LoRaTrainer") as mock_trainer:
            mock_trainer.return_value.load_model.return_value = MagicMock()
            
            # Activate the adapter
            adapter_manager.activate_adapter(sample_metadata.adapter_id)
            
            # Check that the adapter was activated
            assert adapter_manager.active_adapter == sample_metadata.adapter_id
            assert mock_trainer.return_value.load_model.called

    def test_deactivate_adapter(self, adapter_manager, sample_metadata):
        """Test deactivating an adapter."""
        # Register and activate an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Set the active adapter
        adapter_manager.active_adapter = sample_metadata.adapter_id
        
        # Deactivate the adapter
        adapter_manager.deactivate_adapter()
        
        # Check that the adapter was deactivated
        assert adapter_manager.active_adapter is None

    def test_update_adapter_metadata(self, adapter_manager, sample_metadata):
        """Test updating adapter metadata."""
        # Register an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Update the metadata
        updated_metadata = sample_metadata.__dict__.copy()
        updated_metadata["success_rate"] = 0.9
        updated_metadata["priority"] = AdapterPriority.HIGH.value
        
        adapter_manager.update_adapter_metadata(sample_metadata.adapter_id, updated_metadata)
        
        # Check that the metadata was updated
        adapter = adapter_manager.get_adapter(sample_metadata.adapter_id)
        assert adapter.metadata.success_rate == 0.9
        assert adapter.metadata.priority == AdapterPriority.HIGH

    def test_find_adapters_for_error(self, adapter_manager, sample_metadata):
        """Test finding adapters for a specific error type."""
        # Register an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Find adapters for a NameError
        adapters = adapter_manager.find_adapters_for_error("NameError")
        
        # Check that the adapter was found
        assert len(adapters) == 1
        assert adapters[0].metadata.adapter_id == sample_metadata.adapter_id
        
        # Find adapters for a TypeError (not in the adapter's error_types)
        adapters = adapter_manager.find_adapters_for_error("TypeError")
        
        # Check that no adapters were found
        assert len(adapters) == 0

    def test_prune_adapters(self, adapter_manager):
        """Test pruning underperforming adapters."""
        # Create and register multiple adapters with different success rates
        for i, success_rate in enumerate([0.3, 0.5, 0.7, 0.9]):
            metadata = AdapterMetadata(
                adapter_id=f"adapter_{i}",
                model_name="gpt2",
                description=f"Adapter {i}",
                version="1.0.0",
                created_at="2023-01-01T00:00:00",
                success_rate=success_rate,
                priority=AdapterPriority.MEDIUM,
                error_types=["NameError"],
                tags=["test"],
            )
            
            adapter_path = os.path.join(adapter_manager.adapters_dir, metadata.adapter_id)
            os.makedirs(adapter_path, exist_ok=True)
            adapter_manager.register_adapter(adapter_path, metadata)
        
        # Prune adapters with success rate < 0.6
        adapter_manager.prune_adapters(min_success_rate=0.6)
        
        # Check that only adapters with success rate >= 0.6 remain
        assert len(adapter_manager.adapters) == 2
        assert "adapter_2" in adapter_manager.adapters  # success_rate = 0.7
        assert "adapter_3" in adapter_manager.adapters  # success_rate = 0.9
        assert "adapter_0" not in adapter_manager.adapters  # success_rate = 0.3
        assert "adapter_1" not in adapter_manager.adapters  # success_rate = 0.5

    def test_process_judge_feedback(self, adapter_manager, sample_metadata):
        """Test processing feedback from JudgeAgent."""
        # Register an adapter
        adapter_path = os.path.join(adapter_manager.adapters_dir, sample_metadata.adapter_id)
        os.makedirs(adapter_path, exist_ok=True)
        adapter_manager.register_adapter(adapter_path, sample_metadata)
        
        # Set the active adapter
        adapter_manager.active_adapter = sample_metadata.adapter_id
        
        # Process feedback with a high score
        adapter_manager.process_judge_feedback(0.9, "Good patch")
        
        # Check that the success rate was updated
        adapter = adapter_manager.get_adapter(sample_metadata.adapter_id)
        assert adapter.metadata.success_rate > 0.85  # Original success_rate was 0.85
        
        # Process feedback with a low score
        adapter_manager.process_judge_feedback(0.3, "Bad patch")
        
        # Check that the success rate was updated
        adapter = adapter_manager.get_adapter(sample_metadata.adapter_id)
        assert adapter.metadata.success_rate < 0.85  # Original success_rate was 0.85
