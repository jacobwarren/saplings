"""
Tests for the LoRA fine-tuning pipeline.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from saplings.self_heal.lora_tuning import LoRaTrainer, LoRaConfig, TrainingMetrics

# Check if LoRA dependencies are installed
try:
    from saplings.self_heal.lora_tuning import _HAS_LORA_DEPS
except ImportError:
    _HAS_LORA_DEPS = False

# Skip all tests if LoRA dependencies are not installed
pytestmark = pytest.mark.skipif(
    not _HAS_LORA_DEPS,
    reason="LoRA fine-tuning dependencies not found. Install them with 'pip install saplings[lora]' or 'poetry install --extras lora'."
)


class TestLoRaTrainer:
    """Tests for the LoRaTrainer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_pairs(self, temp_dir):
        """Create sample success pairs for testing."""
        pairs_file = os.path.join(temp_dir, "success_pairs.jsonl")
        with open(pairs_file, "w") as f:
            f.write(json.dumps({
                "original_code": "def foo():\n    print(bar)\n",
                "patched_code": "def foo():\n    bar = None\n    print(bar)\n",
                "error": "NameError: name 'bar' is not defined",
                "error_info": {
                    "type": "NameError",
                    "message": "name 'bar' is not defined",
                    "patterns": ["undefined_variable"],
                    "variable": "bar",
                },
                "timestamp": "2023-01-01T00:00:00",
            }) + "\n")
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
        return pairs_file

    @pytest.fixture
    def lora_config(self):
        """Create a LoRaConfig instance for testing."""
        return LoRaConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            learning_rate=5e-5,
            num_train_epochs=3,
            batch_size=4,
            gradient_accumulation_steps=1,
        )

    @pytest.fixture
    def lora_trainer(self, temp_dir, lora_config):
        """Create a LoRaTrainer instance for testing."""
        return LoRaTrainer(
            model_name="gpt2",
            output_dir=os.path.join(temp_dir, "lora_output"),
            config=lora_config,
        )

    def test_initialization(self, lora_trainer, temp_dir, lora_config):
        """Test initialization of LoRaTrainer."""
        assert lora_trainer.model_name == "gpt2"
        assert lora_trainer.output_dir == os.path.join(temp_dir, "lora_output")
        assert lora_trainer.config == lora_config
        assert lora_trainer.gasa_tune is False

    def test_load_data(self, lora_trainer, sample_pairs):
        """Test loading data from a JSONL file."""
        with patch("saplings.self_heal.lora_tuning.Dataset.from_pandas") as mock_from_pandas:
            mock_from_pandas.return_value = MagicMock()

            dataset = lora_trainer.load_data(sample_pairs)

            assert mock_from_pandas.called
            # Check that the dataframe passed to from_pandas has the expected columns
            df = mock_from_pandas.call_args[0][0]
            assert "input" in df.columns
            assert "output" in df.columns
            assert len(df) == 2

    def test_preprocess_data(self, lora_trainer):
        """Test preprocessing data for training."""
        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = MagicMock()

        with patch("saplings.self_heal.lora_tuning.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()

            processed_dataset = lora_trainer.preprocess_data(mock_dataset)

            assert mock_tokenizer.called
            assert mock_dataset.map.called

    def test_train(self, lora_trainer, sample_pairs):
        """Test training the LoRA model."""
        with patch("saplings.self_heal.lora_tuning.AutoModelForCausalLM.from_pretrained") as mock_model, \
             patch("saplings.self_heal.lora_tuning.AutoTokenizer.from_pretrained") as mock_tokenizer, \
             patch("saplings.self_heal.lora_tuning.get_peft_model") as mock_get_peft_model, \
             patch("saplings.self_heal.lora_tuning.Trainer") as mock_trainer, \
             patch("saplings.self_heal.lora_tuning.Dataset.from_pandas") as mock_from_pandas:

            # Mock the model and tokenizer
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_get_peft_model.return_value = MagicMock()
            mock_trainer.return_value = MagicMock()
            mock_trainer.return_value.train.return_value = None
            mock_trainer.return_value.evaluate.return_value = {"eval_loss": 0.1}
            mock_from_pandas.return_value = MagicMock()

            # Train the model
            metrics = lora_trainer.train(sample_pairs)

            # Check that the expected methods were called
            assert mock_model.called
            assert mock_tokenizer.called
            assert mock_get_peft_model.called
            assert mock_trainer.called
            assert mock_trainer.return_value.train.called
            assert mock_trainer.return_value.evaluate.called

            # Check the returned metrics
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.eval_loss == 0.1

    def test_gasa_tune(self, lora_trainer, sample_pairs):
        """Test GASA-specific tuning."""
        # Enable GASA tuning
        lora_trainer.gasa_tune = True

        with patch("saplings.self_heal.lora_tuning.AutoModelForCausalLM.from_pretrained") as mock_model, \
             patch("saplings.self_heal.lora_tuning.AutoTokenizer.from_pretrained") as mock_tokenizer, \
             patch("saplings.self_heal.lora_tuning.get_peft_model") as mock_get_peft_model, \
             patch("saplings.self_heal.lora_tuning.Trainer") as mock_trainer, \
             patch("saplings.self_heal.lora_tuning.Dataset.from_pandas") as mock_from_pandas:

            # Mock the model and tokenizer
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_get_peft_model.return_value = MagicMock()
            mock_trainer.return_value = MagicMock()
            mock_trainer.return_value.train.return_value = None
            mock_trainer.return_value.evaluate.return_value = {"eval_loss": 0.1}
            mock_from_pandas.return_value = MagicMock()

            # Train the model with GASA tuning
            metrics = lora_trainer.train(sample_pairs)

            # Check that the expected methods were called
            assert mock_model.called
            assert mock_tokenizer.called
            assert mock_get_peft_model.called
            assert mock_trainer.called
            assert mock_trainer.return_value.train.called
            assert mock_trainer.return_value.evaluate.called

            # Check the returned metrics
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.eval_loss == 0.1

            # Check that GASA-specific target modules were used
            config = mock_get_peft_model.call_args[1]["peft_config"]
            assert "k_proj" in config.target_modules  # GASA-specific module

    def test_save_and_load(self, lora_trainer, temp_dir):
        """Test saving and loading the LoRA model."""
        with patch("saplings.self_heal.lora_tuning.AutoModelForCausalLM.from_pretrained") as mock_model, \
             patch("saplings.self_heal.lora_tuning.AutoTokenizer.from_pretrained") as mock_tokenizer, \
             patch("saplings.self_heal.lora_tuning.PeftModel.from_pretrained") as mock_peft_model:

            # Mock the model and tokenizer
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_peft_model.return_value = MagicMock()

            # Create a mock trained model
            mock_trained_model = MagicMock()
            mock_trained_model.save_pretrained.return_value = None

            # Save the model
            lora_trainer.save_model(mock_trained_model)

            # Check that save_pretrained was called
            assert mock_trained_model.save_pretrained.called

            # Load the model
            loaded_model = lora_trainer.load_model()

            # Check that the expected methods were called
            assert mock_model.called
            assert mock_peft_model.called
            assert loaded_model is not None

    def test_integration_with_patch_generator(self, lora_trainer, sample_pairs, temp_dir):
        """Test integration with PatchGenerator."""
        from saplings.self_heal.patch_generator import PatchGenerator, Patch, PatchStatus
        from saplings.self_heal.success_pair_collector import SuccessPairCollector

        # Create a success pair collector
        collector = SuccessPairCollector(storage_dir=os.path.join(temp_dir, "success_pairs"))

        # Create a patch generator with the collector
        patch_generator = PatchGenerator(
            max_retries=3,
            success_pair_collector=collector
        )

        # Create a sample patch
        patch = Patch(
            original_code="def foo():\n    print(bar)\n",
            patched_code="def foo():\n    bar = None\n    print(bar)\n",
            error="NameError: name 'bar' is not defined",
            error_info={
                "type": "NameError",
                "message": "name 'bar' is not defined",
                "patterns": ["undefined_variable"],
                "variable": "bar",
            },
            status=PatchStatus.GENERATED,
        )

        # Apply the patch
        patch_generator.apply_patch(patch)

        # Mark the patch as successful
        patch.status = PatchStatus.VALIDATED
        patch_generator.after_success(patch)

        # Export the success pairs
        pairs_file = os.path.join(temp_dir, "export.jsonl")
        collector.export_to_jsonl(pairs_file)

        # Mock the training process
        with patch("saplings.self_heal.lora_tuning.AutoModelForCausalLM.from_pretrained") as mock_model, \
             patch("saplings.self_heal.lora_tuning.AutoTokenizer.from_pretrained") as mock_tokenizer, \
             patch("saplings.self_heal.lora_tuning.get_peft_model") as mock_get_peft_model, \
             patch("saplings.self_heal.lora_tuning.Trainer") as mock_trainer, \
             patch("saplings.self_heal.lora_tuning.Dataset.from_pandas") as mock_from_pandas:

            # Mock the model and tokenizer
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_get_peft_model.return_value = MagicMock()
            mock_trainer.return_value = MagicMock()
            mock_trainer.return_value.train.return_value = None
            mock_trainer.return_value.evaluate.return_value = {"eval_loss": 0.1}
            mock_from_pandas.return_value = MagicMock()

            # Train the model using the collected success pairs
            metrics = lora_trainer.train(pairs_file)

            # Check that the expected methods were called
            assert mock_model.called
            assert mock_tokenizer.called
            assert mock_get_peft_model.called
            assert mock_trainer.called
            assert mock_trainer.return_value.train.called

            # Check the returned metrics
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.eval_loss == 0.1
