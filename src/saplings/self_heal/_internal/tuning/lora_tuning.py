from __future__ import annotations

"""
LoRA fine-tuning module for Saplings.

This module provides the LoRA fine-tuning pipeline for continual improvement of models.
"""


import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

# Try to import APScheduler for scheduling
try:
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    from apscheduler.triggers.cron import CronTrigger  # type: ignore

    _HAS_SCHEDULER_DEPS = True
except ImportError:
    _HAS_SCHEDULER_DEPS = False

    # Define placeholder classes for type checking
    class BackgroundScheduler:
        """Placeholder for BackgroundScheduler when not installed."""

        running = False

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "APScheduler not installed. Install with 'pip install saplings[lora]'"
            )

        def start(self):
            pass

        def shutdown(self):
            pass

        def add_job(self, *args, **kwargs):
            pass

    class CronTrigger:
        """Placeholder for CronTrigger when not installed."""

        @staticmethod
        def from_crontab(crontab):
            raise ImportError(
                "APScheduler not installed. Install with 'pip install saplings[lora]'"
            )

    logger = logging.getLogger(__name__)
    logger.warning(
        "APScheduler not found. Install it with 'pip install apscheduler' "
        "to enable scheduling functionality."
    )

logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    # Define types for PEFT
    # Use Protocol to define interfaces without requiring imports
    from typing import List, Protocol

    import pandas as pd
    import torch
    from datasets import Dataset

    class PeftLoraConfig(Protocol):
        """Protocol for PEFT LoraConfig."""

        def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            bias: str,
            task_type: Any,
            target_modules: List[str] | None,
        ) -> None: ...

    class PeftModel(Protocol):
        """Protocol for PEFT PeftModel."""

        @staticmethod
        def from_pretrained(
            model: Any,
            model_id: str,
            device_map: str,
        ) -> Any: ...

        def save_pretrained(self, save_directory: str) -> None: ...

    class TaskType(Protocol):
        """Protocol for PEFT TaskType."""

        CAUSAL_LM: Any

    def get_peft_model(model: Any, peft_config: Any) -> Any: ...
    def prepare_model_for_kbit_training(model: Any) -> Any: ...

    # Transformers types
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments
    from transformers.utils.quantization_config import BitsAndBytesConfig

# Runtime imports
try:
    import pandas as pd
    import torch
    from datasets import Dataset

    # Import peft with proper error handling
    try:
        from peft import LoraConfig as PeftLoraConfig  # type: ignore
        from peft import (  # type: ignore
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False

        # Define placeholder classes that match the Protocol interfaces
        class _PeftLoraConfig:
            """Placeholder for PeftLoraConfig when not installed."""

            def __init__(
                self,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=None,
                target_modules=None,
            ):
                raise ImportError(
                    "PEFT is not installed. Install with 'pip install saplings[lora]'"
                )

        class _PeftModel:
            """Placeholder for PeftModel when not installed."""

            @staticmethod
            def from_pretrained(model, model_id, device_map="auto"):
                raise ImportError(
                    "PEFT is not installed. Install with 'pip install saplings[lora]'"
                )

            def save_pretrained(self, save_directory):
                raise ImportError(
                    "PEFT is not installed. Install with 'pip install saplings[lora]'"
                )

        class _TaskType:
            """Placeholder TaskType class."""

            CAUSAL_LM = None

        def _get_peft_model(model, peft_config):
            """Placeholder for get_peft_model when not installed."""
            raise ImportError("PEFT is not installed. Install with 'pip install saplings[lora]'")

        def _prepare_model_for_kbit_training(model):
            """Placeholder for prepare_model_for_kbit_training when not installed."""
            raise ImportError("PEFT is not installed. Install with 'pip install saplings[lora]'")

        # Assign placeholders
        PeftLoraConfig = _PeftLoraConfig
        PeftModel = _PeftModel
        TaskType = _TaskType
        get_peft_model = _get_peft_model
        prepare_model_for_kbit_training = _prepare_model_for_kbit_training

    # Import transformers with proper module paths
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    # BitsAndBytesConfig is used in the code when loading models with quantization
    from transformers.utils.quantization_config import BitsAndBytesConfig  # noqa: F401

    _HAS_LORA_DEPS = True
except ImportError:
    _HAS_LORA_DEPS = False
    logger.warning(
        "LoRA fine-tuning dependencies not found. "
        "Install them with 'pip install saplings[lora]' or "
        "'poetry install --extras lora'."
    )


@dataclass
class LoRaConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] | None = None
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Initialize default values."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


@dataclass
class TrainingMetrics:
    """Metrics from training."""

    train_loss: float
    eval_loss: float
    train_runtime: float
    train_samples_per_second: float
    epoch: float
    timestamp: str | None = None

    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "train_runtime": self.train_runtime,
            "train_samples_per_second": self.train_samples_per_second,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
        }


class LoRaTrainer:
    """
    Trainer for LoRA fine-tuning.

    This class provides functionality for fine-tuning models using LoRA.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        config: LoRaConfig | None = None,
        gasa_tune: bool = False,
    ) -> None:
        """
        Initialize the LoRA trainer.

        Args:
        ----
            model_name: Name of the base model to fine-tune
            output_dir: Directory to save the fine-tuned model
            config: LoRA configuration
            gasa_tune: Whether to tune specifically for GASA

        """
        # Check if LoRA dependencies are installed
        if not _HAS_LORA_DEPS:
            msg = (
                "LoRA fine-tuning dependencies not found. "
                "Install them with 'pip install saplings[lora]' or "
                "'poetry install --extras lora'."
            )
            raise ImportError(msg)

        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config or LoRaConfig()
        self.gasa_tune = gasa_tune

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, data_path: str) -> Any:
        """
        Load data from a JSONL file.

        Args:
        ----
            data_path: Path to the JSONL file

        Returns:
        -------
            Any: Loaded dataset (Dataset type when dependencies are installed)

        """
        # Load the data
        pairs = []
        with open(data_path) as f:
            for line in f:
                pair = json.loads(line.strip())
                pairs.append(pair)

        # Convert to a format suitable for training
        data = []
        for pair in pairs:
            # Create input-output pairs
            data.append(
                {
                    "input": f"Error: {pair['error']}\nCode: {pair['original_code']}",
                    "output": pair["patched_code"],
                }
            )

        # Convert to a pandas DataFrame
        df = pd.DataFrame(data)

        # Convert to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        logger.info(f"Loaded {len(dataset)} examples from {data_path}")

        return dataset

    def preprocess_data(self, dataset: Any) -> Any:
        """
        Preprocess the dataset for training.

        Args:
        ----
            dataset: Dataset to preprocess

        Returns:
        -------
            Any: Preprocessed dataset (Dataset type when dependencies are installed)

        """
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Define the preprocessing function
        def preprocess_function(examples):
            # Combine input and output with a separator
            texts = [
                f"{input_text}\n\nFixed Code:\n{output_text}"
                for input_text, output_text in zip(examples["input"], examples["output"])
            ]

            # Tokenize the texts
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Prepare the labels (same as input_ids for causal language modeling)
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Apply the preprocessing function
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        logger.info(f"Preprocessed {len(processed_dataset)} examples")

        return processed_dataset

    def load_model(self):
        """
        Load a fine-tuned model.

        Returns
        -------
            Any: Loaded model (PeftModel type when dependencies are installed)

        """
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Check if PEFT is available
        if not PEFT_AVAILABLE:
            msg = (
                "PEFT library not found. "
                "Install it with 'pip install saplings[lora]' or "
                "'poetry install --extras lora'."
            )
            raise ImportError(msg)

        # Load the PEFT model
        # Cast PeftModel to Any to avoid Protocol instantiation issues
        model = cast("Any", PeftModel).from_pretrained(
            base_model,
            self.output_dir,
            device_map="auto",
        )

        logger.info(f"Model loaded from {self.output_dir}")

        return model

    def train(self, data_path: str) -> TrainingMetrics:
        """
        Train the model using LoRA.

        Args:
        ----
            data_path: Path to the training data

        Returns:
        -------
            TrainingMetrics: Metrics from training

        """
        # Load and preprocess the data
        dataset = self.load_data(data_path)
        processed_dataset = self.preprocess_data(dataset)

        # Split the dataset into train and validation
        split_dataset = processed_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Check if PEFT is available
        if not PEFT_AVAILABLE:
            msg = (
                "PEFT library not found. "
                "Install it with 'pip install saplings[lora]' or "
                "'poetry install --extras lora'."
            )
            raise ImportError(msg)

        # Prepare the model for training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        if self.gasa_tune:
            # For GASA tuning, we need to include attention modules
            base_modules = self.config.target_modules or []
            target_modules = [*base_modules, "k_proj", "o_proj"]
        else:
            target_modules = self.config.target_modules

        # Cast PeftLoraConfig to Any to avoid Protocol instantiation issues
        peft_config = cast("Any", PeftLoraConfig)(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        # Get the PEFT model
        model = get_peft_model(model, peft_config=peft_config)

        # Set up training arguments
        # Create a dictionary of arguments first
        args_dict = {
            "output_dir": self.output_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "save_strategy": "epoch",
            "logging_dir": os.path.join(self.output_dir, "logs"),
            "logging_steps": 10,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",
        }

        # Add evaluation_strategy if it exists in the TrainingArguments class
        if hasattr(TrainingArguments, "evaluation_strategy"):
            args_dict["evaluation_strategy"] = "epoch"
        elif hasattr(TrainingArguments, "eval_strategy"):
            args_dict["eval_strategy"] = "epoch"

        # Create the TrainingArguments object
        training_args = TrainingArguments(**args_dict)

        # Create the data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create the trainer
        # Cast model to Any to avoid type checking issues with PeftModel
        trainer = Trainer(
            model=cast("Any", model),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train the model
        logger.info(f"Starting training with {len(train_dataset)} examples")
        train_result = trainer.train()

        # Evaluate the model
        logger.info(f"Evaluating model with {len(eval_dataset)} examples")
        eval_result = trainer.evaluate()

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        model.save_pretrained(self.output_dir)

        # Create and return metrics
        metrics = TrainingMetrics(
            train_loss=train_result.training_loss,
            eval_loss=eval_result["eval_loss"],
            train_runtime=train_result.metrics["train_runtime"],
            train_samples_per_second=train_result.metrics["train_samples_per_second"],
            epoch=train_result.metrics["epoch"],
        )

        # Save metrics
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Training completed with eval_loss: {metrics.eval_loss}")

        return metrics
