"""
LoRA fine-tuning module for Saplings.

This module provides the LoRA fine-tuning pipeline for continual improvement of models.
"""

import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import APScheduler for scheduling
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    _HAS_SCHEDULER_DEPS = True
except ImportError:
    _HAS_SCHEDULER_DEPS = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "APScheduler not found. Install it with 'pip install apscheduler' "
        "to enable scheduling functionality."
    )

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig as PeftLoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
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
    target_modules: List[str] = None
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
    timestamp: str = None

    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
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
        config: Optional[LoRaConfig] = None,
        gasa_tune: bool = False,
    ):
        """
        Initialize the LoRA trainer.

        Args:
            model_name: Name of the base model to fine-tune
            output_dir: Directory to save the fine-tuned model
            config: LoRA configuration
            gasa_tune: Whether to tune specifically for GASA
        """
        # Check if LoRA dependencies are installed
        if not _HAS_LORA_DEPS:
            raise ImportError(
                "LoRA fine-tuning dependencies not found. "
                "Install them with 'pip install saplings[lora]' or "
                "'poetry install --extras lora'."
            )

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
            data_path: Path to the JSONL file

        Returns:
            Any: Loaded dataset (Dataset type when dependencies are installed)
        """
        # Load the data
        pairs = []
        with open(data_path, "r") as f:
            for line in f:
                pair = json.loads(line.strip())
                pairs.append(pair)

        # Convert to a format suitable for training
        data = []
        for pair in pairs:
            # Create input-output pairs
            data.append({
                "input": f"Error: {pair['error']}\nCode: {pair['original_code']}",
                "output": pair["patched_code"],
            })

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
            dataset: Dataset to preprocess

        Returns:
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

    def train(self, data_path: str) -> TrainingMetrics:
        """
        Train the model using LoRA.

        Args:
            data_path: Path to the training data

        Returns:
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

        # Prepare the model for training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        if self.gasa_tune:
            # For GASA tuning, we need to include attention modules
            target_modules = self.config.target_modules + ["k_proj", "o_proj"]
        else:
            target_modules = self.config.target_modules

        peft_config = PeftLoraConfig(
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
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

        # Create the data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create the trainer
        trainer = Trainer(
            model=model,
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
        self.save_model(model)

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

    def save_model(self, model: Any) -> None:
        """
        Save the fine-tuned model.

        Args:
            model: Fine-tuned model to save (PeftModel type when dependencies are installed)
        """
        model.save_pretrained(self.output_dir)

        # Save the configuration
        config_path = os.path.join(self.output_dir, "lora_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"Model saved to {self.output_dir}")

    def load_model(self) -> Any:
        """
        Load a fine-tuned model.

        Returns:
            Any: Loaded model (PeftModel type when dependencies are installed)
        """
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load the PEFT model
        model = PeftModel.from_pretrained(
            base_model,
            self.output_dir,
            device_map="auto",
        )

        logger.info(f"Model loaded from {self.output_dir}")

        return model

    def schedule_nightly_training(self, data_path: str, cron_expression: str = "0 0 * * *") -> None:
        """
        Schedule nightly training using APScheduler.

        Args:
            data_path: Path to the training data
            cron_expression: Cron expression for scheduling (default: midnight every day)
        """
        if not _HAS_SCHEDULER_DEPS:
            logger.warning(
                "APScheduler not found. Install it with 'pip install apscheduler' "
                "to enable scheduling functionality."
            )
            return

        # Create a background scheduler
        scheduler = BackgroundScheduler()

        # Define the training job
        def training_job():
            try:
                logger.info(f"Starting scheduled training with data from {data_path}")
                metrics = self.train(data_path)
                logger.info(f"Scheduled training completed with eval_loss: {metrics.eval_loss}")

                # Save training record
                record_path = os.path.join(self.output_dir, "training_history.jsonl")
                with open(record_path, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "data_path": data_path,
                        "metrics": metrics.to_dict()
                    }) + "\n")
            except Exception as e:
                logger.error(f"Error in scheduled training: {e}")

        # Add the job to the scheduler with the specified cron expression
        scheduler.add_job(
            training_job,
            CronTrigger.from_crontab(cron_expression),
            id="nightly_training",
            replace_existing=True,
            misfire_grace_time=3600  # Allow job to be missed by up to 1 hour
        )

        # Start the scheduler if it's not already running
        if not scheduler.running:
            scheduler.start()

            # Register signal handlers for clean shutdown
            def shutdown_scheduler(signum, frame):
                logger.info("Shutting down scheduler...")
                scheduler.shutdown()
                sys.exit(0)

            signal.signal(signal.SIGINT, shutdown_scheduler)
            signal.signal(signal.SIGTERM, shutdown_scheduler)

        logger.info(f"Scheduled nightly training with data from {data_path}")
        logger.info(f"Cron expression: {cron_expression}")
        logger.info("Training will run automatically according to the schedule.")

    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate a fine-tuned model.

        Args:
            test_data_path: Path to the test data

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Load the model
        model = self.load_model()

        # Load and preprocess the data
        dataset = self.load_data(test_data_path)
        processed_dataset = self.preprocess_data(dataset)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create the data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create the trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.output_dir,
                per_device_eval_batch_size=self.config.batch_size,
                report_to="none",
            ),
            data_collator=data_collator,
        )

        # Evaluate the model
        logger.info(f"Evaluating model with {len(processed_dataset)} examples")
        eval_result = trainer.evaluate(eval_dataset=processed_dataset)

        logger.info(f"Evaluation completed with metrics: {eval_result}")

        return eval_result
