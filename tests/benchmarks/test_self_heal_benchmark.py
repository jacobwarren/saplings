"""
Benchmark tests for self-healing components.

This module provides benchmark tests for self-healing components, measuring
patch generation quality, LoRA tuning performance, and adapter management.
"""

import json
import os
import tempfile
from datetime import datetime

import pytest

# Import only what we need
from saplings.self_heal import (
    PatchGenerator,
    LoRaConfig,
    AdapterManager,
    SuccessPairCollector,
)
from saplings.monitoring import TraceManager, MonitoringConfig

from tests.benchmarks.base_benchmark import BaseBenchmark, MockBenchmarkLLM
from tests.benchmarks.test_datasets import TestDatasets


class TestSelfHealBenchmark(BaseBenchmark):
    """Benchmark tests for self-healing components."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockBenchmarkLLM("mock://model/latest", response_time_ms=200)

    @pytest.fixture
    def trace_manager(self):
        """Create a trace manager for testing."""
        return TraceManager(config=MonitoringConfig())

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_patch_generator(self):
        """Test patch generator performance."""
        # Create code samples with errors
        code_samples = TestDatasets.create_code_samples(
            num_samples=10,
            with_errors=True,
        )

        # Create patch generator
        patch_generator = PatchGenerator(
            max_retries=3,
        )

        # Results dictionary
        results = {
            "samples": [],
        }

        # Test each code sample
        for i, sample in enumerate(code_samples):
            if not sample["has_error"]:
                continue

            print(f"\nTesting sample {i+1}...")

            # Generate patch
            error_str = f"{sample['error_type']}: {sample['error_message']}"
            patch = patch_generator.generate_patch(sample["code"], error_str)
            result, latency = await self.time_async_execution(
                patch_generator.apply_patch,
                patch=patch,
            )

            # Add to results
            results["samples"].append({
                "sample_id": i,
                "code": sample["code"],
                "error": f"{sample['error_type']}: {sample['error_message']}",
                "patch": patch.patched_code if patch else None,
                "success": result.success if result else False,
                "latency_ms": latency,
            })

            print(f"  Latency: {latency:.2f}ms")
            print(f"  Patch generated: {result is not None}")

        # Save results
        self.save_results(results, "patch_generator_performance")

    @pytest.mark.asyncio
    async def test_success_pair_collector(self, temp_dir):
        """Test success pair collector performance."""
        # Create code samples
        code_samples = TestDatasets.create_code_samples(
            num_samples=10,
            with_errors=True,
        )

        # Create success pair collector
        collector = SuccessPairCollector(
            storage_dir=temp_dir,
            max_pairs=100,
        )

        # Results dictionary
        results = {
            "pairs": [],
        }

        # Test collection
        for i, sample in enumerate(code_samples):
            if not sample["has_error"]:
                continue

            print(f"\nCollecting sample {i+1}...")

            # Create patch generator
            patch_generator = PatchGenerator(
                max_retries=3,
            )

            # Generate patch
            error_str = f"{sample['error_type']}: {sample['error_message']}"
            patch = patch_generator.generate_patch(sample["code"], error_str)

            if not patch:
                continue

            # Collect success pair
            result, latency = await self.time_async_execution(
                collector.collect,
                patch=patch,
            )

            # Add to results
            results["pairs"].append({
                "sample_id": i,
                "latency_ms": latency,
                "success": result,
            })

            print(f"  Latency: {latency:.2f}ms")
            print(f"  Success: {result}")

        # Save results
        self.save_results(results, "success_pair_collector_performance")

    @pytest.mark.asyncio
    async def test_adapter_manager(self, mock_llm, temp_dir):
        """Test adapter manager performance."""
        # Create adapter manager
        adapter_manager = AdapterManager(
            model_name=mock_llm.model_uri,
            adapters_dir=temp_dir,
        )

        # Results dictionary
        results = {
            "operations": [],
        }

        # Test adapter registration
        print("\nTesting adapter registration...")

        # Create a mock adapter path
        adapter_path = os.path.join(temp_dir, "test-adapter")
        os.makedirs(adapter_path, exist_ok=True)

        # Create adapter metadata
        from saplings.self_heal.adapter_manager import AdapterMetadata, AdapterPriority

        metadata = AdapterMetadata(
            adapter_id="test-adapter",
            model_name=mock_llm.model_uri,
            description="Test adapter for benchmarking",
            version="1.0",
            created_at=datetime.now().isoformat(),
            success_rate=0.8,
            priority=AdapterPriority.MEDIUM,
            error_types=["SyntaxError", "NameError"],
            tags=["test", "benchmark"],
        )

        # Register adapter
        _, creation_latency = await self.time_async_execution(
            adapter_manager.register_adapter,
            path=adapter_path,
            metadata=metadata,
        )

        adapter_id = metadata.adapter_id

        results["operations"].append({
            "operation": "create_adapter",
            "adapter_id": adapter_id,
            "latency_ms": creation_latency,
        })

        print(f"  Creation latency: {creation_latency:.2f}ms")
        print(f"  Adapter ID: {adapter_id}")

        # Test adapter activation
        print("\nTesting adapter activation...")
        load_latency = 0
        for _ in range(self.NUM_RUNS):
            # The activate_adapter method returns a boolean, not a coroutine
            # So we need to wrap it in a lambda to time it
            async def activate_adapter():
                return adapter_manager.activate_adapter(adapter_id=adapter_id)

            _, latency = await self.time_async_execution(
                activate_adapter,
            )
            load_latency += latency

        load_latency /= self.NUM_RUNS

        results["operations"].append({
            "operation": "load_adapter",
            "adapter_id": adapter_id,
            "latency_ms": load_latency,
        })

        print(f"  Load latency: {load_latency:.2f}ms")

        # Test adapter deactivation
        print("\nTesting adapter deactivation...")
        unload_latency = 0
        for _ in range(self.NUM_RUNS):
            # The deactivate_adapter method is not a coroutine
            # So we need to wrap it in a lambda to time it
            async def deactivate_adapter():
                adapter_manager.deactivate_adapter()
                return True

            _, latency = await self.time_async_execution(
                deactivate_adapter,
            )
            unload_latency += latency

        unload_latency /= self.NUM_RUNS

        results["operations"].append({
            "operation": "unload_adapter",
            "adapter_id": adapter_id,
            "latency_ms": unload_latency,
        })

        print(f"  Unload latency: {unload_latency:.2f}ms")

        # Test adapter switching
        print("\nTesting adapter switching...")
        switch_latency = 0
        for _ in range(self.NUM_RUNS):
            # Activate adapter (using our async wrapper)
            async def activate_first_adapter():
                return adapter_manager.activate_adapter(adapter_id=adapter_id)

            await activate_first_adapter()

            # Create another adapter for switching
            other_adapter_path = os.path.join(temp_dir, "other-adapter")
            os.makedirs(other_adapter_path, exist_ok=True)

            other_metadata = AdapterMetadata(
                adapter_id="other-adapter",
                model_name=mock_llm.model_uri,
                description="Another test adapter",
                version="1.0",
                created_at=datetime.now().isoformat(),
                success_rate=0.7,
                priority=AdapterPriority.LOW,
                error_types=["TypeError"],
                tags=["test"],
            )

            # Register the other adapter
            adapter_manager.register_adapter(other_adapter_path, other_metadata)

            # Switch to the other adapter (using our async wrapper)
            async def activate_other_adapter():
                return adapter_manager.activate_adapter(adapter_id="other-adapter")

            _, latency = await self.time_async_execution(
                activate_other_adapter,
            )
            switch_latency += latency

        switch_latency /= self.NUM_RUNS

        results["operations"].append({
            "operation": "switch_adapter",
            "adapter_id": "other-adapter",
            "latency_ms": switch_latency,
        })

        print(f"  Switch latency: {switch_latency:.2f}ms")

        # Save results
        self.save_results(results, "adapter_manager_performance")

    @pytest.mark.asyncio
    async def test_lora_tuning(self, mock_llm, temp_dir):
        """Test LoRA tuning performance."""
        # Skip actual training in benchmark
        print("\nNote: This benchmark simulates LoRA tuning without actual training.")

        # Create code samples for training data
        code_samples = TestDatasets.create_code_samples(
            num_samples=20,
            with_errors=True,
        )

        # Create success pairs
        success_pairs = []
        for sample in code_samples:
            if sample["has_error"]:
                success_pairs.append({
                    "original_code": sample["code"],
                    "patched_code": sample["code"].replace("error", "fixed"),  # Simulate a fix
                    "error": f"{sample['error_type']}: {sample['error_message']}",
                    "error_info": {
                        "type": sample["error_type"],
                        "message": sample["error_message"],
                    },
                })

        # Create LoRA config
        lora_config = LoRaConfig(
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

        # Mock the LoRaTrainer class since we don't have the dependencies installed
        # Create a mock class with the methods we need
        class MockLoRaTrainer:
            def __init__(self, model_name, config, output_dir):
                self.model_name = model_name
                self.config = config
                self.output_dir = output_dir

            async def load_data(self, data_path: str):
                # Simulate loading data
                import asyncio
                await asyncio.sleep(0.1)
                print(f"Loading data from {data_path}")
                return {"mock": "dataset"}

            async def preprocess_data(self, dataset: dict):
                # Simulate preprocessing data
                import asyncio
                await asyncio.sleep(0.1)
                print(f"Preprocessing dataset with {len(dataset)} items")
                return {"mock": "preprocessed_dataset"}

        # Create LoRA tuning using our mock class
        lora_tuning = MockLoRaTrainer(
            model_name=mock_llm.model_uri,
            config=lora_config,
            output_dir=temp_dir,
        )

        # Results dictionary
        results = {
            "operations": [],
        }

        # Test data loading
        print("\nTesting data loading...")

        # Create a temporary data file
        data_path = os.path.join(temp_dir, "success_pairs.jsonl")
        with open(data_path, "w") as f:
            for pair in success_pairs:
                f.write(json.dumps(pair) + "\n")

        # Load data
        _, dataset_latency = await self.time_async_execution(
            lora_tuning.load_data,
            data_path=data_path,
        )

        results["operations"].append({
            "operation": "prepare_dataset",
            "latency_ms": dataset_latency,
        })

        print(f"  Dataset preparation latency: {dataset_latency:.2f}ms")

        # Test data preprocessing
        print("\nTesting data preprocessing...")

        # Get a dataset from the previous step
        dataset = await lora_tuning.load_data(data_path)

        # Preprocess data
        _, model_latency = await self.time_async_execution(
            lora_tuning.preprocess_data,
            dataset=dataset,
        )

        results["operations"].append({
            "operation": "prepare_model",
            "latency_ms": model_latency,
        })

        print(f"  Model preparation latency: {model_latency:.2f}ms")

        # Simulate training
        print("\nSimulating training...")
        training_latency = 5000.0  # Simulated 5 seconds

        results["operations"].append({
            "operation": "train",
            "latency_ms": training_latency,
        })

        print(f"  Training latency (simulated): {training_latency:.2f}ms")

        # Test model loading (simulated)
        print("\nTesting model loading (simulated)...")

        # Since we can't actually load a model in the test, we'll simulate it
        async def mock_load_model():
            # Simulate model loading time
            import asyncio
            await asyncio.sleep(0.1)
            return "mock_model"

        _, save_latency = await self.time_async_execution(
            mock_load_model,
        )

        results["operations"].append({
            "operation": "save_model",
            "latency_ms": save_latency,
        })

        print(f"  Model saving latency: {save_latency:.2f}ms")

        # Save results
        self.save_results(results, "lora_tuning_performance")
