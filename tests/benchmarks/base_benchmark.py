"""
Base benchmark class for Saplings benchmarks.

This module provides a base class for all benchmark tests in the Saplings framework.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        """Initialize the mock tokenizer."""
        self.unk_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        """Tokenize text."""
        # Simple tokenization by splitting on spaces
        tokens = text.split()

        # Create a mock tokens object with PyTorch-like attributes
        class MockTokens:
            def __init__(self, tokens):
                # Create a numpy array to simulate PyTorch tensor
                import numpy as np
                self.input_ids = np.array([list(range(len(tokens)))], dtype=np.int32)

        return MockTokens(tokens)

    def convert_tokens_to_ids(self, token):
        """Convert token to ID."""
        # Simple mock implementation
        return 1 if token else 0


class BaseBenchmark:
    """Base class for all benchmark tests."""

    # Number of runs for each benchmark (reduced for faster testing)
    NUM_RUNS = 2

    # Whether to save benchmark results to disk
    SAVE_RESULTS = True

    # Output directory for benchmark results
    OUTPUT_DIR = "benchmark_results"

    @classmethod
    def setup_class(cls):
        """Set up the benchmark class."""
        # Create output directory if it doesn't exist
        if cls.SAVE_RESULTS:
            os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage in MB.

        Returns:
            float: Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0

    @staticmethod
    def time_execution(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Time the execution of a function.

        Args:
            func: Function to time
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Tuple[Any, float]: Function result and execution time in milliseconds
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms

    @staticmethod
    async def time_async_execution(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Time the execution of a function, which may be async or not.

        Args:
            func: Function to time (async or not)
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Tuple[Any, float]: Function result and execution time in milliseconds
        """
        import inspect
        start_time = time.time()

        # Check if the function is a coroutine function
        if inspect.iscoroutinefunction(func):
            # Async function
            result = await func(*args, **kwargs)
        else:
            # Regular function
            result = func(*args, **kwargs)

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms

    @classmethod
    def save_results(cls, results: Dict[str, Any], name: str) -> str:
        """
        Save benchmark results to disk.

        Args:
            results: Benchmark results
            name: Name of the benchmark

        Returns:
            str: Path to the saved results file
        """
        if not cls.SAVE_RESULTS:
            return ""

        # Add timestamp
        results["timestamp"] = datetime.now().isoformat()

        # Create filename
        filename = f"{cls.OUTPUT_DIR}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Save results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        return filename

    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of values.

        Args:
            values: List of values

        Returns:
            Dict[str, float]: Statistics
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }


class MockBenchmarkLLM(LLM):
    """Mock LLM for benchmarking."""

    def __init__(self, model_uri: str, **kwargs):
        """Initialize the mock LLM."""
        self.model_uri = model_uri
        self.kwargs = kwargs
        self.generate_calls = []
        self.streaming_calls = []
        self.response_time_ms = kwargs.get("response_time_ms", 100)
        self.token_count = kwargs.get("token_count", 100)

        # Add a simple tokenizer
        self.tokenizer = MockTokenizer()

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from the model."""
        # Record the call
        self.generate_calls.append({
            "prompt": prompt,
            "kwargs": kwargs,
        })

        # Simulate processing time
        await self._simulate_processing()

        # Return a mock response
        return LLMResponse(
            text=f"Response to: {prompt[:50]}...",
            model_uri=self.model_uri,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": self.token_count,
                "total_tokens": len(prompt.split()) + self.token_count,
            },
            metadata={"model": "benchmark-model"},
        )

    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate text from the model with streaming output."""
        # Record the call
        self.streaming_calls.append({
            "prompt": prompt,
            "kwargs": kwargs,
        })

        # Create a response
        response = f"Response to: {prompt[:50]}..."
        chunks = [response[i:i+5] for i in range(0, len(response), 5)]

        # Return chunks as an async generator
        for chunk in chunks:
            # Simulate processing time for each chunk
            await self._simulate_processing(chunk_size=5)
            yield chunk

    async def _simulate_processing(self, chunk_size: int = None):
        """
        Simulate processing time.

        Args:
            chunk_size: Size of the chunk being processed
        """
        if chunk_size:
            # Scale processing time based on chunk size
            processing_time = self.response_time_ms * (chunk_size / 100) / 1000
        else:
            processing_time = self.response_time_ms / 1000

        import asyncio
        await asyncio.sleep(processing_time)

    def get_metadata(self) -> ModelMetadata:
        """Get metadata for the model."""
        return ModelMetadata(
            name="benchmark-model",
            provider="benchmark-provider",
            version="1.0",
            capabilities=[],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        # Simple estimation: 1 token per 4 characters
        return max(1, len(text) // 4)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            float: Estimated cost in USD
        """
        # Mock cost: $0.001 per 1K prompt tokens, $0.002 per 1K completion tokens
        prompt_cost = prompt_tokens * 0.001 / 1000
        completion_cost = completion_tokens * 0.002 / 1000
        return prompt_cost + completion_cost
