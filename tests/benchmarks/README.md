# Saplings Benchmark Tests

This directory contains benchmark tests for the Saplings framework. These tests measure the performance of various components of the framework, including GASA, retrieval, planning, execution, and self-healing.

## Running Benchmark Tests

Benchmark tests are designed to be run separately from regular unit tests, as they can be resource-intensive and time-consuming. They are also excluded from CI runs by default to prevent hanging issues.

To run the benchmark tests:

```bash
# Run all benchmark tests
pytest tests/benchmarks/

# Run a specific benchmark test
pytest tests/benchmarks/test_gasa_benchmark.py

# Run a specific test function
pytest tests/benchmarks/test_self_heal_benchmark.py::TestSelfHealBenchmark::test_patch_generator
```

## Safety Measures

The benchmark tests include several safety measures to prevent hanging issues:

1. **Execution Guards**: Code samples with potential infinite recursion bugs include guards to prevent actual infinite recursion during testing.

2. **Timeouts**: All benchmark tests have timeouts to prevent them from running indefinitely.

3. **Mock Components**: Many components are mocked to avoid dependencies on external services or resources.

## Benchmark Results

Benchmark results are saved to the `benchmark_results` directory by default. Each benchmark run creates a new JSON file with a timestamp in the filename.

## Warning

Some of the code samples in the benchmark tests intentionally contain bugs (like infinite recursion) for testing error detection and self-healing capabilities. These samples are intended for static analysis only, not for execution. They include guards to prevent actual infinite recursion, but should still be handled with care.
