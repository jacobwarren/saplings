# Saplings Test Suite

This directory contains the test suite for the Saplings library. The tests are organized into several categories to ensure comprehensive coverage of the library's functionality.

## Test Structure

- **Unit Tests**: Located in `tests/unit/` and component-specific directories
- **Integration Tests**: Located in `tests/integration/`
- **End-to-End Tests**: Located in `tests/e2e/`
- **Security Tests**: Located in `tests/security/`
- **Benchmark Tests**: Located in `tests/benchmarks/`

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Specific Test Categories

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run end-to-end tests
pytest tests/e2e/

# Run security tests
pytest tests/security/

# Run specific component tests
pytest tests/unit/adapters/
pytest tests/unit/memory/
pytest tests/unit/gasa/
pytest tests/unit/services/
```

### Running Benchmark Tests

Benchmark tests are marked with the `benchmark` marker and are skipped by default. To run them:

```bash
pytest tests/benchmarks/ -m benchmark
```

### Running Tests with External APIs

Some tests require API keys for external services like OpenAI, Anthropic, etc. These tests are skipped if the required API keys are not available. To run them, set the appropriate environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
pytest tests/integration/test_model_adapters.py
```

## Test Fixtures

The test suite uses several fixtures to set up the test environment:

- `reset_di`: Resets the dependency injection container before and after each test
- `test_config`: Creates a test configuration
- `test_container`: Creates a test container with the test configuration

## Test Markers

The test suite uses several markers to categorize tests:

- `benchmark`: Marks benchmark tests that measure performance
- `nocov`: Marks tests that should be excluded from coverage reports
- `timeout`: Marks tests with a specific timeout

## Adding New Tests

When adding new tests, follow these guidelines:

1. Place the test in the appropriate directory based on its category
2. Use descriptive test names that clearly indicate what is being tested
3. Use fixtures to set up the test environment
4. Use markers to categorize the test if necessary
5. Skip tests that require external services if the required API keys are not available

## Test Dependencies

The test suite requires the following dependencies:

- pytest
- pytest-cov
- pytest-asyncio
- pytest-timeout

These dependencies are included in the `dev` and `test` groups in the `pyproject.toml` file.
