# Contributing to Saplings

Thank you for your interest in contributing to Saplings! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/jacobwarren/saplings.git`
3. Set up the development environment:
   ```bash
   cd saplings
   poetry install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Add tests for your changes

4. Run the tests to ensure everything passes:
   ```bash
   poetry run pytest
   ```

5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request from your fork to the main repository

## Coding Standards

- Follow PEP 8 style guide for Python code
- Use type hints for function parameters and return values
- Write docstrings for all public modules, functions, classes, and methods
- Keep lines under 100 characters
- Use meaningful variable and function names

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

## Documentation

- Update documentation for any new features or changes to existing functionality
- Document public APIs with clear examples
- Keep the README.md up to date

## Pull Request Process

1. Ensure your code follows the coding standards
2. Update the documentation as necessary
3. Include tests for your changes
4. Ensure the test suite passes
5. Update the README.md with details of changes if appropriate
6. The pull request will be merged once it has been reviewed and approved

## Adding New Plugins

Saplings is designed to be extensible. To add a new plugin:

1. Create a new module in the appropriate directory (e.g., `src/saplings/plugins/your_plugin/`)
2. Implement the required interface for your plugin type
3. Add entry points in `pyproject.toml` to register your plugin
4. Add tests for your plugin
5. Document your plugin in the plugin's README.md

## License

By contributing to Saplings, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
