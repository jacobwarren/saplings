# Saplings Judge

This package provides judge functionality for Saplings agents.

## API Structure

The judge module follows the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.judge` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the judge components, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import (
    JudgeAgent,
    JudgeConfig,
    Rubric,
    RubricItem
)

# Alternative: Import directly from the API module
from saplings.api.judge import (
    JudgeAgent,
    JudgeConfig,
    Rubric,
    RubricItem
)
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.judge.judge_agent import JudgeAgent  # Wrong
```

## Available Components

The following judge components are available:

- `JudgeAgent`: Agent for judging outputs
- `JudgeConfig`: Configuration for the judge agent
- `JudgeResult`: Result of a judge evaluation
- `Rubric`: Rubric for evaluating outputs
- `RubricItem`: Item in a rubric for evaluating outputs
- `ScoringDimension`: Dimension for scoring outputs
- `CritiqueFormat`: Format for judge critiques

## Judge Agent

The judge agent evaluates outputs based on a rubric:

```python
from saplings import JudgeAgent, JudgeConfig, Rubric, RubricItem

# Create a rubric
rubric = Rubric(
    name="Code Quality Rubric",
    description="Evaluates the quality of code",
    items=[
        RubricItem(
            name="Correctness",
            description="Does the code work as expected?",
            weight=0.4
        ),
        RubricItem(
            name="Readability",
            description="Is the code easy to read and understand?",
            weight=0.3
        ),
        RubricItem(
            name="Efficiency",
            description="Is the code efficient?",
            weight=0.3
        )
    ]
)

# Create a judge config
config = JudgeConfig(
    rubric=rubric,
    critique_format=CritiqueFormat.MARKDOWN,
    provider="openai",
    model_name="gpt-4o"
)

# Create a judge agent
judge = JudgeAgent(config=config)

# Judge an output
result = judge.judge(
    task="Write a function to calculate the factorial of a number",
    output="""
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """
)

# Print the result
print(f"Score: {result.score}")
print(f"Critique: {result.critique}")
```

## Rubrics

Rubrics define the criteria for evaluating outputs:

```python
from saplings import Rubric, RubricItem

# Create a rubric
rubric = Rubric(
    name="Essay Rubric",
    description="Evaluates the quality of an essay",
    items=[
        RubricItem(
            name="Content",
            description="Does the essay address the topic?",
            weight=0.4
        ),
        RubricItem(
            name="Organization",
            description="Is the essay well-organized?",
            weight=0.3
        ),
        RubricItem(
            name="Grammar",
            description="Is the grammar correct?",
            weight=0.3
        )
    ]
)
```

## Implementation Details

The judge implementations are located in the `_internal` directory:

- `_internal/judge_agent.py`: Implementation of the judge agent
- `_internal/config.py`: Judge configuration
- `_internal/rubric.py`: Rubric implementation

These internal implementations are wrapped by the public API in `saplings.api.judge` to provide stability annotations and a consistent interface.
