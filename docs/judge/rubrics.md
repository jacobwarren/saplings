# Rubric-based Evaluation System

The Saplings JudgeAgent module provides a flexible rubric-based evaluation system for assessing the quality of AI-generated outputs. This document explains how to use the rubric system and how to create custom rubrics.

## What is a Rubric?

A rubric is a scoring guide that defines criteria for evaluating outputs. Each rubric consists of:

- A name and description
- A set of scoring dimensions (e.g., relevance, correctness, coherence)
- Weights for each dimension
- Criteria for different score levels within each dimension

Rubrics help ensure consistent and fair evaluation of outputs by providing clear guidelines for scoring.

## Using Predefined Rubric Templates

Saplings provides several predefined rubric templates for common evaluation scenarios:

- **General**: A general-purpose rubric for evaluating outputs
- **Code**: A rubric for evaluating code outputs
- **Creative**: A rubric for evaluating creative writing outputs
- **Educational**: A rubric for evaluating educational content
- **Factual**: A rubric for evaluating factual content
- **Safety**: A rubric for evaluating content safety

To use a predefined template, simply pass the template name to the `judge_with_template` method:

```python
from saplings.judge import JudgeAgent

judge = JudgeAgent(model=my_model)
result = await judge.judge_with_template(
    output="This is the AI's response.",
    prompt="This is the user's prompt.",
    template="code"  # Use the code evaluation template
)
```

## Creating Custom Rubrics

You can create custom rubrics to suit your specific evaluation needs:

```python
from saplings.judge.config import Rubric, RubricItem, ScoringDimension

# Create a custom rubric
my_rubric = Rubric(
    name="My Custom Rubric",
    description="A custom rubric for my specific use case",
    items=[
        RubricItem(
            dimension=ScoringDimension.RELEVANCE,
            weight=2.0,  # Higher weight for relevance
            description="How relevant the output is to the prompt",
            criteria={
                "0.0": "Completely irrelevant",
                "0.5": "Somewhat relevant",
                "1.0": "Highly relevant",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.CORRECTNESS,
            weight=1.5,  # Medium weight for correctness
            description="How factually correct the output is",
            criteria={
                "0.0": "Contains major errors",
                "0.5": "Contains minor errors",
                "1.0": "Factually correct",
            },
        ),
        # Add more dimensions as needed
    ],
)

# Use the custom rubric
result = await judge.judge(
    output="This is the AI's response.",
    prompt="This is the user's prompt.",
    rubric=my_rubric
)
```

## Loading Rubrics from Files

You can also define rubrics in YAML or JSON files and load them at runtime:

### YAML Example

```yaml
name: My YAML Rubric
description: A rubric defined in YAML
items:
  - dimension: relevance
    weight: 1.0
    description: How relevant the output is
    criteria:
      "0.0": Not relevant
      "0.5": Somewhat relevant
      "1.0": Very relevant
  - dimension: correctness
    weight: 2.0
    description: How correct the output is
    criteria:
      "0.0": Not correct
      "0.5": Somewhat correct
      "1.0": Very correct
```

### JSON Example

```json
{
  "name": "My JSON Rubric",
  "description": "A rubric defined in JSON",
  "items": [
    {
      "dimension": "relevance",
      "weight": 1.0,
      "description": "How relevant the output is",
      "criteria": {
        "0.0": "Not relevant",
        "0.5": "Somewhat relevant",
        "1.0": "Very relevant"
      }
    },
    {
      "dimension": "correctness",
      "weight": 2.0,
      "description": "How correct the output is",
      "criteria": {
        "0.0": "Not correct",
        "0.5": "Somewhat correct",
        "1.0": "Very correct"
      }
    }
  ]
}
```

To load a rubric from a file:

```python
from saplings.judge.rubric import RubricLoader

# Load from YAML
yaml_rubric = RubricLoader.load_from_file("path/to/rubric.yaml")

# Load from JSON
json_rubric = RubricLoader.load_from_file("path/to/rubric.json")

# Use the loaded rubric
result = await judge.judge(
    output="This is the AI's response.",
    prompt="This is the user's prompt.",
    rubric=yaml_rubric
)
```

## Available Scoring Dimensions

Saplings provides the following scoring dimensions:

- **RELEVANCE**: How relevant the output is to the prompt
- **CORRECTNESS**: How factually correct the output is
- **COHERENCE**: How coherent and well-structured the output is
- **HELPFULNESS**: How helpful the output is
- **CONCISENESS**: How concise and to-the-point the output is
- **CREATIVITY**: How creative and original the output is
- **SAFETY**: How safe and appropriate the output is
- **CUSTOM**: For custom dimensions not covered by the predefined ones

You can use any combination of these dimensions in your rubrics.

## Weighted Scoring

The overall score is calculated as a weighted average of the dimension scores, using the weights defined in the rubric. This allows you to emphasize certain dimensions over others based on your evaluation priorities.

For example, if correctness is more important than creativity for your use case, you can assign a higher weight to the correctness dimension.

## Validation

Saplings provides a `RubricValidator` class to validate rubrics:

```python
from saplings.judge.rubric import RubricValidator

# Validate a rubric
errors = RubricValidator.validate(my_rubric)

if errors:
    print("Rubric validation failed:")
    for error in errors:
        print(f"- {error}")
else:
    print("Rubric is valid!")
```

This helps ensure that your rubrics are well-formed and will work correctly with the judge agent.

## Best Practices

- **Be specific**: Define clear criteria for each score level
- **Use appropriate weights**: Assign weights based on the relative importance of each dimension
- **Include multiple dimensions**: Evaluate outputs from multiple perspectives
- **Validate your rubrics**: Use the `RubricValidator` to check for errors
- **Start with templates**: Use predefined templates as a starting point for custom rubrics
- **Document your rubrics**: Include clear descriptions for each dimension and criterion
