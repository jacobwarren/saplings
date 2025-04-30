# JudgeAgent and Executor Quick Start Guide

This guide will help you quickly get started with using the JudgeAgent and Executor components in the Saplings framework.

## Installation

First, make sure you have Saplings installed:

```bash
pip install saplings
```

## Basic Usage

### 1. Create a Model

First, create an LLM model to use with the JudgeAgent and Executor:

```python
from saplings.core.model_adapter import LLM

# Create a model from a URI
model = LLM.from_uri("anthropic://claude-3-opus")

# Or use a specific model implementation
from saplings.core.model_adapter.anthropic import AnthropicLLM
model = AnthropicLLM(api_key="your-api-key", model_name="claude-3-opus")
```

### 2. Create a JudgeAgent

Create a JudgeAgent with the model:

```python
from saplings.judge import JudgeAgent, JudgeConfig

# Create a JudgeAgent with default configuration
judge_agent = JudgeAgent(model=model)

# Or create a JudgeAgent with custom configuration
judge_config = JudgeConfig(
    threshold=0.8,  # Higher threshold for passing verification
    critique_format="markdown",  # Use markdown format for critiques
)
judge_agent = JudgeAgent(model=model, config=judge_config)
```

### 3. Create an Executor

Create an Executor with the model and JudgeAgent:

```python
from saplings.executor import Executor, ExecutorConfig, VerificationStrategy, RefinementStrategy

# Create an Executor with default configuration
executor = Executor(model=model, judge_agent=judge_agent)

# Or create an Executor with custom configuration
executor_config = ExecutorConfig(
    verification_strategy=VerificationStrategy.JUDGE,  # Use JudgeAgent for verification
    refinement_strategy=RefinementStrategy.ITERATIVE,  # Use iterative refinement
    max_refinement_attempts=3,  # Maximum number of refinement attempts
)
executor = Executor(
    model=model,
    config=executor_config,
    judge_agent=judge_agent,
)
```

### 4. Execute a Prompt

Execute a prompt with the Executor:

```python
import asyncio

async def main():
    # Execute a prompt
    prompt = "Explain the concept of quantum computing to a high school student."
    result = await executor.execute(prompt)

    # Print the result
    print(f"Output: {result.text}")
    print(f"Verified: {result.verified}")
    print(f"Score: {result.verification_score}")
    print(f"Feedback: {result.verification_feedback}")
    print(f"Refinement attempts: {result.refinement_attempts}")

# Run the async function
asyncio.run(main())
```

## Advanced Usage

### Using Streaming Output

You can use streaming output to see the generation in real-time:

```python
async def main():
    # Define callbacks for streaming
    def on_draft(text):
        print(f"Draft: {text[:50]}...")

    def on_chunk(text):
        print(f"Chunk: {text}")

    # Execute with streaming
    prompt = "Explain the concept of quantum computing to a high school student."
    result = await executor.execute(
        prompt=prompt,
        stream=True,
        on_draft=on_draft,
        on_chunk=on_chunk,
    )

    # Print the final result
    print(f"Final output: {result.text}")
    print(f"Verified: {result.verified}")
    print(f"Score: {result.verification_score}")

asyncio.run(main())
```

### Using Custom Rubrics

You can create custom rubrics for specific evaluation needs:

```python
from saplings.judge import Rubric, RubricItem, ScoringDimension

# Create a custom rubric
custom_rubric = Rubric(
    name="Educational Content Rubric",
    description="Rubric for evaluating educational content",
    items=[
        RubricItem(
            dimension=ScoringDimension.RELEVANCE,
            weight=1.0,
            description="How relevant the content is to the topic",
            criteria={
                "0.0": "Completely off-topic",
                "0.5": "Somewhat relevant but with tangents",
                "1.0": "Highly focused on the topic",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.CORRECTNESS,
            weight=2.0,  # Higher weight for correctness
            description="How factually correct the content is",
            criteria={
                "0.0": "Contains major factual errors",
                "0.5": "Contains minor factual errors",
                "1.0": "Factually correct",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.COHERENCE,
            weight=1.0,
            description="How well-structured and clear the content is",
            criteria={
                "0.0": "Poorly structured and unclear",
                "0.5": "Somewhat structured but could be clearer",
                "1.0": "Well-structured and very clear",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.HELPFULNESS,
            weight=1.5,  # Higher weight for helpfulness
            description="How helpful the content is for learning",
            criteria={
                "0.0": "Not helpful for learning",
                "0.5": "Somewhat helpful but could be improved",
                "1.0": "Very helpful for learning",
            },
        ),
    ],
)

# Update the JudgeAgent with the custom rubric
judge_config = JudgeConfig(
    rubric=custom_rubric,
    threshold=0.75,
)
judge_agent = JudgeAgent(model=model, config=judge_config)
```

### Direct Use of JudgeAgent

You can use the JudgeAgent directly without the Executor:

```python
async def main():
    # Create a prompt and output
    prompt = "Explain the concept of quantum computing to a high school student."
    output = "Quantum computing uses quantum bits or qubits which can be in multiple states at once, unlike classical bits that are either 0 or 1. This allows quantum computers to perform certain calculations much faster than classical computers."

    # Judge the output using JudgeAgent
    result = await judge_agent.judge(output=output, prompt=prompt)

    # Print the result
    print(f"Passed: {result.passed}")
    print(f"Overall score: {result.overall_score}")
    print(f"Dimension scores:")
    for score in result.dimension_scores:
        print(f"  {score.dimension.value}: {score.score} - {score.explanation}")
    print(f"Critique: {result.critique}")
    print(f"Suggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")

    # Format the critique
    formatted_critique = judge_agent.format_critique(result)
    print(f"Formatted critique:\n{formatted_critique}")

asyncio.run(main())
```

### Using Different Verification Strategies

You can use different verification strategies:

```python
# No verification
executor.config.verification_strategy = VerificationStrategy.NONE

# Basic verification (simple checks)
executor.config.verification_strategy = VerificationStrategy.BASIC

# Judge verification (using JudgeAgent)
executor.config.verification_strategy = VerificationStrategy.JUDGE

# Full verification (using both JudgeAgent and ValidatorRegistry)
executor.config.verification_strategy = VerificationStrategy.FULL
```

### Using Different Refinement Strategies

You can use different refinement strategies:

```python
# No refinement
executor.config.refinement_strategy = RefinementStrategy.NONE

# Simple retry
executor.config.refinement_strategy = RefinementStrategy.RETRY

# Feedback refinement
executor.config.refinement_strategy = RefinementStrategy.FEEDBACK

# Iterative refinement
executor.config.refinement_strategy = RefinementStrategy.ITERATIVE
```

## Configuration Reference

### JudgeConfig Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `rubric` | Rubric for evaluation | Default rubric |
| `threshold` | Threshold for passing verification (0.0 to 1.0) | 0.7 |
| `critique_format` | Format for critique output (simple, structured, json, markdown) | structured |
| `include_scores` | Whether to include scores in the critique | True |
| `include_suggestions` | Whether to include improvement suggestions in the critique | True |
| `enforce_budget` | Whether to enforce budget constraints | True |
| `max_tokens_per_judgment` | Maximum tokens per judgment | None |
| `max_cost_per_judgment` | Maximum cost per judgment in USD | None |
| `model_uri` | URI of the model to use for judgment | None |

### ExecutorConfig Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `verification_strategy` | Strategy for verifying generated outputs | BASIC |
| `verification_threshold` | Threshold for verification (0.0 to 1.0) | 0.7 |
| `refinement_strategy` | Strategy for refining rejected outputs | FEEDBACK |
| `max_refinement_attempts` | Maximum number of refinement attempts | 3 |
| `enable_speculative_execution` | Whether to enable speculative execution | True |
| `draft_temperature` | Temperature for draft generation | 0.2 |
| `final_temperature` | Temperature for final generation | 0.7 |
| `enable_streaming` | Whether to enable streaming output | True |
| `stream_chunk_size` | Number of tokens to generate per streaming chunk | 10 |
| `enable_gasa` | Whether to enable GASA | True |
| `cache_results` | Whether to cache results | True |

## Troubleshooting

### JudgeAgent Issues

- **Verification Always Fails**: Try lowering the threshold in JudgeConfig
- **Slow Verification**: Check the model's performance or use a faster model
- **Poor Quality Feedback**: Try a different model or adjust the rubric

### Executor Issues

- **Refinement Not Improving**: Try a different refinement strategy
- **Too Many Refinement Attempts**: Reduce max_refinement_attempts
- **Streaming Not Working**: Ensure enable_streaming is True

## Next Steps

- Learn more about [GASA mask injection](gasa.md)
- Explore [speculative execution](speculative_execution.md)
- Understand [streaming output](streaming.md)
- Integrate with [memory and documents](memory.md)
