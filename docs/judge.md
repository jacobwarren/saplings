# Judge System

The Judge system in Saplings provides a powerful framework for evaluating outputs based on scoring criteria and providing structured feedback.

## Overview

The Judge system consists of several key components:

- **JudgeAgent**: Main class for evaluating outputs and providing feedback
- **JudgeConfig**: Configuration for the judge agent
- **Rubric**: Defines criteria for evaluating outputs
- **RubricItem**: Individual item in a rubric with dimension, weight, and criteria
- **ScoringDimension**: Predefined dimensions for scoring outputs
- **CritiqueFormat**: Format for critique output
- **RubricTemplate**: Predefined rubric templates for different use cases

This system enables agents to evaluate their own outputs or the outputs of other agents, providing a foundation for self-improvement and quality control.

## Core Concepts

### Rubrics

Rubrics define the criteria for evaluating outputs. Each rubric has:

- **Name**: Identifier for the rubric
- **Description**: Purpose of the rubric
- **Items**: List of RubricItems that define the dimensions and criteria

Rubrics provide a structured way to evaluate outputs across multiple dimensions, with each dimension having its own weight and criteria.

### Scoring Dimensions

Scoring dimensions represent different aspects of an output that can be evaluated:

- **Relevance**: How relevant the output is to the prompt
- **Correctness**: How factually correct the output is
- **Coherence**: How coherent and well-structured the output is
- **Conciseness**: How concise the output is
- **Helpfulness**: How helpful the output is
- **Creativity**: How creative the output is
- **Safety**: How safe and appropriate the output is
- **Custom**: Custom dimensions defined by the user

Each dimension can have a different weight in the overall score, allowing for customized evaluation based on the specific requirements.

### Critique Formats

The Judge system supports several formats for critique output:

- **Simple**: Basic text feedback
- **Structured**: Feedback organized into sections
- **JSON**: Structured feedback in JSON format
- **Markdown**: Feedback formatted with Markdown

These formats provide flexibility in how feedback is presented and consumed, making it easier to integrate with different systems and workflows.

### Rubric Templates

The Judge system includes several predefined rubric templates for common use cases:

- **General**: General-purpose evaluation
- **Code**: Evaluation of code outputs
- **Creative**: Evaluation of creative writing
- **Educational**: Evaluation of educational content
- **Factual**: Evaluation of factual content
- **Safety**: Evaluation of safety and appropriateness

These templates provide a starting point for creating custom rubrics, making it easier to get started with the Judge system.

## API Reference

### JudgeAgent

```python
class JudgeAgent:
    def __init__(
        self,
        model: Optional[LLM] = None,
        config: Optional[JudgeConfig] = None,
    ):
        """Initialize the judge agent."""

    async def judge(
        self,
        output: str,
        prompt: str,
        rubric: Optional[Rubric] = None,
        template: Optional[str] = None,
    ) -> JudgeResult:
        """Judge an output."""

    async def judge_with_template(
        self,
        output: str,
        prompt: str,
        template: str,
    ) -> JudgeResult:
        """Judge an output using a predefined rubric template."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the judge agent."""
```

### JudgeResult

```python
class JudgeResult(BaseModel):
    output: str  # Output that was judged
    prompt: str  # Prompt that generated the output
    passed: bool  # Whether the output passed verification
    overall_score: float  # Overall score (0.0 to 1.0)
    dimension_scores: List[DimensionScore]  # Scores for individual dimensions
    critique: str  # Critique of the output
    suggestions: List[str]  # Suggestions for improvement
    metadata: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert the judgment result to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeResult":
        """Create a judgment result from a dictionary."""
```

### DimensionScore

```python
class DimensionScore(BaseModel):
    dimension: ScoringDimension  # Dimension being scored
    score: float  # Score for this dimension (0.0 to 1.0)
    explanation: str  # Explanation for the score
```

### JudgeConfig

```python
class JudgeConfig(BaseModel):
    # Scoring settings
    rubric: Rubric = Rubric.default()  # Rubric for evaluation
    threshold: float = 0.7  # Threshold for passing verification (0.0 to 1.0)

    # Critique settings
    critique_format: CritiqueFormat = CritiqueFormat.STRUCTURED  # Format for critique output
    include_scores: bool = True  # Whether to include scores in the critique
    include_suggestions: bool = True  # Whether to include improvement suggestions

    # Budget settings
    enforce_budget: bool = True  # Whether to enforce budget constraints
    max_tokens_per_judgment: Optional[int] = None  # Maximum tokens per judgment
    max_cost_per_judgment: Optional[float] = None  # Maximum cost per judgment in USD

    # Model settings
    model_uri: Optional[str] = None  # URI of the model to use for judgment

    @classmethod
    def default(cls) -> "JudgeConfig":
        """Create a default configuration."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "JudgeConfig":
        """Create a configuration from a dictionary."""
```

### Rubric

```python
class Rubric(BaseModel):
    name: str  # Name of the rubric
    description: str  # Description of the rubric
    items: List[RubricItem]  # Items in the rubric

    @classmethod
    def default(cls) -> "Rubric":
        """Create a default rubric."""
```

### RubricItem

```python
class RubricItem(BaseModel):
    dimension: ScoringDimension  # Dimension to score
    weight: float = 1.0  # Weight of this dimension in the overall score
    description: str  # Description of what this dimension measures
    criteria: Dict[str, str]  # Criteria for different score levels
```

### Enums

```python
class ScoringDimension(str, Enum):
    """Dimensions for scoring outputs."""
    RELEVANCE = "relevance"  # How relevant the output is to the prompt
    CORRECTNESS = "correctness"  # How factually correct the output is
    COHERENCE = "coherence"  # How coherent and well-structured the output is
    CONCISENESS = "conciseness"  # How concise the output is
    HELPFULNESS = "helpfulness"  # How helpful the output is
    CREATIVITY = "creativity"  # How creative the output is
    SAFETY = "safety"  # How safe and appropriate the output is
    CUSTOM = "custom"  # Custom dimension defined by the user

class CritiqueFormat(str, Enum):
    """Format for critique output."""
    SIMPLE = "simple"  # Simple text feedback
    STRUCTURED = "structured"  # Structured feedback with sections
    JSON = "json"  # JSON-formatted feedback
    MARKDOWN = "markdown"  # Markdown-formatted feedback

class RubricTemplate(str, Enum):
    """Predefined rubric templates."""
    GENERAL = "general"  # General-purpose evaluation
    CODE = "code"  # Code evaluation
    CREATIVE = "creative"  # Creative writing evaluation
    EDUCATIONAL = "educational"  # Educational content evaluation
    FACTUAL = "factual"  # Factual content evaluation
    SAFETY = "safety"  # Safety evaluation
```

## Usage Examples

### Basic Usage

```python
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    )
)

# Judge an output
import asyncio
result = asyncio.run(judge.judge(
    output="The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum.",
    prompt="What is the capital of France and what is it known for?",
))

# Print the result
print(f"Passed: {result.passed}")
print(f"Overall score: {result.overall_score}")
print(f"Critique: {result.critique}")
print(f"Suggestions: {result.suggestions}")

# Print dimension scores
for score in result.dimension_scores:
    print(f"{score.dimension}: {score.score} - {score.explanation}")
```

### Using Rubric Templates

```python
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    )
)

# Judge an output using the code template
import asyncio
result = asyncio.run(judge.judge_with_template(
    output="""
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Calculate the 10th Fibonacci number
result = fibonacci(10)
print(f"The 10th Fibonacci number is {result}")
""",
    prompt="Write a Python function to calculate the nth Fibonacci number.",
    template="code",
))

# Print the result
print(f"Passed: {result.passed}")
print(f"Overall score: {result.overall_score}")
print(f"Critique: {result.critique}")
print(f"Suggestions: {result.suggestions}")

# Print dimension scores
for score in result.dimension_scores:
    print(f"{score.dimension}: {score.score} - {score.explanation}")
```

### Custom Rubric

```python
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, RubricItem, ScoringDimension

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a custom rubric
custom_rubric = Rubric(
    name="API Documentation Rubric",
    description="Rubric for evaluating API documentation",
    items=[
        RubricItem(
            dimension=ScoringDimension.CORRECTNESS,
            weight=2.0,
            description="How accurate and correct the documentation is",
            criteria={
                "0.0": "Contains major errors or inaccuracies",
                "0.25": "Contains several errors or inaccuracies",
                "0.5": "Contains minor errors or inaccuracies",
                "0.75": "Mostly accurate with minor issues",
                "1.0": "Completely accurate and correct",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.COHERENCE,
            weight=1.5,
            description="How well-structured and organized the documentation is",
            criteria={
                "0.0": "Poorly structured and difficult to follow",
                "0.25": "Somewhat disorganized with unclear structure",
                "0.5": "Adequately structured but could be improved",
                "0.75": "Well-structured with minor organizational issues",
                "1.0": "Excellently structured and very easy to follow",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.CONCISENESS,
            weight=1.0,
            description="How concise and to-the-point the documentation is",
            criteria={
                "0.0": "Extremely verbose with much irrelevant information",
                "0.25": "Verbose with significant irrelevant information",
                "0.5": "Somewhat concise but could be more focused",
                "0.75": "Mostly concise with minor verbosity",
                "1.0": "Very concise and to-the-point",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.HELPFULNESS,
            weight=1.5,
            description="How helpful the documentation is for developers",
            criteria={
                "0.0": "Not helpful for developers",
                "0.25": "Minimally helpful with major gaps",
                "0.5": "Somewhat helpful but missing important information",
                "0.75": "Helpful with minor gaps",
                "1.0": "Very helpful with comprehensive information",
            },
        ),
    ],
)

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    )
)

# Judge an output using the custom rubric
import asyncio
result = asyncio.run(judge.judge(
    output="""
# User API

## GET /users/{id}

Retrieves a user by their ID.

### Parameters

- `id` (path, required): The ID of the user to retrieve.

### Response

```json
{
  "id": "123",
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2023-01-01T00:00:00Z"
}
```

### Errors

- `404 Not Found`: User not found.
- `500 Internal Server Error`: Server error.
""",
    prompt="Write API documentation for a GET endpoint that retrieves a user by ID.",
    rubric=custom_rubric,
))

# Print the result
print(f"Passed: {result.passed}")
print(f"Overall score: {result.overall_score}")
print(f"Critique: {result.critique}")
print(f"Suggestions: {result.suggestions}")

# Print dimension scores
for score in result.dimension_scores:
    print(f"{score.dimension}: {score.score} - {score.explanation}")
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    )
)

# Create an agent with the judge
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        judge_agent=judge,
    )
)

# Run a task
import asyncio
result = asyncio.run(agent.run(
    "Explain the concept of graph-based memory and its advantages."
))

# The agent will use the judge to evaluate its output before returning it
print(result)
```

## Advanced Features

### JSON Critique Format

The Judge system supports JSON-formatted critiques for easier parsing and integration:

```python
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig, CritiqueFormat

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent with JSON critique format
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format=CritiqueFormat.JSON,
    )
)

# Judge an output
import asyncio
result = asyncio.run(judge.judge(
    output="The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum.",
    prompt="What is the capital of France and what is it known for?",
))

# Parse the critique as JSON
import json
critique_json = json.loads(result.critique)
print(json.dumps(critique_json, indent=2))
```

### Self-Improvement Loop

The Judge system can be used in a self-improvement loop to help agents learn from their mistakes:

```python
from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig
from saplings.self_heal import PatchGenerator

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.8,
        critique_format="structured",
    )
)

# Create a patch generator
patch_generator = PatchGenerator(model=model)

# Create an agent
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_self_healing=True,
    )
)

# Self-improvement loop
import asyncio

async def self_improvement_loop(question, max_iterations=3):
    for i in range(max_iterations):
        print(f"\nIteration {i+1}:")

        # Generate an answer
        answer = await agent.run(question)
        print(f"Answer: {answer}")

        # Judge the answer
        judgment = await judge.judge(
            output=answer,
            prompt=question,
        )
        print(f"Score: {judgment.overall_score}")
        print(f"Critique: {judgment.critique}")

        # Check if the answer is good enough
        if judgment.passed:
            print("Answer passed verification!")
            return answer

        # Generate a patch
        patch = await patch_generator.generate_patch(
            prompt=question,
            output=answer,
            feedback=judgment.critique,
        )
        print(f"Patch: {patch}")

        # Apply the patch (in a real implementation, this would update the agent's behavior)
        # For this example, we'll just continue to the next iteration

    print("Reached maximum iterations without a passing answer.")
    return answer

# Run the self-improvement loop
question = "Explain the concept of quantum computing in simple terms."
final_answer = asyncio.run(self_improvement_loop(question))
```

### Custom Scoring Dimensions

The Judge system supports custom scoring dimensions for specialized evaluation:

```python
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, RubricItem, ScoringDimension

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a custom rubric with custom dimensions
custom_rubric = Rubric(
    name="Translation Quality Rubric",
    description="Rubric for evaluating translation quality",
    items=[
        RubricItem(
            dimension="accuracy",  # Custom dimension
            weight=2.0,
            description="How accurately the translation conveys the original meaning",
            criteria={
                "0.0": "Completely inaccurate translation",
                "0.25": "Major inaccuracies in translation",
                "0.5": "Some inaccuracies in translation",
                "0.75": "Minor inaccuracies in translation",
                "1.0": "Perfectly accurate translation",
            },
        ),
        RubricItem(
            dimension="fluency",  # Custom dimension
            weight=1.5,
            description="How natural and fluent the translation sounds in the target language",
            criteria={
                "0.0": "Completely unnatural and non-fluent",
                "0.25": "Major issues with fluency",
                "0.5": "Somewhat fluent but with noticeable issues",
                "0.75": "Mostly fluent with minor issues",
                "1.0": "Perfectly fluent and natural",
            },
        ),
        RubricItem(
            dimension="style",  # Custom dimension
            weight=1.0,
            description="How well the translation preserves the style of the original text",
            criteria={
                "0.0": "Completely different style from the original",
                "0.25": "Major differences in style",
                "0.5": "Some differences in style",
                "0.75": "Minor differences in style",
                "1.0": "Perfectly preserves the original style",
            },
        ),
    ],
)

# Create a judge agent
judge = JudgeAgent(
    model=model,
    config=JudgeConfig(
        threshold=0.7,
        critique_format="structured",
    )
)

# Judge a translation
import asyncio
result = asyncio.run(judge.judge(
    output="Bonjour, comment allez-vous aujourd'hui?",
    prompt="Translate the following English text to French: 'Hello, how are you today?'",
    rubric=custom_rubric,
))

# Print the result
print(f"Passed: {result.passed}")
print(f"Overall score: {result.overall_score}")
print(f"Critique: {result.critique}")
print(f"Suggestions: {result.suggestions}")

# Print dimension scores
for score in result.dimension_scores:
    print(f"{score.dimension}: {score.score} - {score.explanation}")
```

## Implementation Details

### Judgment Process

The judgment process works as follows:

1. **Prompt Creation**: Create a prompt for the judge model that includes the original prompt, the output to be judged, and the rubric
2. **Model Invocation**: Send the prompt to the model and get a response
3. **Response Parsing**: Parse the model's response to extract scores, critique, and suggestions
4. **Score Calculation**: Calculate the overall score based on the dimension scores and weights
5. **Result Creation**: Create a JudgeResult object with all the judgment information

### Rubric Templates

The Judge system includes several predefined rubric templates:

#### General Template

The general template includes dimensions for relevance, correctness, coherence, and helpfulness, with equal weights for all dimensions.

#### Code Template

The code template includes dimensions for correctness, coherence, conciseness, and safety, with higher weights for correctness and safety to emphasize the importance of functional and secure code.

#### Creative Template

The creative template includes dimensions for creativity, coherence, engagement, and originality, with higher weights for creativity and engagement to emphasize the importance of creative and engaging content.

#### Educational Template

The educational template includes dimensions for correctness, coherence, helpfulness, and engagement, with higher weights for correctness and helpfulness to emphasize the importance of accurate and helpful educational content.

#### Factual Template

The factual template includes dimensions for correctness, relevance, coherence, and conciseness, with a much higher weight for correctness to emphasize the importance of factual accuracy.

#### Safety Template

The safety template includes dimensions for safety, appropriateness, bias, and harm prevention, with higher weights for safety and harm prevention to emphasize the importance of safe and appropriate content.

### Budget Enforcement

The Judge system includes budget enforcement to prevent excessive token usage:

- **Token Limit**: Maximum number of tokens per judgment
- **Cost Limit**: Maximum cost per judgment in USD

If either limit is exceeded, the judgment will be aborted and an error will be raised.

## Extension Points

The Judge system is designed to be extensible:

### Custom Rubrics

You can create custom rubrics for specialized evaluation:

```python
from saplings.judge import Rubric, RubricItem, ScoringDimension

# Create a custom rubric
custom_rubric = Rubric(
    name="Custom Rubric",
    description="A custom rubric for specialized evaluation",
    items=[
        RubricItem(
            dimension=ScoringDimension.CUSTOM,
            weight=1.0,
            description="Custom dimension",
            criteria={
                "0.0": "Poor",
                "0.5": "Average",
                "1.0": "Excellent",
            },
        ),
        # Add more items as needed
    ],
)
```

### Custom Critique Formats

You can create custom critique formats by extending the `CritiqueFormat` enum:

```python
from enum import Enum
from saplings.judge import CritiqueFormat

# Extend the CritiqueFormat enum
class CustomCritiqueFormat(str, Enum):
    HTML = "html"  # HTML-formatted feedback
    XML = "xml"  # XML-formatted feedback

# Use the custom format in your config
from saplings.judge import JudgeConfig

config = JudgeConfig(
    critique_format="html",  # Will be converted to the custom format
)
```

### Custom Judgment Logic

You can create custom judgment logic by extending the `JudgeAgent` class:

```python
from saplings.judge import JudgeAgent, JudgeResult

class CustomJudgeAgent(JudgeAgent):
    async def judge(self, output, prompt, rubric=None, template=None):
        # Call the parent method to get the base result
        result = await super().judge(output, prompt, rubric, template)

        # Add custom logic
        # For example, add a custom check for specific keywords
        if "important_keyword" in output.lower():
            # Boost the score for outputs that contain the keyword
            result.overall_score = min(1.0, result.overall_score + 0.1)
            result.passed = result.overall_score >= self.config.threshold

        return result
```

## Conclusion

The Judge system in Saplings provides a powerful framework for evaluating outputs based on scoring criteria and providing structured feedback. By using rubrics, scoring dimensions, and critique formats, it enables agents to evaluate their own outputs or the outputs of other agents, providing a foundation for self-improvement and quality control.
