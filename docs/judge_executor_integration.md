# JudgeAgent and Executor Integration

This document explains how the JudgeAgent and Executor components work together in the Saplings framework to provide verification and refinement of generated outputs.

## Overview

The integration between JudgeAgent and Executor enables:

1. **Quality Verification**: Evaluating generated outputs against multiple dimensions
2. **Structured Feedback**: Providing detailed critiques and improvement suggestions
3. **Refinement Pipeline**: Iteratively improving outputs that don't meet quality thresholds
4. **Flexible Configuration**: Supporting different verification and refinement strategies

## JudgeAgent

The JudgeAgent is responsible for evaluating the quality of generated outputs based on configurable criteria.

### Key Components

#### JudgeConfig

The `JudgeConfig` class defines the configuration for the JudgeAgent:

```python
class JudgeConfig(BaseModel):
    # Scoring settings
    rubric: Rubric
    threshold: float = 0.7  # Threshold for passing verification (0.0 to 1.0)
    
    # Critique settings
    critique_format: CritiqueFormat = CritiqueFormat.STRUCTURED
    include_scores: bool = True
    include_suggestions: bool = True
    
    # Budget settings
    enforce_budget: bool = True
    max_tokens_per_judgment: Optional[int] = None
    max_cost_per_judgment: Optional[float] = None
    
    # Model settings
    model_uri: Optional[str] = None
```

#### Rubric

A `Rubric` defines the evaluation criteria:

```python
class Rubric(BaseModel):
    name: str
    description: str = ""
    items: List[RubricItem] = []
```

Each `RubricItem` represents a dimension to evaluate:

```python
class RubricItem(BaseModel):
    dimension: ScoringDimension
    weight: float = 1.0
    description: str = ""
    criteria: Dict[str, str] = {}
```

The default rubric evaluates outputs on:
- **Relevance**: How relevant the output is to the prompt
- **Correctness**: How factually correct the output is
- **Coherence**: How coherent and well-structured the output is
- **Helpfulness**: How helpful the output is

#### JudgeResult

The `JudgeResult` class represents the result of a judgment:

```python
class JudgeResult(BaseModel):
    output: str
    prompt: str
    passed: bool
    overall_score: float
    dimension_scores: List[DimensionScore] = []
    critique: str = ""
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}
```

### JudgeAgent Methods

#### judge()

The core method of the JudgeAgent is `judge()`, which evaluates an output:

```python
async def judge(self, output: str, prompt: str, rubric: Optional[Rubric] = None) -> JudgeResult:
```

This method:
1. Creates a judgment prompt based on the output, original prompt, and rubric
2. Sends the prompt to the LLM
3. Parses the response to extract scores, critique, and suggestions
4. Determines if the output passes verification based on the threshold
5. Returns a JudgeResult

#### format_critique()

The `format_critique()` method formats a JudgeResult for display:

```python
def format_critique(self, result: JudgeResult) -> str:
```

It supports multiple formats:
- **JSON**: Machine-readable JSON format
- **Markdown**: Human-readable markdown format
- **Structured**: Plain text with structured sections
- **Simple**: Minimal plain text format

## Executor

The Executor is responsible for generating text with speculative execution, verification, and refinement.

### Key Components

#### ExecutorConfig

The `ExecutorConfig` class defines the configuration for the Executor:

```python
class ExecutorConfig(BaseModel):
    # Verification settings
    verification_strategy: VerificationStrategy = VerificationStrategy.BASIC
    verification_threshold: float = 0.7
    
    # Refinement settings
    refinement_strategy: RefinementStrategy = RefinementStrategy.FEEDBACK
    max_refinement_attempts: int = 3
```

#### VerificationStrategy

The `VerificationStrategy` enum defines the strategies for verifying outputs:

```python
class VerificationStrategy(str, Enum):
    NONE = "none"        # No verification
    BASIC = "basic"      # Basic verification with simple checks
    JUDGE = "judge"      # Verification using JudgeAgent
    VALIDATOR = "validator"  # Verification using ValidatorRegistry
    FULL = "full"        # Full verification using both JudgeAgent and ValidatorRegistry
```

#### RefinementStrategy

The `RefinementStrategy` enum defines the strategies for refining rejected outputs:

```python
class RefinementStrategy(str, Enum):
    NONE = "none"        # No refinement
    RETRY = "retry"      # Simple retry with the same prompt
    FEEDBACK = "feedback"  # Retry with feedback from verification
    ITERATIVE = "iterative"  # Iterative refinement with multiple feedback cycles
```

### Executor Methods

#### _verify_output()

The `_verify_output()` method verifies an output using the configured verification strategy:

```python
async def _verify_output(
    self, 
    output: str, 
    prompt: str, 
    verification_strategy: Optional[VerificationStrategy] = None, 
    **kwargs
) -> Tuple[bool, Optional[float], Optional[str]]:
```

This method:
1. Determines the verification strategy to use
2. Applies the appropriate verification logic
3. Returns a tuple of (passed, score, feedback)

For the `JUDGE` strategy, it:
1. Checks if a JudgeAgent is provided
2. Calls the JudgeAgent's `judge()` method
3. Formats the critique using `format_critique()`
4. Returns the verification result

#### _refine_output()

The `_refine_output()` method refines a rejected output using the configured refinement strategy:

```python
async def _refine_output(
    self,
    prompt: str,
    rejected_output: str,
    verification_feedback: Optional[str],
    documents: Optional[List[Document]] = None,
    attempt: int = 1,
    **kwargs,
) -> LLMResponse:
```

This method:
1. Determines the refinement strategy to use
2. Applies the appropriate refinement logic
3. Returns a refined response

For the `FEEDBACK` strategy, it:
1. Creates a new prompt with the original prompt and verification feedback
2. Generates a new response with the feedback prompt

For the `ITERATIVE` strategy, it:
1. Creates a new prompt with the original prompt, rejected output, and verification feedback
2. Generates a new response with the iterative prompt

#### execute()

The `execute()` method is the main entry point for generating text:

```python
async def execute(
    self,
    prompt: str,
    documents: Optional[List[Document]] = None,
    stream: Optional[bool] = None,
    on_draft: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> ExecutionResult:
```

This method:
1. Generates a draft response if speculative execution is enabled
2. Generates a final response
3. Verifies the output using `_verify_output()`
4. Refines the output if verification fails using `_refine_output()`
5. Returns an ExecutionResult

## Integration Flow

The integration between JudgeAgent and Executor follows this flow:

1. **Initialization**:
   - Create a JudgeAgent with a model and configuration
   - Create an Executor with a model, configuration, and the JudgeAgent

2. **Execution**:
   - Call `executor.execute()` with a prompt
   - The Executor generates a response
   - The Executor calls `_verify_output()` to verify the response

3. **Verification**:
   - If using `VerificationStrategy.JUDGE`, the Executor calls `judge_agent.judge()`
   - The JudgeAgent evaluates the output and returns a JudgeResult
   - The Executor formats the critique using `judge_agent.format_critique()`

4. **Refinement**:
   - If verification fails, the Executor calls `_refine_output()`
   - The Executor creates a new prompt based on the refinement strategy
   - The Executor generates a new response
   - The process repeats until verification passes or max attempts are reached

## Example Usage

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig, VerificationStrategy, RefinementStrategy
from saplings.judge import JudgeAgent, JudgeConfig

# Create a model
model = LLM.from_uri("anthropic://claude-3-opus")

# Create a JudgeAgent
judge_config = JudgeConfig(threshold=0.8)
judge_agent = JudgeAgent(model=model, config=judge_config)

# Create an Executor
executor_config = ExecutorConfig(
    verification_strategy=VerificationStrategy.JUDGE,
    refinement_strategy=RefinementStrategy.ITERATIVE,
    max_refinement_attempts=3,
)
executor = Executor(
    model=model,
    config=executor_config,
    judge_agent=judge_agent,
)

# Execute a prompt
prompt = "Explain the concept of quantum computing to a high school student."
result = await executor.execute(prompt)

# Check the result
print(f"Output: {result.text}")
print(f"Verified: {result.verified}")
print(f"Score: {result.verification_score}")
print(f"Feedback: {result.verification_feedback}")
print(f"Refinement attempts: {result.refinement_attempts}")
```

## Advanced Configuration

### Custom Rubrics

You can create custom rubrics for specific evaluation needs:

```python
from saplings.judge import Rubric, RubricItem, ScoringDimension

# Create a custom rubric for code evaluation
code_rubric = Rubric(
    name="Code Evaluation Rubric",
    description="Rubric for evaluating code outputs",
    items=[
        RubricItem(
            dimension=ScoringDimension.CORRECTNESS,
            weight=2.0,  # Higher weight for correctness
            description="How correct and functional the code is",
            criteria={
                "0.0": "Code does not compile or has major errors",
                "0.5": "Code compiles but has minor errors or inefficiencies",
                "1.0": "Code is correct, efficient, and follows best practices",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.COHERENCE,
            weight=1.0,
            description="How well-structured and readable the code is",
            criteria={
                "0.0": "Code is poorly structured and hard to read",
                "0.5": "Code has decent structure but could be more readable",
                "1.0": "Code is well-structured, readable, and well-documented",
            },
        ),
    ],
)

# Use the custom rubric in the JudgeAgent
result = await judge_agent.judge(output=code_output, prompt=code_prompt, rubric=code_rubric)
```

### Custom Verification Strategies

You can implement custom verification logic in the Executor:

```python
# Extend the Executor class with custom verification
class CustomExecutor(Executor):
    async def _verify_output(
        self, output: str, prompt: str, verification_strategy: Optional[VerificationStrategy] = None, **kwargs
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        # Call the parent method for standard strategies
        if verification_strategy != "custom":
            return await super()._verify_output(output, prompt, verification_strategy, **kwargs)
        
        # Custom verification logic
        # ...
        
        return passed, score, feedback
```

## Conclusion

The integration between JudgeAgent and Executor provides a powerful framework for generating high-quality outputs with verification and refinement. By configuring the verification and refinement strategies, you can control how outputs are evaluated and improved to meet your quality standards.
