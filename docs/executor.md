# Execution System

The execution system in Saplings is responsible for generating text with advanced features like speculative execution, GASA integration, streaming output, verification, and refinement.

## Overview

The execution system consists of several key components:

- **Executor**: Main class that handles text generation with advanced features
- **ExecutorConfig**: Configuration for the executor
- **ExecutionResult**: Result of an execution, including text and metadata
- **VerificationStrategy**: Strategies for verifying generated outputs
- **RefinementStrategy**: Strategies for refining rejected outputs

This system provides a powerful foundation for generating high-quality text while optimizing for performance and quality.

## Core Concepts

### Speculative Execution

Speculative execution generates a draft response with a low temperature (more deterministic) before generating the final response with a higher temperature (more creative). This approach can:

- Reduce latency by starting with a faster draft generation
- Improve quality by using the draft as a starting point
- Provide early feedback through draft callbacks

### Verification

Verification ensures that generated outputs meet quality standards. The system supports several verification strategies:

- **None**: No verification
- **Basic**: Simple checks for length and content
- **Judge**: Verification using a JudgeAgent
- **Validator**: Verification using a ValidatorRegistry
- **Full**: Comprehensive verification using both JudgeAgent and ValidatorRegistry
- **Self-Consistency**: Verification by comparing multiple outputs

### Refinement

Refinement improves outputs that fail verification. The system supports several refinement strategies:

- **None**: No refinement
- **Retry**: Simple retry with the same prompt
- **Feedback**: Retry with feedback from verification
- **Iterative**: Iterative refinement with multiple feedback cycles

### GASA Integration

The executor integrates with Graph-Aligned Sparse Attention (GASA) to improve efficiency and grounding:

- Applies GASA masks to focus attention on relevant context
- Supports different fallback strategies for models without sparse attention
- Optimizes prompt composition based on document relationships

### Streaming

The executor supports streaming output for improved user experience:

- Generates text in chunks for real-time display
- Provides callbacks for draft completion and chunk delivery
- Maintains verification and refinement capabilities with streaming

## API Reference

### Executor

```python
class Executor:
    def __init__(
        self,
        model: Optional[LLM] = None,
        config: Optional[ExecutorConfig] = None,
        gasa_config: Optional[GASAConfig] = None,
        dependency_graph: Optional[DependencyGraph] = None,
        judge_agent: Optional[JudgeAgent] = None,
        validator_registry: Optional[ValidatorRegistry] = None,
        trace_manager: Optional[TraceManager] = None,
    ):
        """Initialize the executor."""

    async def execute(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        stream: Optional[bool] = None,
        on_draft: Optional[Callable[[str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a prompt to generate text."""

    async def _generate_draft(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a draft response."""

    async def _generate_final(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a final response."""

    async def _verify_output(
        self,
        output: str,
        prompt: str,
        verification_strategy: Optional[VerificationStrategy] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Verify an output using the configured verification strategy."""

    async def _refine_output(
        self,
        prompt: str,
        rejected_output: str,
        verification_feedback: Optional[str],
        documents: Optional[List[Document]] = None,
        attempt: int = 1,
        **kwargs: Any,
    ) -> LLMResponse:
        """Refine a rejected output."""

    async def _execute_streaming(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        on_draft: Optional[Callable[[str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a prompt with streaming output."""

    async def _apply_gasa(
        self,
        prompt: str,
        documents: List[Document],
        generation_type: str = "final",
        model_supports_sparse_attention: bool = False,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply GASA to a prompt."""
```

### ExecutionResult

```python
class ExecutionResult:
    def __init__(
        self,
        text: str,
        model_uri: str,
        usage: Dict[str, int],
        metadata: Dict[str, Any],
        verified: bool = False,
        verification_score: Optional[float] = None,
        verification_feedback: Optional[str] = None,
        draft_latency_ms: Optional[int] = None,
        final_latency_ms: Optional[int] = None,
        total_latency_ms: Optional[int] = None,
        refinement_attempts: int = 0,
        response: Optional[LLMResponse] = None,
    ):
        """Initialize the execution result."""

    @classmethod
    def from_llm_response(cls, response: LLMResponse, **kwargs) -> "ExecutionResult":
        """Create an execution result from an LLM response."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution result to a dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create an execution result from a dictionary."""
```

### ExecutorConfig

```python
class ExecutorConfig(BaseModel):
    # Speculative execution settings
    enable_speculative_execution: bool = True
    draft_temperature: float = 0.2
    final_temperature: float = 0.7
    max_draft_tokens: Optional[int] = None
    max_final_tokens: Optional[int] = None

    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_size: int = 10

    # GASA settings
    enable_gasa: bool = True
    gasa_config: Optional[Dict[str, Any]] = None

    # Verification settings
    verification_strategy: VerificationStrategy = VerificationStrategy.BASIC
    verification_threshold: float = 0.7

    # Refinement settings
    refinement_strategy: RefinementStrategy = RefinementStrategy.FEEDBACK
    max_refinement_attempts: int = 3

    # Performance settings
    cache_results: bool = True
    cache_dir: Optional[str] = None

    # Logging settings
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> "ExecutorConfig":
        """Create a default configuration."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExecutorConfig":
        """Create a configuration from a dictionary."""
```

### Enums

```python
class VerificationStrategy(str, Enum):
    """Strategy for verifying generated outputs."""
    NONE = "none"  # No verification
    BASIC = "basic"  # Basic verification with simple checks
    JUDGE = "judge"  # Verification using JudgeAgent
    VALIDATOR = "validator"  # Verification using ValidatorRegistry
    FULL = "full"  # Full verification using both JudgeAgent and ValidatorRegistry
    SELF_CONSISTENCY = "self_consistency"  # Verification using self-consistency checking

class RefinementStrategy(str, Enum):
    """Strategy for refining rejected outputs."""
    NONE = "none"  # No refinement
    RETRY = "retry"  # Simple retry with the same prompt
    FEEDBACK = "feedback"  # Retry with feedback from verification
    ITERATIVE = "iterative"  # Iterative refinement with multiple feedback cycles
```

## Usage Examples

### Basic Usage

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create an executor
executor = Executor(
    model=model,
    config=ExecutorConfig(
        enable_speculative_execution=True,
        verification_strategy="basic",
        refinement_strategy="feedback",
    )
)

# Execute a prompt
import asyncio
result = asyncio.run(executor.execute("Explain the concept of graph-based memory"))

# Print the result
print(result.text)
print(f"Verified: {result.verified}")
print(f"Tokens used: {result.usage.get('total_tokens', 0)}")
print(f"Latency: {result.total_latency_ms} ms")
```

### Streaming Output

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create an executor with streaming enabled
executor = Executor(
    model=model,
    config=ExecutorConfig(
        enable_streaming=True,
        stream_chunk_size=10,
    )
)

# Define callbacks
def on_draft(text):
    print(f"Draft: {text}")

def on_chunk(chunk):
    print(f"Chunk: {chunk}", end="", flush=True)

# Execute a prompt with streaming
import asyncio
result = asyncio.run(
    executor.execute(
        "Explain the concept of graph-based memory",
        stream=True,
        on_draft=on_draft,
        on_chunk=on_chunk,
    )
)

# Print the final result
print("\n\nFinal result:")
print(result.text)
```

### GASA Integration

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig
from saplings.memory import MemoryStore, DependencyGraph, Document

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
documents = []
for i in range(5):
    doc = memory.add_document(
        content=f"Document {i} about graph-based memory and its applications.",
        metadata={"source": f"doc_{i}.txt"}
    )
    documents.append(doc)

# Build dependency graph
graph.build_from_memory(memory)

# Create GASA configuration
gasa_config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    fallback_strategy="prompt_composer",
)

# Create an executor with GASA
executor = Executor(
    model=model,
    config=ExecutorConfig(enable_gasa=True),
    gasa_config=gasa_config,
    dependency_graph=graph,
)

# Execute a prompt with documents
import asyncio
result = asyncio.run(
    executor.execute(
        "Summarize the key points about graph-based memory",
        documents=documents,
    )
)

# Print the result
print(result.text)
```

### Verification and Refinement

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig, VerificationStrategy, RefinementStrategy
from saplings.judge import JudgeAgent, JudgeConfig
from saplings.validator.registry import ValidatorRegistry

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a judge agent
judge_agent = JudgeAgent(
    model=model,
    config=JudgeConfig(
        criteria=["relevance", "accuracy", "completeness", "coherence"],
    )
)

# Create a validator registry
validator_registry = ValidatorRegistry()

# Create an executor with full verification and iterative refinement
executor = Executor(
    model=model,
    config=ExecutorConfig(
        verification_strategy=VerificationStrategy.FULL,
        verification_threshold=0.8,
        refinement_strategy=RefinementStrategy.ITERATIVE,
        max_refinement_attempts=3,
    ),
    judge_agent=judge_agent,
    validator_registry=validator_registry,
)

# Execute a prompt
import asyncio
result = asyncio.run(
    executor.execute(
        "Explain the concept of graph-based memory and its applications in AI systems",
    )
)

# Print the result
print(result.text)
print(f"Verified: {result.verified}")
print(f"Verification score: {result.verification_score}")
print(f"Verification feedback: {result.verification_feedback}")
print(f"Refinement attempts: {result.refinement_attempts}")
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.executor import ExecutorConfig, VerificationStrategy, RefinementStrategy

# Create an agent with custom executor configuration
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        executor_config=ExecutorConfig(
            enable_speculative_execution=True,
            verification_strategy=VerificationStrategy.BASIC,
            refinement_strategy=RefinementStrategy.FEEDBACK,
            max_refinement_attempts=2,
        ),
    )
)

# Run a query
import asyncio
result = asyncio.run(agent.run("Explain the concept of graph-based memory"))

# Print the result
print(result)
```

## Advanced Features

### Custom Verification

The executor supports custom verification logic:

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig, VerificationStrategy

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create an executor
executor = Executor(
    model=model,
    config=ExecutorConfig(
        verification_strategy=VerificationStrategy.BASIC,
    )
)

# Define a custom verification function
async def custom_verify(output, prompt, **kwargs):
    # Check if output contains specific keywords
    required_keywords = ["graph", "memory", "nodes", "edges"]
    missing_keywords = [kw for kw in required_keywords if kw not in output.lower()]

    if missing_keywords:
        return False, 0.5, f"Missing required keywords: {', '.join(missing_keywords)}"

    # Check if output is long enough
    if len(output.split()) < 100:
        return False, 0.7, "Output is too short (less than 100 words)"

    # Verification passed
    return True, 1.0, "Output meets all requirements"

# Override the verification method
executor._verify_output = custom_verify

# Execute a prompt
import asyncio
result = asyncio.run(
    executor.execute(
        "Explain the concept of graph-based memory",
    )
)

# Print the result
print(result.text)
print(f"Verified: {result.verified}")
print(f"Verification score: {result.verification_score}")
print(f"Verification feedback: {result.verification_feedback}")
```

### Performance Optimization

The executor includes several performance optimizations:

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create an executor with performance optimizations
executor = Executor(
    model=model,
    config=ExecutorConfig(
        # Enable speculative execution for faster initial responses
        enable_speculative_execution=True,
        draft_temperature=0.1,

        # Cache results to avoid redundant generation
        cache_results=True,
        cache_dir="./cache",

        # Disable verification and refinement for maximum speed
        verification_strategy="none",
        refinement_strategy="none",
    )
)

# Execute multiple prompts
import asyncio
import time

async def benchmark():
    prompts = [
        "Explain the concept of graph-based memory",
        "What are the advantages of graph-based memory?",
        "How does graph-based memory compare to vector-based memory?",
        "Explain the concept of graph-based memory",  # Duplicate to test caching
    ]

    start_time = time.time()

    for prompt in prompts:
        result = await executor.execute(prompt)
        print(f"Prompt: {prompt[:20]}...")
        print(f"Latency: {result.total_latency_ms} ms")
        print()

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")

asyncio.run(benchmark())
```

### Self-Consistency Verification

The executor supports self-consistency verification by comparing multiple outputs:

```python
from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig, VerificationStrategy

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create an executor with self-consistency verification
executor = Executor(
    model=model,
    config=ExecutorConfig(
        verification_strategy=VerificationStrategy.SELF_CONSISTENCY,
    )
)

# Define a custom self-consistency verification function
async def self_consistency_verify(output, prompt, **kwargs):
    # Generate multiple outputs
    num_samples = 3
    samples = []

    for i in range(num_samples):
        # Generate a sample with different temperature
        response = await executor.model.generate(
            prompt,
            temperature=0.7 + (i * 0.1),  # Vary temperature
        )
        samples.append(response.text)

    # Compare the original output with the samples
    consistent_count = 0
    for sample in samples:
        # Simple similarity check (in a real implementation, use semantic similarity)
        similarity = len(set(output.split()) & set(sample.split())) / len(set(output.split() + sample.split()))
        if similarity > 0.7:
            consistent_count += 1

    # Calculate consistency score
    consistency_score = consistent_count / num_samples

    # Determine if output is verified
    verified = consistency_score >= 0.5

    return verified, consistency_score, f"Consistency score: {consistency_score:.2f}"

# Override the verification method
executor._verify_output = self_consistency_verify

# Execute a prompt
import asyncio
result = asyncio.run(
    executor.execute(
        "Explain the concept of graph-based memory",
    )
)

# Print the result
print(result.text)
print(f"Verified: {result.verified}")
print(f"Verification score: {result.verification_score}")
print(f"Verification feedback: {result.verification_feedback}")
```

## Implementation Details

### Verification Process

The verification process works as follows:

1. **Strategy Selection**: Choose the appropriate verification strategy
2. **Verification Execution**: Apply the selected strategy to the output
3. **Score Calculation**: Calculate a verification score (0.0 to 1.0)
4. **Threshold Comparison**: Compare the score to the verification threshold
5. **Feedback Generation**: Generate feedback for failed verification

#### Basic Verification

Basic verification performs simple checks:

- Ensures the output is not empty
- Checks that the output is not too short
- Verifies that the output is not just repeating the prompt

#### Judge Verification

Judge verification uses a JudgeAgent:

- Evaluates the output against specific criteria
- Provides a detailed assessment and score
- Generates feedback for improvement

#### Validator Verification

Validator verification uses a ValidatorRegistry:

- Applies registered validators to the output
- Checks for specific requirements or constraints
- Aggregates validation results

#### Full Verification

Full verification combines Judge and Validator approaches:

- First applies validators to check requirements
- Then uses the judge to evaluate quality
- Provides comprehensive feedback

### Refinement Process

The refinement process works as follows:

1. **Strategy Selection**: Choose the appropriate refinement strategy
2. **Prompt Modification**: Modify the prompt based on the strategy
3. **Regeneration**: Generate a new output with the modified prompt
4. **Verification**: Verify the new output
5. **Iteration**: Repeat if necessary, up to the maximum attempts

#### Retry Refinement

Retry refinement simply retries with the original prompt:

- Uses the same prompt without modifications
- Relies on model randomness for different outputs

#### Feedback Refinement

Feedback refinement incorporates verification feedback:

- Adds feedback to the prompt
- Instructs the model to address the issues
- Provides context about the refinement attempt

#### Iterative Refinement

Iterative refinement builds on previous attempts:

- Includes both the rejected output and feedback
- Asks the model to improve the previous attempt
- Provides context about the refinement progress

### Speculative Execution Process

The speculative execution process works as follows:

1. **Draft Generation**: Generate a draft with low temperature
2. **Draft Callback**: Call the draft callback if provided
3. **Final Generation**: Generate the final output with normal temperature
4. **Verification**: Verify the final output
5. **Refinement**: Refine the output if necessary

### Streaming Process

The streaming process works as follows:

1. **Draft Generation**: Generate a draft if speculative execution is enabled
2. **Draft Callback**: Call the draft callback if provided
3. **Streaming Generation**: Generate the final output in chunks
4. **Chunk Callbacks**: Call the chunk callback for each chunk
5. **Verification and Refinement**: Verify and refine the complete output

## Extension Points

The execution system is designed to be extensible:

### Custom Executor

You can create a custom executor by extending the `Executor` class:

```python
from saplings.executor import Executor, ExecutorConfig, ExecutionResult

class CustomExecutor(Executor):
    async def execute(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        # Custom execution logic
        # ...
        return result

    async def _verify_output(
        self,
        output: str,
        prompt: str,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        # Custom verification logic
        # ...
        return verified, score, feedback

    async def _refine_output(
        self,
        prompt: str,
        rejected_output: str,
        verification_feedback: Optional[str],
        **kwargs: Any,
    ) -> LLMResponse:
        # Custom refinement logic
        # ...
        return response
```

### Custom Verification Strategy

You can create a custom verification strategy by extending the `VerificationStrategy` enum and implementing the corresponding logic:

```python
from enum import Enum
from saplings.executor import VerificationStrategy

# Extend the VerificationStrategy enum
class CustomVerificationStrategy(str, Enum):
    SEMANTIC = "semantic"  # Semantic verification

# Implement the strategy in your executor
class CustomExecutor(Executor):
    async def _verify_output(
        self,
        output: str,
        prompt: str,
        verification_strategy: Optional[VerificationStrategy] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        strategy = verification_strategy or self.config.verification_strategy

        if strategy == "semantic":
            # Implement semantic verification
            # ...
            return verified, score, feedback
        else:
            # Fall back to default verification
            return await super()._verify_output(output, prompt, verification_strategy, **kwargs)
```

### Custom Refinement Strategy

You can create a custom refinement strategy by extending the `RefinementStrategy` enum and implementing the corresponding logic:

```python
from enum import Enum
from saplings.executor import RefinementStrategy

# Extend the RefinementStrategy enum
class CustomRefinementStrategy(str, Enum):
    GUIDED = "guided"  # Guided refinement

# Implement the strategy in your executor
class CustomExecutor(Executor):
    async def _refine_output(
        self,
        prompt: str,
        rejected_output: str,
        verification_feedback: Optional[str],
        documents: Optional[List[Document]] = None,
        attempt: int = 1,
        **kwargs: Any,
    ) -> LLMResponse:
        strategy = self.config.refinement_strategy

        if strategy == "guided":
            # Implement guided refinement
            # ...
            return response
        else:
            # Fall back to default refinement
            return await super()._refine_output(
                prompt, rejected_output, verification_feedback, documents, attempt, **kwargs
            )
```

## Conclusion

The execution system in Saplings provides a powerful foundation for generating high-quality text with advanced features like speculative execution, GASA integration, streaming output, verification, and refinement. By combining these features, it enables the creation of agents that produce more accurate, relevant, and useful outputs while optimizing for performance and user experience.
