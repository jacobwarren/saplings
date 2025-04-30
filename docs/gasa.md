# Graph-Aligned Sparse Attention (GASA)

## Overview

Graph-Aligned Sparse Attention (GASA) is a technique that improves the efficiency and effectiveness of attention mechanisms in transformer models by leveraging the structure of document dependency graphs. GASA injects a binary attention mask—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are ≤ h hops apart in the graph, while routing others through a lightweight global summary token.

This approach:
- Reduces computational cost (up to 40% fewer FLOPs)
- Improves grounding by focusing the model's attention on relevant context
- Enhances performance on tasks requiring complex reasoning over multiple documents

## Key Components

### MaskBuilder

The `MaskBuilder` class is the core component of GASA, responsible for building sparse attention masks based on document dependency graphs:

```python
class MaskBuilder:
    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        # Initialize the mask builder
```

The `MaskBuilder` takes a dependency graph, configuration, and tokenizer as input, and provides methods for building attention masks.

### MaskFormat

GASA supports multiple mask formats:

```python
class MaskFormat(str, Enum):
    DENSE = "dense"  # Dense matrix (numpy array)
    SPARSE = "sparse"  # Sparse matrix (scipy.sparse)
    BLOCK_SPARSE = "block_sparse"  # Block-sparse format (list of blocks)
```

### MaskType

GASA supports different types of attention masks:

```python
class MaskType(str, Enum):
    ATTENTION = "attention"  # Regular attention mask (0 = masked, 1 = attend)
    GLOBAL_ATTENTION = "global_attention"  # Global attention mask (1 = global attention)
```

### GASAConfig

The `GASAConfig` class provides configuration options for GASA:

```python
class GASAConfig(BaseModel):
    enabled: bool = True
    max_hops: int = 2  # Maximum number of hops for attention (h parameter)
    mask_strategy: MaskStrategy = MaskStrategy.BINARY
    fallback_strategy: FallbackStrategy = FallbackStrategy.BLOCK_DIAGONAL
    global_tokens: List[str] = ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"]
    summary_token: str = "[SUM]"
    add_summary_token: bool = True
    block_size: int = 512
    overlap: int = 64
    soft_mask_temperature: float = 0.1
    cache_masks: bool = True
    cache_dir: Optional[str] = None
    visualize: bool = False
    visualization_dir: Optional[str] = None
```

## How GASA Works

### 1. Building the Mask

The `build_mask` method in `MaskBuilder` is the main entry point for creating attention masks:

```python
def build_mask(
    self,
    documents: List[Document],
    prompt: str,
    format: MaskFormat = MaskFormat.DENSE,
    mask_type: MaskType = MaskType.ATTENTION,
) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
    # Build an attention mask based on the document dependency graph
```

The process involves:

1. **Tokenizing the prompt**: Converting the prompt text to tokens
2. **Mapping tokens to chunks**: Identifying which document chunks each token belongs to
3. **Building chunk adjacency**: Creating an adjacency matrix for chunks based on graph distance
4. **Expanding to token level**: Converting the chunk-level adjacency to a token-level mask
5. **Handling global tokens**: Ensuring special tokens like [CLS] can attend to all other tokens

### 2. Integration with the Executor

The `Executor` class integrates GASA by initializing a `MaskBuilder` and applying the mask during generation:

```python
# Initialize GASA if enabled
self.mask_builder = None
if self.config.enable_gasa and self.dependency_graph is not None:
    self.mask_builder = MaskBuilder(
        graph=self.dependency_graph,
        config=self.gasa_config,
        tokenizer=getattr(self.model, "tokenizer", None),
    )
```

When generating text, the Executor applies the GASA mask:

```python
# Apply GASA mask if enabled
if self.config.enable_gasa and self.mask_builder is not None and documents is not None:
    mask = self.mask_builder.build_mask(
        documents=documents,
        prompt=prompt,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )
    final_kwargs["attention_mask"] = mask
```

### 3. Fallback Strategies

For models that don't support sparse attention, GASA provides fallback strategies:

#### Block-Diagonal Packing

The `BlockDiagonalPacker` class reorders tokens so that related tokens are close to each other, allowing the model to use its limited attention window effectively:

```python
class BlockDiagonalPacker:
    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        # Initialize the block-diagonal packer
```

The packing process:
1. Maps tokens to chunks
2. Groups chunks by document and graph distance
3. Creates a reordering based on chunk groups
4. Applies the reordering to input IDs and attention mask

## Visualization

GASA includes a `MaskVisualizer` class for debugging and understanding attention masks:

```python
class MaskVisualizer:
    def visualize_mask(
        self,
        mask: Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        token_labels: Optional[List[str]] = None,
        highlight_tokens: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 10),
    ) -> Optional[Any]:
        # Visualize an attention mask
```

The visualizer can create heatmaps of attention masks, highlight specific tokens, and analyze mask sparsity.

## Example Usage

### Using GASA with Agent

The easiest way to use GASA is through the Agent class:

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph

# Create memory components
memory_store = MemoryStore()
dependency_graph = DependencyGraph()

# Index your repository or documents
memory_store.index_repository("/path/to/your/repo")
dependency_graph.build_from_memory(memory_store)

# Create agent configuration with GASA enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_gasa=True,  # Enable GASA
    gasa_max_hops=2,   # Set maximum hops for attention
)

# Create agent
agent = Agent(config=config)

# Set memory components
agent.memory_store = memory_store
agent.dependency_graph = dependency_graph

# Run a task (GASA will be automatically applied)
import asyncio
result = asyncio.run(agent.run("Analyze the architecture of this codebase"))
```

### Using GASA with Executor

You can also use GASA directly with the Executor:

```python
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig
from saplings.core.model_adapter import LLM
from saplings.memory import DependencyGraph

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create a dependency graph
graph = DependencyGraph()
# ... populate the graph ...

# Create GASA configuration
gasa_config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    cache_masks=True,
)

# Create executor configuration
executor_config = ExecutorConfig(
    enable_gasa=True,
    max_tokens=1024,
    temperature=0.7,
)

# Create executor with GASA
executor = Executor(
    model=model,
    config=executor_config,
    gasa_config=gasa_config,
    dependency_graph=graph,
)

# Execute with documents (GASA will be applied)
documents = [...]  # List of documents
result = await executor.execute(
    prompt="Summarize these documents:",
    documents=documents,
)
```

### Low-Level GASA Usage

```python
from saplings.gasa import GASAConfig, MaskBuilder, MaskFormat, MaskType
from saplings.memory import DependencyGraph, Document

# Create a dependency graph
graph = DependencyGraph()

# Add documents and relationships to the graph
# ...

# Create a GASA configuration
gasa_config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    cache_masks=True,
)

# Create a mask builder
mask_builder = MaskBuilder(
    graph=graph,
    config=gasa_config,
    tokenizer=tokenizer,  # Your tokenizer
)

# Build a mask
documents = [...]  # List of documents used in the prompt
prompt = "Summarize the following documents: ..."
mask = mask_builder.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)

# Use the mask with your model
outputs = model(
    input_ids=input_ids,
    attention_mask=mask,
    # Other model inputs...
)
```

### Visualization

```python
from saplings.gasa import MaskVisualizer

# Create a visualizer
visualizer = MaskVisualizer()

# Visualize the mask
visualizer.visualize_mask(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="mask_visualization.png",
    title="GASA Attention Mask",
    show=True,
)

# Analyze mask sparsity
visualizer.visualize_mask_sparsity(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="mask_sparsity.png",
    title="GASA Mask Sparsity",
    show=True,
)
```

## Advanced Configuration

### Soft Masks

Instead of binary masks, you can use soft masks with continuous values between 0 and 1:

```python
gasa_config = GASAConfig(
    mask_strategy=MaskStrategy.SOFT,
    soft_mask_temperature=0.1,  # Lower = closer to binary
)
```

### Custom Global Tokens

You can specify which tokens should have global attention:

```python
gasa_config = GASAConfig(
    global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
)
```

### Block Size and Overlap

For block-diagonal packing, you can configure the block size and overlap:

```python
gasa_config = GASAConfig(
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    block_size=1024,
    overlap=128,
)
```

## Performance Considerations

- **Computational Efficiency**: GASA can reduce the computational cost of attention by up to 40% by focusing attention on relevant tokens.
- **Memory Usage**: Sparse masks can reduce memory usage compared to dense attention.
- **Caching**: GASA supports caching masks to avoid redundant computation.
- **Tokenizer Compatibility**: GASA works best with tokenizers that provide token-to-text mapping.

## Limitations

- **Model Compatibility**: Not all models support sparse attention masks. Use fallback strategies for incompatible models.
- **Graph Quality**: The effectiveness of GASA depends on the quality of the dependency graph.
- **Hyperparameter Sensitivity**: Performance can be sensitive to the `max_hops` parameter.

## Future Directions

- **Learned Masks**: Support for learned attention masks that adapt to the task.
- **Multi-head Attention**: Different masks for different attention heads.
- **Dynamic Masks**: Masks that change during generation based on context.
- **Integration with Other Techniques**: Combining GASA with other attention optimization techniques.
