# Graph-Aligned Sparse Attention (GASA)

Graph-Aligned Sparse Attention (GASA) is a novel technique that improves efficiency and grounding in language models by focusing attention on relevant context based on document relationships.

## Overview

GASA injects a binary attention mask—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are within a defined number of hops in the graph, while routing others through lightweight global summary tokens.

This approach provides two key benefits:
1. **Reduced Computational Cost**: Up to 40% fewer FLOPs by eliminating unnecessary attention calculations
2. **Improved Grounding**: Better reasoning by focusing the model's attention on relevant context

## Core Concepts

### Attention Masks

GASA works by creating sparse attention masks that control which tokens can attend to each other:

- **Binary Masks**: Simple 0/1 masks where 1 indicates allowed attention
- **Soft Masks**: Continuous values between 0 and 1 for more nuanced attention
- **Learned Masks**: Masks that adapt based on fine-tuning

### Graph Distance

The key insight of GASA is using the dependency graph to determine which tokens should attend to each other:

- Tokens from documents that are closely related in the graph (within `max_hops`) have full attention
- Tokens from unrelated documents have limited or no direct attention
- Global tokens (like `[CLS]`, `[SEP]`, summary tokens) attend to all other tokens

### Fallback Strategies

For models that don't support custom attention masks (like third-party APIs), GASA provides several fallback strategies:

- **Block-Diagonal Packing**: Reorders tokens so related tokens are close to each other
- **Prompt Composer**: Structures prompts based on graph relationships
- **Shadow Model Tokenization**: Uses a small local model for tokenization and mask generation

## API Reference

### GASAConfig

```python
class GASAConfig(BaseModel):
    enabled: bool = True  # Whether to enable GASA
    max_hops: int = 2  # Maximum number of hops for attention (h parameter)
    mask_strategy: MaskStrategy = MaskStrategy.BINARY  # Strategy for applying attention masks
    fallback_strategy: FallbackStrategy = FallbackStrategy.BLOCK_DIAGONAL  # Fallback strategy
    global_tokens: List[str] = ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"]  # Global tokens
    summary_token: str = "[SUM]"  # Token used for global summary
    add_summary_token: bool = True  # Whether to add a summary token if not present
    block_size: int = 512  # Block size for block-diagonal packing
    overlap: int = 64  # Overlap between blocks for block-diagonal packing
    soft_mask_temperature: float = 0.1  # Temperature for soft masks
    cache_masks: bool = True  # Whether to cache generated masks
    cache_dir: Optional[str] = None  # Directory to cache masks
    visualize: bool = False  # Whether to generate visualizations
    visualization_dir: Optional[str] = None  # Directory to save visualizations

    # Shadow model configuration
    enable_shadow_model: bool = False  # Whether to enable shadow model for tokenization
    shadow_model_name: str = "Qwen/Qwen3-0.6B"  # Name of the shadow model to use
    shadow_model_device: str = "cpu"  # Device to use for the shadow model
    shadow_model_cache_dir: Optional[str] = None  # Directory to cache the shadow model

    # Prompt composer configuration
    enable_prompt_composer: bool = False  # Whether to enable the graph-aware prompt composer
    focus_tags: bool = True  # Whether to add focus tags to important context
    core_tag: str = "[CORE_CTX]"  # Tag for core context
    near_tag: str = "[NEAR_CTX]"  # Tag for near context
    summary_tag: str = "[SUMMARY_CTX]"  # Tag for summary context
```

### GASAService

```python
class GASAService:
    def __init__(
        self,
        graph: Optional[DependencyGraph] = None,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the GASA service."""

    def build_mask(
        self,
        documents: List[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
        """Build an attention mask based on the document dependency graph."""

    def reorder_tokens(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: List[int],
        attention_mask: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[List[int], Optional[Union[List[int], np.ndarray]], Dict[int, int]]:
        """Reorder tokens for block-diagonal packing."""

    def compose_prompt(
        self,
        documents: List[Document],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Compose a prompt based on graph relationships."""

    def apply_gasa(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: Optional[List[int]] = None,
        attention_mask: Optional[Union[List[int], np.ndarray]] = None,
        model_supports_sparse_attention: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply GASA to a prompt and input IDs."""

    def visualize_mask(
        self,
        mask: Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]],
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
    ) -> Optional[Any]:
        """Visualize an attention mask."""
```

### MaskBuilder

```python
class MaskBuilder:
    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the mask builder."""

    def build_mask(
        self,
        documents: List[Document],
        prompt: str,
        format: MaskFormat = MaskFormat.DENSE,
        mask_type: MaskType = MaskType.ATTENTION,
    ) -> Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]]:
        """Build an attention mask based on the document dependency graph."""
```

### BlockDiagonalPacker

```python
class BlockDiagonalPacker:
    def __init__(
        self,
        graph: Optional[DependencyGraph] = None,
        config: Optional[GASAConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize the block-diagonal packer."""

    def reorder_tokens(
        self,
        documents: List[Document],
        prompt: str,
        input_ids: List[int],
        attention_mask: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[List[int], Optional[Union[List[int], np.ndarray]], Dict[int, int]]:
        """Reorder tokens for block-diagonal packing."""
```

### GASAPromptComposer

```python
class GASAPromptComposer:
    def __init__(
        self,
        graph: DependencyGraph,
        config: Optional[GASAConfig] = None,
    ):
        """Initialize the prompt composer."""

    def compose_prompt(
        self,
        documents: List[Document],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Compose a prompt based on graph relationships."""
```

### ShadowModelTokenizer

```python
class ShadowModelTokenizer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        use_fast: bool = True,
        fallback_to_simple: bool = True,
        cpu_only: bool = False,
        alternative_models: Optional[List[str]] = None,
    ):
        """Initialize the shadow model tokenizer."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to token IDs."""

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to tokens."""

    def __call__(self, text: str, return_tensors: Optional[str] = None) -> object:
        """Tokenize text and return a compatible object."""
```

### MaskVisualizer

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
        """Visualize an attention mask."""

    def visualize_mask_sparsity(
        self,
        mask: Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]],
        format: MaskFormat,
        mask_type: MaskType,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (10, 5),
    ) -> Optional[Any]:
        """Visualize the sparsity of an attention mask."""

    def visualize_mask_comparison(
        self,
        masks: List[
            Tuple[Union[np.ndarray, sp.spmatrix, List[Dict[str, Any]]], MaskFormat, MaskType, str]
        ],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (15, 10),
    ) -> Optional[Any]:
        """Visualize a comparison of multiple attention masks."""
```

## Usage Examples

### Basic Usage with Agent

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph

# Create memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
for i in range(5):
    memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )

# Build dependency graph
graph.build_from_memory(memory)

# Add custom relationships
graph.add_relationship("doc_0", "doc_1", "relates_to", 0.9)
graph.add_relationship("doc_0", "doc_2", "relates_to", 0.8)

# Create agent with GASA enabled
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        gasa_max_hops=2,
        gasa_strategy="binary",
        gasa_fallback="prompt_composer",
        gasa_shadow_model=True,
        gasa_shadow_model_name="Qwen/Qwen3-0.6B",
        gasa_prompt_composer=True,
    )
)

# Set memory components
agent.memory_store = memory
agent.dependency_graph = graph

# Run a query
import asyncio
result = asyncio.run(agent.run("Explain the relationship between documents 0, 1, and 2."))
print(result)
```

### Using GASAService Directly

```python
from saplings.memory import MemoryStore, DependencyGraph, Document
from saplings.gasa import GASAService, GASAConfig, MaskFormat, MaskType
from saplings.core.model_adapter import LLM

# Create memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
documents = []
for i in range(5):
    doc = memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )
    documents.append(doc)

# Build dependency graph
graph.build_from_memory(memory)

# Create GASA configuration
config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    fallback_strategy="block_diagonal",
    visualize=True,
    visualization_dir="./gasa_visualizations",
)

# Create GASA service
gasa_service = GASAService(
    graph=graph,
    config=config,
)

# Create a prompt
prompt = "Summarize the following documents:\n\n"
for i, doc in enumerate(documents):
    prompt += f"Document {i}: {doc.content}\n\n"
prompt += "Summary:"

# Build a mask
mask = gasa_service.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)

# Visualize the mask
gasa_service.visualize_mask(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="./gasa_mask.png",
    title="GASA Attention Mask",
)

# Create a model
model = LLM.create("vllm", "meta-llama/Llama-3.1-8B-Instruct")

# Apply GASA to the prompt
result = gasa_service.apply_gasa(
    documents=documents,
    prompt=prompt,
    model_supports_sparse_attention=True,
)

# Generate text with the modified prompt and attention mask
response = model.generate(
    prompt=result["prompt"],
    attention_mask=result.get("attention_mask"),
)

print(response.text)
```

### Using GASA with Third-Party LLMs

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig

# Create memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
for i in range(5):
    memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )

# Build dependency graph
graph.build_from_memory(memory)

# Create agent with GASA enabled for OpenAI
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        gasa_max_hops=2,
        gasa_strategy="binary",
        gasa_fallback="prompt_composer",  # Use prompt composer for OpenAI
        gasa_shadow_model=True,  # Enable shadow model for tokenization
        gasa_shadow_model_name="Qwen/Qwen3-0.6B",
        gasa_prompt_composer=True,  # Enable prompt composer
    )
)

# Set memory components
agent.memory_store = memory
agent.dependency_graph = graph

# Run a query
import asyncio
result = asyncio.run(agent.run("Summarize all the documents."))
print(result)
```

### Visualizing GASA Masks

```python
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig, MaskBuilder, MaskVisualizer, MaskFormat, MaskType
import matplotlib.pyplot as plt

# Create memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
for i in range(5):
    memory.add_document(
        content=f"Document {i} about machine learning and artificial intelligence.",
        metadata={"source": f"doc_{i}.txt"}
    )

# Build dependency graph
graph.build_from_memory(memory)

# Create GASA configuration
config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    visualize=True,
    visualization_dir="./gasa_visualizations",
)

# Create mask builder
mask_builder = MaskBuilder(
    graph=graph,
    config=config,
)

# Create prompt
prompt = "Summarize the following documents:\n\n"
for i, doc in enumerate(memory.get_documents()):
    prompt += f"Document {i}: {doc.content}\n\n"
prompt += "Summary:"

# Build mask
mask = mask_builder.build_mask(
    documents=memory.get_documents(),
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)

# Create visualizer
visualizer = MaskVisualizer()

# Visualize mask
fig = visualizer.visualize_mask(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    title="GASA Attention Mask",
    show=True,
)

# Save visualization
plt.savefig("gasa_mask.png")

# Visualize mask sparsity
fig = visualizer.visualize_mask_sparsity(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    title="GASA Mask Sparsity",
    show=True,
)

# Save visualization
plt.savefig("gasa_sparsity.png")
```

## GASA with Third-Party LLMs

When using GASA with third-party LLM APIs like OpenAI and Anthropic, which don't expose low-level attention mechanisms, Saplings provides several alternative approaches:

### 1. Shadow Model Tokenization

The shadow model approach uses a small local model for tokenization and mask generation:

```python
from saplings.tokenizers import ShadowModelTokenizer

# Create a shadow model tokenizer
tokenizer = ShadowModelTokenizer(
    model_name="Qwen/Qwen3-0.6B",
    device="cpu",
    fallback_to_simple=True,
)

# Tokenize text
tokens = tokenizer.tokenize("Hello, world!")
print(tokens)

# Convert to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
```

The shadow model tokenizer:
- Loads a small local model (default: Qwen/Qwen3-0.6B)
- Uses it for tokenization only (not for generation)
- Falls back to a simple tokenizer if the model can't be loaded
- Supports alternative models for environments without Triton (like Apple Silicon)

### 2. Graph-Aware Prompt Composition

The prompt composer structures prompts based on graph relationships:

```python
from saplings.gasa import GASAPromptComposer, GASAConfig
from saplings.memory import DependencyGraph

# Create a dependency graph
graph = DependencyGraph()
# ... add nodes and relationships ...

# Create a prompt composer
composer = GASAPromptComposer(
    graph=graph,
    config=GASAConfig(
        enable_prompt_composer=True,
        focus_tags=True,
        core_tag="[CORE_CTX]",
        near_tag="[NEAR_CTX]",
        summary_tag="[SUMMARY_CTX]",
    ),
)

# Compose a prompt
composed_prompt = composer.compose_prompt(
    documents=documents,
    prompt="Summarize the following documents:",
    system_prompt="You are a helpful assistant.",
)

print(composed_prompt)
```

The prompt composer:
- Reorders document chunks based on graph relationships
- Adds focus tags to important context
- Structures the prompt to guide the model's attention

### 3. Block-Diagonal Packing

The block-diagonal packer reorders tokens to create a block-diagonal structure:

```python
from saplings.gasa import BlockDiagonalPacker, GASAConfig
from saplings.memory import DependencyGraph

# Create a dependency graph
graph = DependencyGraph()
# ... add nodes and relationships ...

# Create a block-diagonal packer
packer = BlockDiagonalPacker(
    graph=graph,
    config=GASAConfig(
        block_size=512,
        overlap=64,
    ),
)

# Reorder tokens
reordered_input_ids, reordered_attention_mask, position_mapping = packer.reorder_tokens(
    documents=documents,
    prompt="Summarize the following documents:",
    input_ids=input_ids,
    attention_mask=attention_mask,
)
```

The block-diagonal packer:
- Groups tokens based on document relationships
- Reorders tokens so related tokens are close to each other
- Allows the model to use its limited attention window effectively

## Implementation Details

### Mask Building Process

The mask building process works as follows:

1. **Token Mapping**: Map tokens to document chunks
2. **Chunk Adjacency**: Build an adjacency matrix for chunks based on graph distance
3. **Token-Level Expansion**: Expand the chunk adjacency matrix to token level
4. **Global Token Handling**: Ensure global tokens attend to all other tokens
5. **Format Conversion**: Convert the mask to the requested format

### Graph Distance Calculation

Graph distance is calculated using the dependency graph:

1. **Node Identification**: Identify the nodes corresponding to document chunks
2. **Shortest Path**: Find the shortest path between nodes in the graph
3. **Distance Thresholding**: Allow attention if the distance is within `max_hops`

### Fallback Strategy Selection

The fallback strategy is selected based on the model and configuration:

1. **Direct Mask Injection**: For models that support sparse attention (e.g., vLLM)
2. **Block-Diagonal Packing**: For models that don't support sparse attention but accept reordered tokens
3. **Prompt Composition**: For third-party APIs where we can only modify the prompt

## Advanced Features

### Mask Visualization

GASA provides visualization tools for attention masks:

```python
from saplings.gasa import MaskVisualizer, MaskFormat, MaskType
import matplotlib.pyplot as plt

# Create a visualizer
visualizer = MaskVisualizer()

# Visualize a mask
fig = visualizer.visualize_mask(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    title="GASA Attention Mask",
    show=True,
)

# Save the visualization
plt.savefig("gasa_mask.png")
```

### Mask Caching

GASA can cache generated masks to improve performance:

```python
from saplings.gasa import GASAConfig, GASAService

# Create a configuration with caching enabled
config = GASAConfig(
    cache_masks=True,
    cache_dir="./gasa_cache",
)

# Create a GASA service
service = GASAService(
    graph=graph,
    config=config,
)

# Build a mask (will be cached)
mask = service.build_mask(
    documents=documents,
    prompt=prompt,
)

# Build the same mask again (will use cache)
mask = service.build_mask(
    documents=documents,
    prompt=prompt,
)
```

### Custom Mask Strategies

GASA supports different mask strategies:

```python
from saplings.gasa import GASAConfig, MaskStrategy

# Binary mask (0/1)
binary_config = GASAConfig(
    mask_strategy=MaskStrategy.BINARY,
)

# Soft mask (continuous values)
soft_config = GASAConfig(
    mask_strategy=MaskStrategy.SOFT,
    soft_mask_temperature=0.1,  # Lower = closer to binary
)

# Learned mask (requires fine-tuning)
learned_config = GASAConfig(
    mask_strategy=MaskStrategy.LEARNED,
)
```

## Conclusion

Graph-Aligned Sparse Attention (GASA) is a powerful technique that improves efficiency and grounding in language models. By focusing attention on relevant context based on document relationships, GASA reduces computational cost and improves reasoning quality.

Saplings provides a comprehensive implementation of GASA with support for various models and fallback strategies for third-party LLM APIs. Whether you're using a local model with vLLM or a third-party API like OpenAI or Anthropic, you can benefit from GASA's improvements.
