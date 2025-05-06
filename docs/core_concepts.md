# Core Concepts

Saplings is a graph-first, self-improving agent framework designed to build intelligent agents that can understand, reason about, and interact with complex information structures. This document explains the core concepts and architecture of the Saplings framework.

For a hands-on introduction to using Saplings, see the [Quick Start Guide](./quick_start.md).

## Agent Architecture

Saplings agents are composed of several key components that work together to provide a comprehensive framework for building intelligent agents:

### 1. Memory

The memory system in Saplings combines vector storage with graph-based memory to create a rich representation of information:

- **MemoryStore**: The central component that manages documents and their relationships
- **Document**: The basic unit of information, containing content and metadata
- **DependencyGraph**: Represents relationships between documents, entities, and concepts
- **VectorStore**: Enables efficient similarity search using embeddings
- **Indexer**: Extracts entities and relationships from documents

### 2. Retrieval

The retrieval system finds relevant information based on queries using a cascaded approach:

- **CascadeRetriever**: Orchestrates the entire retrieval pipeline
- **TFIDFRetriever**: Performs initial filtering using TF-IDF
- **EmbeddingRetriever**: Finds similar documents using embeddings
- **GraphExpander**: Expands results using the dependency graph
- **EntropyCalculator**: Determines when to stop retrieval based on information gain

### 3. Planning

The planning system breaks down complex tasks into manageable steps:

- **SequentialPlanner**: Creates and optimizes execution plans
- **PlanStep**: Represents a single step in a plan
- **BudgetStrategy**: Manages resource allocation and constraints

### 4. Execution

The execution system carries out individual steps using models and tools:

- **Executor**: Executes prompts with retrieved context
- **RefinementStrategy**: Improves outputs through iterative refinement
- **VerificationStrategy**: Verifies outputs against expectations

### 5. Validation

The validation system ensures outputs meet quality standards:

- **ValidatorService**: Validates outputs against requirements
- **JudgeAgent**: Evaluates output quality and provides feedback
- **ValidatorRegistry**: Manages validator plugins

### 6. Self-Healing

The self-healing system enables agents to learn from errors and improve over time:

- **PatchGenerator**: Generates patches for errors
- **SuccessPairCollector**: Collects successful error-fix pairs
- **AdapterManager**: Manages LoRA adapters for different error types
- **LoRaTrainer**: Fine-tunes models using Low-Rank Adaptation

### 7. Tools

The tools system provides functionality for agents to interact with the world:

- **Tool**: Base class for all tools
- **ToolRegistry**: Manages tool registration and discovery
- **Default Tools**: Built-in tools for common tasks
- **MCPClient**: Client for Machine Control Protocol servers

### 8. Tool Factory

The tool factory enables dynamic creation of tools:

- **ToolFactory**: Creates tools from specifications
- **ToolValidator**: Validates generated tool code
- **Sandbox**: Provides secure execution environment

### 9. Monitoring

The monitoring system tracks agent performance and behavior:

- **TraceManager**: Manages execution traces
- **BlameGraph**: Identifies bottlenecks and error sources
- **Visualization**: Provides visual representations of performance

### 10. Orchestration

The orchestration system coordinates multiple agents:

- **GraphRunner**: Coordinates agents in a graph structure
- **AgentNode**: Represents an agent in the orchestration graph
- **NegotiationStrategy**: Manages agent negotiation

### 11. Modality Support

The modality support system enables agents to work with different types of content:

- **ModalityService**: Manages different modalities
- **Modality Handlers**: Process specific content types (text, image, audio, video)

### 12. Graph-Aligned Sparse Attention (GASA)

GASA is a novel technique that improves efficiency and grounding:

- **MaskBuilder**: Generates attention masks from dependency graphs
- **BlockDiagonalPacker**: Reorders tokens for models without sparse attention
- **GASAPromptComposer**: Structures prompts based on graph relationships
- **ShadowModelTokenizer**: Uses a small model for tokenization with third-party LLMs

## Core Design Principles

Saplings is built on several key design principles:

### 1. Dependency Inversion

Saplings uses interfaces to decouple high-level modules from low-level implementations, following the Dependency Inversion Principle. This allows for flexible composition and easy replacement of components.

### 2. Composition over Inheritance

Saplings favors composition over inheritance, using delegation to specialized services rather than complex inheritance hierarchies. This makes the code more modular and easier to maintain.

### 3. Single Responsibility

Each component in Saplings has a clear, focused responsibility, following the Single Responsibility Principle. This makes the code easier to understand, test, and maintain.

### 4. Interface Segregation

Saplings defines clear, cohesive interfaces for each service, following the Interface Segregation Principle. This prevents clients from depending on methods they don't use.

### 5. Extensibility

Saplings is designed to be extensible, with plugin systems for models, memory stores, validators, indexers, and tools. This allows users to customize the framework to their needs.

### 6. Budget Awareness

Saplings is designed to be budget-aware, with planning and execution systems that respect resource constraints. This helps prevent runaway costs and ensures efficient resource usage.

## Lifecycle of a Request

When a user sends a request to a Saplings agent, it goes through the following stages:

1. **Planning**: The planner breaks down the task into steps
2. **Retrieval**: The retriever finds relevant information for each step
3. **Execution**: The executor carries out each step using the model and tools
4. **Validation**: The validator ensures outputs meet quality standards
5. **Self-Healing**: If errors occur, the self-healing system attempts to fix them
6. **Monitoring**: The monitoring system tracks performance and behavior

Throughout this process, the agent uses its memory to store and retrieve information, and may use tools to interact with the world. The GASA system improves efficiency and grounding by focusing attention on relevant context.

## Conclusion

Saplings provides a comprehensive framework for building intelligent agents that can understand, reason about, and interact with complex information structures. By combining vector storage with graph-based memory, cascaded retrieval, budget-aware planning, and self-healing capabilities, Saplings enables the creation of agents that are more efficient, grounded, and capable than traditional RAG systems.
