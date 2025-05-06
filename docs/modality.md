# Modality Support

The Modality system in Saplings provides comprehensive support for different modalities (text, image, audio, video) that can be used by agents to process and generate content in different formats.

## Overview

The Modality system consists of several key components:

- **ModalityService**: Manages modality handlers and orchestrates multimodal operations
- **ModalityHandler**: Base class for all modality handlers
- **TextHandler**: Handles text content
- **ImageHandler**: Handles image content
- **AudioHandler**: Handles audio content
- **VideoHandler**: Handles video content

This system enables agents to work with different types of content, providing a unified interface for processing inputs and formatting outputs in various modalities.

## Core Concepts

### Modality Service

The `ModalityService` is the central component that manages modality handlers and provides a unified interface for multimodal operations:

1. **Handler Management**: Initializes and manages handlers for different modalities
2. **Input Processing**: Processes input content in different modalities
3. **Output Formatting**: Formats output content in different modalities
4. **Cross-Modal Conversion**: Converts content between different modalities

### Modality Handlers

Each modality has a dedicated handler that implements the `ModalityHandler` interface:

```python
class ModalityHandler:
    async def process_input(self, input_data: Any) -> Any:
        """Process input data for this modality."""

    async def format_output(self, output: Any) -> Any:
        """Format output data for this modality."""

    def to_message_content(self, data: Any) -> MessageContent:
        """Convert data to MessageContent."""

    @classmethod
    def from_message_content(cls, content: MessageContent) -> Any:
        """Convert MessageContent to data."""
```

The following handlers are available:

- **TextHandler**: Processes and formats text content
- **ImageHandler**: Processes and formats image content (URLs, file paths, or bytes)
- **AudioHandler**: Processes and formats audio content (URLs, file paths, or bytes)
- **VideoHandler**: Processes and formats video content (URLs, file paths, or bytes)

## API Reference

### ModalityService

```python
class ModalityService:
    def __init__(
        self,
        model: LLM,
        supported_modalities: Optional[List[str]] = None,
        trace_manager: Optional["TraceManager"] = None,
    ) -> None:
        """
        Initialize the modality service.

        Args:
            model: LLM model to use for processing
            supported_modalities: List of supported modalities (text, image, audio, video)
            trace_manager: Optional trace manager for monitoring
        """

    def get_handler(self, modality: str) -> Any:
        """
        Get handler for a specific modality.

        Args:
            modality: Modality name (text, image, audio, video)

        Returns:
            ModalityHandler for the specified modality
        """

    def supported_modalities(self) -> List[str]:
        """
        Get list of supported modalities.

        Returns:
            List of supported modality names
        """

    async def process_input(
        self,
        content: Any,
        input_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Process input content in the specified modality.

        Args:
            content: The content to process
            input_modality: The modality of the input content
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
            Processed content
        """

    async def format_output(
        self,
        content: str,
        output_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Format output content in the specified modality.

        Args:
            content: The content to format (usually text)
            output_modality: The desired output modality
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
            Formatted content in the specified modality
        """

    async def convert(
        self,
        content: Any,
        source_modality: str,
        target_modality: str,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Any:
        """
        Convert content between modalities.

        Args:
            content: The content to convert
            source_modality: The modality of the input content
            target_modality: The desired output modality
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
            Converted content in the target modality
        """
```

### TextHandler

```python
class TextHandler(ModalityHandler):
    async def process_input(self, input_data: Any) -> str:
        """
        Process text input.

        Args:
            input_data: Input text data

        Returns:
            Processed text
        """

    async def format_output(self, output: Any) -> str:
        """
        Format output as text.

        Args:
            output: Output data

        Returns:
            Formatted text
        """

    def to_message_content(self, data: str) -> MessageContent:
        """
        Convert text data to MessageContent.

        Args:
            data: Text data

        Returns:
            MessageContent object
        """

    @classmethod
    def from_message_content(cls, content: MessageContent) -> str:
        """
        Convert MessageContent to text data.

        Args:
            content: MessageContent to convert

        Returns:
            Text data
        """
```

### ImageHandler

```python
class ImageHandler(ModalityHandler):
    async def process_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Process image input.

        Args:
            input_data: Input image data (URL, file path, or bytes)

        Returns:
            Processed image data
        """

    async def format_output(self, output: Any) -> Dict[str, Any]:
        """
        Format output as image.

        Args:
            output: Output data

        Returns:
            Formatted image data
        """

    def to_message_content(self, data: Dict[str, Any]) -> MessageContent:
        """
        Convert image data to MessageContent.

        Args:
            data: Image data

        Returns:
            MessageContent object
        """

    @classmethod
    def from_message_content(cls, content: MessageContent) -> Dict[str, Any]:
        """
        Convert MessageContent to image data.

        Args:
            content: MessageContent to convert

        Returns:
            Image data
        """
```

Similar interfaces are available for `AudioHandler` and `VideoHandler`.

## Usage Examples

### Basic Usage

```python
from saplings import Agent, AgentConfig
from saplings.modality import ModalityService, TextHandler, ImageHandler

# Create an agent with multimodal support
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        supported_modalities=["text", "image"],
    )
)

# Process text input
result = await agent.run(
    "Describe this image",
    input_modalities=["text"],
    output_modalities=["text"]
)
print(result)

# Process image input
image_path = "path/to/image.jpg"
result = await agent.run(
    image_path,
    input_modalities=["image"],
    output_modalities=["text"]
)
print(result)
```

### Multimodal Agent

```python
import asyncio
import os
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add text documents
    text_files = [f for f in os.listdir("./dataset") if f.endswith(".txt")]
    for filename in text_files:
        with open(os.path.join("./dataset", filename), "r") as f:
            content = f.read()
            await memory.add_document(
                content=content,
                metadata={"type": "text", "source": filename}
            )

    # Add image documents
    image_files = [f for f in os.listdir("./dataset") if f.endswith((".jpg", ".png"))]
    for filename in image_files:
        image_path = os.path.join("./dataset", filename)
        await memory.add_document(
            content=f"Image file: {filename}",
            metadata={"type": "image", "source": filename, "image_path": image_path}
        )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create a multi-modal agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            supported_modalities=["text", "image"],
            enable_gasa=True,
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run a task that requires multimodal reasoning
    result = await agent.run(
        "Analyze all the documents in memory and describe the relationships between text and images.",
        input_modalities=["text"],
        output_modalities=["text"]
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Modality Handler

```python
from saplings.modality import ModalityHandler
from typing import Any, Dict

class PDFHandler(ModalityHandler):
    """Handler for PDF modality."""

    async def process_input(self, input_data: Any) -> Dict[str, Any]:
        """Process PDF input."""
        # Implementation for processing PDF files
        import fitz  # PyMuPDF

        if isinstance(input_data, str) and input_data.endswith(".pdf"):
            # Input is a file path
            doc = fitz.open(input_data)
            text = ""
            for page in doc:
                text += page.get_text()
            return {"text": text, "pages": len(doc), "source": input_data}

        # Handle other input types
        return {"text": str(input_data)}

    async def format_output(self, output: Any) -> Dict[str, Any]:
        """Format output as PDF."""
        # Implementation for formatting output as PDF
        return {"text": str(output)}

# Register the custom handler
from saplings.modality import get_handler_for_modality

# Monkey patch the get_handler_for_modality function to include PDF handler
original_get_handler = get_handler_for_modality

def custom_get_handler(modality: str, model: LLM) -> ModalityHandler:
    if modality == "pdf":
        return PDFHandler(model)
    return original_get_handler(modality, model)

# Replace the original function
import saplings.modality
saplings.modality.get_handler_for_modality = custom_get_handler
```

## Extension Points

The Modality system is designed to be extensible:

1. **Custom Handlers**: Create custom handlers for new modalities by implementing the `ModalityHandler` interface
2. **Custom Processing**: Override the `process_input` and `format_output` methods to implement custom processing logic
3. **Custom Conversion**: Implement cross-modal conversion by extending the `convert` method in `ModalityService`

## Best Practices

1. **Modality Selection**: Choose the appropriate modalities based on the task requirements
2. **Error Handling**: Implement robust error handling for modality-specific operations
3. **Timeout Management**: Set appropriate timeouts for modality operations
4. **Tracing**: Use the trace manager for monitoring and debugging modality operations
5. **Resource Management**: Properly manage resources for modality handlers, especially for image, audio, and video processing
