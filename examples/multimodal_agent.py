"""
This example demonstrates a multi-modal agent that uses Graph-Aligned Sparse Attention (GASA)
to efficiently process and reason about text, images, and their relationships.
"""

from __future__ import annotations

import asyncio
import os

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.gasa import GASAConfig
from saplings.memory import DependencyGraph, MemoryStore


# A simple image processor class
# In a real application, you'd use a more sophisticated implementation
class SimpleImageProcessor:
    """A simple image processor for demonstration purposes."""

    async def process_image(self, image_path):
        """Process an image file and extract basic information."""
        # In a real application, you'd use a vision model to extract features
        # Here we just return some dummy metadata

        image_size = os.path.getsize(image_path)
        extension = os.path.splitext(image_path)[1].lower()
        filename = os.path.basename(image_path)

        return {
            "text_description": f"An image file named {filename} with {extension} format and size {image_size} bytes.",
            "feature_vector": [0.1, 0.2, 0.3, 0.4, 0.5],  # Dummy embedding
            "metadata": {"size": image_size, "extension": extension, "filename": filename},
        }


async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Create image processor
    image_processor = SimpleImageProcessor()

    # Set up a sample dataset directory
    dataset_path = "./multimodal_dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created directory: {dataset_path}")
        print(
            "This is a demo. In a real application, you would add your own text and image files to this directory."
        )

        # Create a sample text file for demonstration
        with open(os.path.join(dataset_path, "sample_text.txt"), "w") as f:
            f.write(
                "This is a sample text document about artificial intelligence and multimodal learning."
            )
        print("Created a sample text file for demonstration.")

        # Note about images
        print(
            "Note: For this demo to work with real images, add some image files (.jpg, .png) to the multimodal_dataset directory."
        )

    # Check for text files
    text_files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
    print(f"Found {len(text_files)} text files in {dataset_path}")

    # Check for image files
    image_files = [
        f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    print(f"Found {len(image_files)} image files in {dataset_path}")

    # Process text documents
    print("Processing text documents...")
    for filename in text_files:
        file_path = os.path.join(dataset_path, filename)
        with open(file_path) as f:
            content = f.read()
            await memory.add_document(
                content=content, metadata={"type": "text", "source": filename}
            )
            print(f"Added text document: {filename}")

    # Process images
    print("Processing images...")
    for filename in image_files:
        image_path = os.path.join(dataset_path, filename)
        try:
            image_features = await image_processor.process_image(image_path)

            # Add image document with extracted features
            await memory.add_document(
                content=image_features["text_description"],
                metadata={
                    "type": "image",
                    "source": filename,
                    "features": image_features["feature_vector"],
                    "image_path": image_path,
                },
            )
            print(f"Added image document: {filename}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

    # Build a dependency graph that includes cross-modal relationships
    print("Building dependency graph...")
    await graph.build_from_memory(memory)

    # Add explicit relationships between documents if needed
    # In a real application, these might come from semantic similarity or other analysis
    documents = memory.get_documents()
    if len(documents) >= 2:
        # Create some example relationships between the first two documents
        graph.add_relationship(documents[0].id, documents[1].id, "relates_to", 0.8)
        print("Added example relationship between documents")

    # Configure GASA for multi-modal processing
    gasa_config = GASAConfig(
        max_hops=3,  # More hops to allow cross-modal connections
        mask_strategy="binary",
        add_summary_token=True,
    )

    # Create model
    model = LLM.create("openai", "gpt-4o")

    # Create a multi-modal agent
    print("Creating multi-modal agent...")
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            gasa_strategy=gasa_config.mask_strategy,
            gasa_max_hops=gasa_config.max_hops,
            supported_modalities=["text", "image"],
            memory_path="./multimodal_agent_memory",
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Example tasks
    tasks = [
        "Summarize all the documents you have access to, both text and images",
    ]

    if len(documents) >= 2:
        tasks.append(
            f"Describe the relationship between the document '{documents[0].id}' and '{documents[1].id}'"
        )

    tasks.append("What types of content are in your memory and how are they related?")

    # Run each task
    for task in tasks:
        print(f"\n--- Task: {task} ---\n")
        result = await agent.run(task, input_modalities=["text"], output_modalities=["text"])
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
