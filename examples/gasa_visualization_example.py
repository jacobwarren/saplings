"""
Example of using the GASA heatmap and TraceViewer visualization components.

This example demonstrates how to use the visualization components to analyze
GASA attention masks and trace execution.
"""

import asyncio
import numpy as np
import os
import tempfile
from datetime import datetime

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig, MaskBuilder, MaskFormat, MaskType
from saplings.memory import DependencyGraph, Document, DocumentMetadata
from saplings.monitoring import GASAHeatmap, TraceViewer, TraceManager, MonitoringConfig
from saplings.monitoring.config import VisualizationFormat


async def run_gasa_heatmap_example():
    """Run an example of the GASA heatmap visualization."""
    print("=== GASA Heatmap Visualization Example ===")
    
    # Create a dependency graph
    graph = DependencyGraph()
    
    # Create some documents
    doc1 = Document(
        content="This is the first document with some important information.",
        metadata=DocumentMetadata(
            source="example.txt",
            document_id="doc1",
        ),
    )
    
    doc2 = Document(
        content="This is the second document that relates to the first one.",
        metadata=DocumentMetadata(
            source="example2.txt",
            document_id="doc2",
        ),
    )
    
    doc3 = Document(
        content="This is a third document with unrelated information.",
        metadata=DocumentMetadata(
            source="example3.txt",
            document_id="doc3",
        ),
    )
    
    # Add documents to the graph
    node1 = graph.add_document_node(doc1)
    node2 = graph.add_document_node(doc2)
    node3 = graph.add_document_node(doc3)
    
    # Add relationships between documents
    graph.add_relationship(node1.id, node2.id, "references")
    
    # Create a GASA configuration
    gasa_config = GASAConfig(
        max_hops=2,
        mask_strategy="binary",
        cache_masks=True,
    )
    
    # Create a mock tokenizer
    class MockTokenizer:
        def __call__(self, text, return_tensors=None):
            # Simple tokenization by splitting on spaces
            tokens = text.split()
            input_ids = list(range(len(tokens)))
            return type('obj', (object,), {'input_ids': [input_ids]})
    
    tokenizer = MockTokenizer()
    
    # Create a mask builder
    mask_builder = MaskBuilder(
        graph=graph,
        config=gasa_config,
        tokenizer=tokenizer,
    )
    
    # Create a prompt that references the documents
    documents = [doc1, doc2, doc3]
    prompt = "Summarize the following documents: " + " ".join([doc.content for doc in documents])
    
    # Build a mask
    mask = mask_builder.build_mask(
        documents=documents,
        prompt=prompt,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )
    
    # Create a monitoring configuration
    monitoring_config = MonitoringConfig(
        visualization_output_dir="./visualizations",
        visualization_format=VisualizationFormat.HTML,
    )
    
    # Create a GASA heatmap visualizer
    heatmap = GASAHeatmap(
        config=monitoring_config,
        gasa_config=gasa_config,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(monitoring_config.visualization_output_dir, exist_ok=True)
    
    # Visualize the mask
    output_path = os.path.join(monitoring_config.visualization_output_dir, "gasa_mask.html")
    heatmap.visualize_mask(
        mask=mask,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
        output_path=output_path,
        title="GASA Attention Mask",
        show=False,
        interactive=True,
    )
    print(f"Saved GASA mask visualization to {output_path}")
    
    # Visualize mask sparsity
    output_path = os.path.join(monitoring_config.visualization_output_dir, "gasa_sparsity.html")
    heatmap.visualize_mask_sparsity(
        mask=mask,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
        output_path=output_path,
        title="GASA Mask Sparsity",
        show=False,
        interactive=True,
    )
    print(f"Saved GASA mask sparsity visualization to {output_path}")
    
    # Create a second mask with different parameters for comparison
    gasa_config2 = GASAConfig(
        max_hops=1,  # Reduced hops
        mask_strategy="binary",
        cache_masks=True,
    )
    
    mask_builder2 = MaskBuilder(
        graph=graph,
        config=gasa_config2,
        tokenizer=tokenizer,
    )
    
    mask2 = mask_builder2.build_mask(
        documents=documents,
        prompt=prompt,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )
    
    # Visualize mask comparison
    output_path = os.path.join(monitoring_config.visualization_output_dir, "gasa_comparison.html")
    heatmap.visualize_mask_comparison(
        masks=[
            (mask, MaskFormat.DENSE, MaskType.ATTENTION, "h=2"),
            (mask2, MaskFormat.DENSE, MaskType.ATTENTION, "h=1"),
        ],
        output_path=output_path,
        title="GASA Mask Comparison (h=2 vs h=1)",
        show=False,
        interactive=True,
    )
    print(f"Saved GASA mask comparison visualization to {output_path}")


async def run_trace_viewer_example():
    """Run an example of the TraceViewer visualization."""
    print("\n=== TraceViewer Visualization Example ===")
    
    # Create a monitoring configuration
    monitoring_config = MonitoringConfig(
        visualization_output_dir="./visualizations",
        visualization_format=VisualizationFormat.HTML,
    )
    
    # Create a trace manager
    trace_manager = TraceManager(config=monitoring_config)
    
    # Create a trace
    trace = trace_manager.create_trace(trace_id="example-trace")
    
    # Add spans to simulate a complex execution
    root_span = trace_manager.start_span(
        name="execute",
        trace_id=trace.trace_id,
        attributes={"component": "executor"},
    )
    
    # Add child spans
    retrieval_span = trace_manager.start_span(
        name="retrieve_documents",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "retriever"},
    )
    
    # Add event to retrieval span
    retrieval_span.add_event(
        name="documents_found",
        attributes={"count": 3},
    )
    
    # End retrieval span
    trace_manager.end_span(retrieval_span.span_id)
    
    # Add mask generation span
    mask_span = trace_manager.start_span(
        name="generate_mask",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "gasa"},
    )
    
    # Add child spans to mask generation
    graph_span = trace_manager.start_span(
        name="traverse_graph",
        trace_id=trace.trace_id,
        parent_id=mask_span.span_id,
        attributes={"component": "gasa"},
    )
    
    # End graph span
    trace_manager.end_span(graph_span.span_id)
    
    # End mask span
    trace_manager.end_span(mask_span.span_id)
    
    # Add model generation span
    model_span = trace_manager.start_span(
        name="generate_text",
        trace_id=trace.trace_id,
        parent_id=root_span.span_id,
        attributes={"component": "model"},
    )
    
    # End model span
    trace_manager.end_span(model_span.span_id)
    
    # End root span
    trace_manager.end_span(root_span.span_id)
    
    # Create a TraceViewer
    trace_viewer = TraceViewer(
        trace_manager=trace_manager,
        config=monitoring_config,
    )
    
    # Visualize the trace
    output_path = os.path.join(monitoring_config.visualization_output_dir, "trace.html")
    trace_viewer.view_trace(
        trace_id=trace.trace_id,
        output_path=output_path,
        show=False,
    )
    print(f"Saved trace visualization to {output_path}")
    
    # Visualize a specific span
    output_path = os.path.join(monitoring_config.visualization_output_dir, "mask_span.html")
    trace_viewer.view_span(
        trace_id=trace.trace_id,
        span_id=mask_span.span_id,
        output_path=output_path,
        show=False,
    )
    print(f"Saved span visualization to {output_path}")
    
    # Export the trace to JSON
    output_path = os.path.join(monitoring_config.visualization_output_dir, "trace.json")
    trace_viewer.export_trace(
        trace_id=trace.trace_id,
        output_path=output_path,
        format=VisualizationFormat.JSON,
    )
    print(f"Saved trace JSON to {output_path}")


async def main():
    """Run all examples."""
    await run_gasa_heatmap_example()
    await run_trace_viewer_example()


if __name__ == "__main__":
    asyncio.run(main())
