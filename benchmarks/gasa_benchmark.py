"""
Benchmark script for Graph-Aligned Sparse Attention (GASA).

This script measures the performance improvements provided by GASA,
including reduced FLOP count, memory usage, and execution time.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig, MaskBuilder, MaskFormat, MaskType
from saplings.memory import DependencyGraph, Document, DocumentMetadata, MemoryStore
from saplings.memory.config import MemoryConfig
from saplings.monitoring import MonitoringConfig, TraceManager, BlameGraph


class GASABenchmark:
    """Benchmark for Graph-Aligned Sparse Attention (GASA)."""
    
    def __init__(
        self,
        model_uri: str,
        output_dir: str = "./benchmark_results",
        num_runs: int = 5,
        document_sizes: List[int] = [1, 5, 10, 20],
        max_hops_values: List[int] = [1, 2, 3],
    ):
        """
        Initialize the benchmark.
        
        Args:
            model_uri: URI of the model to use
            output_dir: Directory to save results
            num_runs: Number of runs for each configuration
            document_sizes: List of document counts to test
            max_hops_values: List of max_hops values to test
        """
        self.model_uri = model_uri
        self.output_dir = output_dir
        self.num_runs = num_runs
        self.document_sizes = document_sizes
        self.max_hops_values = max_hops_values
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize monitoring
        self.monitoring_config = MonitoringConfig(
            visualization_output_dir=os.path.join(output_dir, "visualizations"),
        )
        self.trace_manager = TraceManager(config=self.monitoring_config)
        self.blame_graph = BlameGraph(trace_manager=self.trace_manager, config=self.monitoring_config)
    
    async def run_benchmark(self):
        """Run the benchmark."""
        print(f"Starting GASA benchmark with model: {self.model_uri}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Document sizes: {self.document_sizes}")
        print(f"Max hops values: {self.max_hops_values}")
        
        # Create model
        model = LLM.from_uri(self.model_uri)
        
        # Results dictionary
        results = {
            "model": self.model_uri,
            "timestamp": datetime.now().isoformat(),
            "num_runs": self.num_runs,
            "configurations": [],
        }
        
        # Run benchmark for each configuration
        for doc_size in self.document_sizes:
            # Create documents and graph
            documents, graph = self._create_test_documents(doc_size)
            
            # Test with GASA disabled (baseline)
            baseline_result = await self._run_configuration(
                model=model,
                documents=documents,
                graph=graph,
                enable_gasa=False,
                max_hops=None,
                name=f"Baseline (No GASA) - {doc_size} docs",
            )
            results["configurations"].append(baseline_result)
            
            # Test with different max_hops values
            for max_hops in self.max_hops_values:
                gasa_result = await self._run_configuration(
                    model=model,
                    documents=documents,
                    graph=graph,
                    enable_gasa=True,
                    max_hops=max_hops,
                    name=f"GASA (h={max_hops}) - {doc_size} docs",
                )
                results["configurations"].append(gasa_result)
        
        # Save results
        self._save_results(results)
        
        # Generate report
        self._generate_report(results)
        
        print(f"Benchmark completed. Results saved to {self.output_dir}")
    
    async def _run_configuration(
        self,
        model: LLM,
        documents: List[Document],
        graph: DependencyGraph,
        enable_gasa: bool,
        max_hops: Optional[int],
        name: str,
    ) -> Dict:
        """
        Run benchmark for a specific configuration.
        
        Args:
            model: LLM model
            documents: List of documents
            graph: Dependency graph
            enable_gasa: Whether to enable GASA
            max_hops: Max hops value for GASA
            name: Name of the configuration
            
        Returns:
            Dict: Results for this configuration
        """
        print(f"\nRunning configuration: {name}")
        
        # Create GASA config
        gasa_config = None
        if enable_gasa and max_hops is not None:
            gasa_config = GASAConfig(
                max_hops=max_hops,
                mask_strategy="binary",
                cache_masks=False,  # Disable caching for benchmark
            )
        
        # Create executor
        executor = Executor(
            model=model,
            config=ExecutorConfig(
                enable_gasa=enable_gasa,
            ),
            gasa_config=gasa_config,
            dependency_graph=graph if enable_gasa else None,
            trace_manager=self.trace_manager,
        )
        
        # Create prompt
        prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(documents):
            prompt += f"Document {i+1}: {doc.content}\n\n"
        
        # Run multiple times and measure
        latencies = []
        token_counts = []
        memory_usages = []
        
        for i in range(self.num_runs):
            print(f"  Run {i+1}/{self.num_runs}...")
            
            # Create trace
            trace = self.trace_manager.create_trace()
            
            # Start span
            span = self.trace_manager.start_span(
                name=f"benchmark_{name}_run_{i+1}",
                trace_id=trace.trace_id,
                attributes={
                    "component": "benchmark",
                    "configuration": name,
                    "enable_gasa": enable_gasa,
                    "max_hops": max_hops,
                    "doc_size": len(documents),
                },
            )
            
            # Measure memory before
            memory_before = self._get_memory_usage()
            
            # Execute
            start_time = time.time()
            result = await executor.execute(
                prompt=prompt,
                documents=documents,
                trace_id=trace.trace_id,
            )
            end_time = time.time()
            
            # Measure memory after
            memory_after = self._get_memory_usage()
            
            # Calculate metrics
            latency = (end_time - start_time) * 1000  # ms
            token_count = result.token_count if hasattr(result, "token_count") else 0
            memory_usage = memory_after - memory_before
            
            # Record metrics
            latencies.append(latency)
            token_counts.append(token_count)
            memory_usages.append(memory_usage)
            
            # End span
            self.trace_manager.end_span(span.span_id)
            
            # Process trace
            self.blame_graph.process_trace(trace)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        avg_token_count = np.mean(token_counts)
        avg_memory_usage = np.mean(memory_usages)
        
        # Get mask sparsity if GASA is enabled
        mask_sparsity = None
        if enable_gasa and max_hops is not None:
            # Create mask builder
            mask_builder = MaskBuilder(
                graph=graph,
                config=gasa_config,
                tokenizer=getattr(model, "tokenizer", None),
            )
            
            # Build mask
            mask = mask_builder.build_mask(
                documents=documents,
                prompt=prompt,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
            )
            
            # Calculate sparsity
            total_elements = mask.size
            nonzero_elements = np.count_nonzero(mask)
            zero_elements = total_elements - nonzero_elements
            mask_sparsity = zero_elements / total_elements
        
        # Create result dictionary
        result = {
            "name": name,
            "enable_gasa": enable_gasa,
            "max_hops": max_hops,
            "doc_size": len(documents),
            "avg_latency_ms": float(avg_latency),
            "std_latency_ms": float(std_latency),
            "avg_token_count": float(avg_token_count),
            "avg_memory_usage_mb": float(avg_memory_usage),
            "mask_sparsity": float(mask_sparsity) if mask_sparsity is not None else None,
            "raw_latencies_ms": [float(l) for l in latencies],
        }
        
        print(f"  Results: avg_latency={avg_latency:.2f}ms, mask_sparsity={mask_sparsity:.2%}")
        
        return result
    
    def _create_test_documents(self, num_documents: int) -> Tuple[List[Document], DependencyGraph]:
        """
        Create test documents and graph for benchmarking.
        
        Args:
            num_documents: Number of documents to create
            
        Returns:
            Tuple[List[Document], DependencyGraph]: Documents and graph
        """
        # Create documents
        documents = []
        for i in range(num_documents):
            doc = Document(
                content=f"This is test document {i+1} with some content for benchmarking. "
                        f"It contains information that relates to other documents in the set. "
                        f"Specifically, it references concepts from documents {max(1, i-1)} "
                        f"and {min(num_documents, i+2)}.",
                metadata=DocumentMetadata(
                    source=f"test_doc_{i+1}.txt",
                    document_id=f"doc_{i+1}",
                ),
            )
            documents.append(doc)
        
        # Create graph
        graph = DependencyGraph()
        
        # Add documents to graph
        nodes = []
        for doc in documents:
            node = graph.add_document_node(doc)
            nodes.append(node)
        
        # Add relationships
        for i in range(len(nodes)):
            # Connect to previous document
            if i > 0:
                graph.add_relationship(nodes[i].id, nodes[i-1].id, "references")
            
            # Connect to next document
            if i < len(nodes) - 1:
                graph.add_relationship(nodes[i].id, nodes[i+1].id, "references")
            
            # Add some long-range connections for larger document sets
            if num_documents > 5:
                # Connect every 3rd document
                if i % 3 == 0 and i + 3 < len(nodes):
                    graph.add_relationship(nodes[i].id, nodes[i+3].id, "references")
        
        return documents, graph
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _save_results(self, results: Dict):
        """
        Save benchmark results to disk.
        
        Args:
            results: Benchmark results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"gasa_benchmark_{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
    
    def _generate_report(self, results: Dict):
        """
        Generate a report from benchmark results.
        
        Args:
            results: Benchmark results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"gasa_benchmark_report_{timestamp}.md")
        
        with open(filename, "w") as f:
            f.write("# GASA Benchmark Report\n\n")
            f.write(f"- **Model**: {results['model']}\n")
            f.write(f"- **Timestamp**: {results['timestamp']}\n")
            f.write(f"- **Number of runs**: {results['num_runs']}\n\n")
            
            f.write("## Summary\n\n")
            
            # Group by document size
            doc_sizes = sorted(set(config["doc_size"] for config in results["configurations"]))
            
            for doc_size in doc_sizes:
                f.write(f"### Document Size: {doc_size}\n\n")
                
                # Get configurations for this document size
                configs = [c for c in results["configurations"] if c["doc_size"] == doc_size]
                
                # Get baseline
                baseline = next((c for c in configs if not c["enable_gasa"]), None)
                
                if baseline:
                    baseline_latency = baseline["avg_latency_ms"]
                    f.write(f"- **Baseline (No GASA)**: {baseline_latency:.2f} ms\n")
                    
                    # Calculate improvements for each GASA configuration
                    for config in configs:
                        if config["enable_gasa"]:
                            latency = config["avg_latency_ms"]
                            improvement = (baseline_latency - latency) / baseline_latency * 100
                            sparsity = config["mask_sparsity"] * 100 if config["mask_sparsity"] is not None else "N/A"
                            
                            f.write(f"- **GASA (h={config['max_hops']})**: {latency:.2f} ms ")
                            f.write(f"({improvement:.2f}% improvement, {sparsity:.2f}% sparsity)\n")
                
                f.write("\n")
            
            f.write("## Detailed Results\n\n")
            
            # Create a table
            f.write("| Configuration | Doc Size | Latency (ms) | Memory (MB) | Sparsity |\n")
            f.write("|--------------|----------|--------------|-------------|----------|\n")
            
            for config in results["configurations"]:
                name = config["name"]
                doc_size = config["doc_size"]
                latency = config["avg_latency_ms"]
                memory = config["avg_memory_usage_mb"]
                sparsity = f"{config['mask_sparsity']*100:.2f}%" if config["mask_sparsity"] is not None else "N/A"
                
                f.write(f"| {name} | {doc_size} | {latency:.2f} | {memory:.2f} | {sparsity} |\n")
            
            f.write("\n")
            
            # Add visualization instructions
            f.write("## Visualizations\n\n")
            f.write("To generate visualizations from this data:\n\n")
            f.write("```python\n")
            f.write("import matplotlib.pyplot as plt\n")
            f.write("import json\n\n")
            f.write("# Load the benchmark results\n")
            f.write(f"with open('{os.path.basename(filename).replace('_report', '')}', 'r') as f:\n")
            f.write("    results = json.load(f)\n\n")
            f.write("# Plot latency by document size\n")
            f.write("plt.figure(figsize=(10, 6))\n\n")
            f.write("doc_sizes = sorted(set(config['doc_size'] for config in results['configurations']))\n")
            f.write("for config_type in ['Baseline', 'GASA (h=1)', 'GASA (h=2)', 'GASA (h=3)']:\n")
            f.write("    latencies = []\n")
            f.write("    for doc_size in doc_sizes:\n")
            f.write("        config = next((c for c in results['configurations'] if c['name'].startswith(config_type) and c['doc_size'] == doc_size), None)\n")
            f.write("        if config:\n")
            f.write("            latencies.append(config['avg_latency_ms'])\n")
            f.write("        else:\n")
            f.write("            latencies.append(None)\n")
            f.write("    plt.plot(doc_sizes, latencies, marker='o', label=config_type)\n\n")
            f.write("plt.xlabel('Number of Documents')\n")
            f.write("plt.ylabel('Latency (ms)')\n")
            f.write("plt.title('GASA Performance by Document Size')\n")
            f.write("plt.legend()\n")
            f.write("plt.grid(True)\n")
            f.write("plt.savefig('gasa_latency_comparison.png')\n")
            f.write("plt.show()\n")
            f.write("```\n")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GASA Benchmark")
    parser.add_argument("--model", type=str, default="local://llama3?model_path=/path/to/model",
                        help="Model URI")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each configuration")
    parser.add_argument("--document-sizes", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="List of document counts to test")
    parser.add_argument("--max-hops", type=int, nargs="+", default=[1, 2, 3],
                        help="List of max_hops values to test")
    
    args = parser.parse_args()
    
    benchmark = GASABenchmark(
        model_uri=args.model,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        document_sizes=args.document_sizes,
        max_hops_values=args.max_hops,
    )
    
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
