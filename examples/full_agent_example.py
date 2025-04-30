"""
Full Agent Example with Saplings.

This example demonstrates a complete agent implementation using Saplings,
including GASA, monitoring, and self-improvement capabilities.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig
from saplings.memory import DependencyGraph, Document, DocumentMetadata, MemoryStore
from saplings.memory.config import MemoryConfig
from saplings.memory.indexers import VectorIndexer, GraphIndexer
from saplings.monitoring import MonitoringConfig, TraceManager, BlameGraph, TraceViewer
from saplings.tool_factory import ToolFactory, ToolFactoryConfig
from saplings.validator import Validator, ValidatorConfig
from saplings.judge import Judge, JudgeConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Research agent that can analyze documents and generate insights.
    
    This agent demonstrates the full capabilities of Saplings, including:
    - Document retrieval and analysis
    - Graph-Aligned Sparse Attention (GASA)
    - Dynamic tool creation
    - Self-improvement through validation and judging
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        model_uri: str,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the research agent.
        
        Args:
            model_uri: URI of the model to use
            memory_path: Path to store agent memory
            output_dir: Directory to save outputs
            enable_gasa: Whether to enable GASA
            enable_monitoring: Whether to enable monitoring
        """
        self.model_uri = model_uri
        self.memory_path = memory_path
        self.output_dir = output_dir
        self.enable_gasa = enable_gasa
        self.enable_monitoring = enable_monitoring
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self._init_memory()
        self._init_monitoring()
        self._init_model()
        self._init_executor()
        self._init_tool_factory()
        self._init_validator()
        self._init_judge()
        
        logger.info(f"Research agent initialized with model: {model_uri}")
        logger.info(f"GASA enabled: {enable_gasa}")
        logger.info(f"Monitoring enabled: {enable_monitoring}")
    
    def _init_memory(self):
        """Initialize memory components."""
        # Create memory config
        memory_config = MemoryConfig(
            store_path=self.memory_path,
        )
        
        # Create memory store
        self.memory_store = MemoryStore(config=memory_config)
        
        # Create dependency graph
        self.graph = DependencyGraph()
        
        # Create indexers
        self.vector_indexer = VectorIndexer(
            memory_store=self.memory_store,
            config=memory_config,
        )
        
        self.graph_indexer = GraphIndexer(
            memory_store=self.memory_store,
            graph=self.graph,
            config=memory_config,
        )
        
        logger.info("Memory components initialized")
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        if self.enable_monitoring:
            # Create monitoring config
            self.monitoring_config = MonitoringConfig(
                visualization_output_dir=os.path.join(self.output_dir, "visualizations"),
            )
            
            # Create trace manager
            self.trace_manager = TraceManager(config=self.monitoring_config)
            
            # Create blame graph
            self.blame_graph = BlameGraph(
                trace_manager=self.trace_manager,
                config=self.monitoring_config,
            )
            
            # Create trace viewer
            self.trace_viewer = TraceViewer(
                trace_manager=self.trace_manager,
                config=self.monitoring_config,
            )
            
            logger.info("Monitoring components initialized")
        else:
            self.trace_manager = None
            logger.info("Monitoring disabled")
    
    def _init_model(self):
        """Initialize the model."""
        self.model = LLM.from_uri(self.model_uri)
        logger.info(f"Model initialized: {self.model_uri}")
    
    def _init_executor(self):
        """Initialize the executor."""
        # Create GASA config if enabled
        gasa_config = None
        if self.enable_gasa:
            gasa_config = GASAConfig(
                max_hops=2,
                mask_strategy="binary",
                cache_masks=True,
            )
        
        # Create executor config
        executor_config = ExecutorConfig(
            enable_gasa=self.enable_gasa,
            max_tokens=1024,
            temperature=0.7,
        )
        
        # Create executor
        self.executor = Executor(
            model=self.model,
            config=executor_config,
            gasa_config=gasa_config,
            dependency_graph=self.graph if self.enable_gasa else None,
            trace_manager=self.trace_manager,
        )
        
        logger.info("Executor initialized")
    
    def _init_tool_factory(self):
        """Initialize the tool factory."""
        # Create tool factory config
        tool_factory_config = ToolFactoryConfig(
            sandbox_enabled=True,
            allowed_imports=["numpy", "pandas", "matplotlib", "plotly"],
        )
        
        # Create tool factory
        self.tool_factory = ToolFactory(
            executor=self.executor,
            config=tool_factory_config,
        )
        
        logger.info("Tool factory initialized")
    
    def _init_validator(self):
        """Initialize the validator."""
        # Create validator config
        validator_config = ValidatorConfig(
            validation_model_uri=self.model_uri,  # Use same model for validation
        )
        
        # Create validator
        self.validator = Validator(
            config=validator_config,
            executor=self.executor,
        )
        
        logger.info("Validator initialized")
    
    def _init_judge(self):
        """Initialize the judge."""
        # Create judge config
        judge_config = JudgeConfig(
            judge_model_uri=self.model_uri,  # Use same model for judging
        )
        
        # Create judge
        self.judge = Judge(
            config=judge_config,
            executor=self.executor,
        )
        
        logger.info("Judge initialized")
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> Document:
        """
        Add a document to the agent's memory.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document: The added document
        """
        # Create document
        doc_metadata = DocumentMetadata(
            source=metadata.get("source", "unknown"),
            document_id=metadata.get("document_id", f"doc_{datetime.now().timestamp()}"),
            **metadata,
        )
        
        document = Document(
            content=content,
            metadata=doc_metadata,
        )
        
        # Add to memory store
        self.memory_store.add_document(document)
        
        # Add to vector index
        self.vector_indexer.index_document(document)
        
        # Add to graph
        node = self.graph_indexer.index_document(document)
        
        logger.info(f"Added document: {document.metadata.document_id}")
        
        return document
    
    async def add_documents_from_directory(self, directory: str, extension: str = ".txt") -> List[Document]:
        """
        Add documents from a directory.
        
        Args:
            directory: Directory containing documents
            extension: File extension to filter by
            
        Returns:
            List[Document]: Added documents
        """
        documents = []
        
        # Check if directory exists
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return documents
        
        # Process each file
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                file_path = os.path.join(directory, filename)
                
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create metadata
                metadata = {
                    "source": file_path,
                    "document_id": filename,
                    "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                    "file_size": os.path.getsize(file_path),
                }
                
                # Add document
                document = await self.add_document(content, metadata)
                documents.append(document)
        
        logger.info(f"Added {len(documents)} documents from {directory}")
        
        return documents
    
    async def analyze_documents(self, query: str) -> Dict[str, Any]:
        """
        Analyze documents based on a query.
        
        Args:
            query: Query to analyze documents
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Analyzing documents with query: {query}")
        
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id
            
            root_span = self.trace_manager.start_span(
                name="analyze_documents",
                trace_id=trace_id,
                attributes={
                    "component": "research_agent",
                    "query": query,
                },
            )
        
        try:
            # Retrieve relevant documents
            if self.trace_manager:
                retrieval_span = self.trace_manager.start_span(
                    name="retrieve_documents",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "retriever"},
                )
            
            documents = self.vector_indexer.search(query, limit=5)
            
            if self.trace_manager and retrieval_span:
                retrieval_span.add_event(
                    name="documents_retrieved",
                    attributes={"count": len(documents)},
                )
                self.trace_manager.end_span(retrieval_span.span_id)
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following documents based on this query: {query}
            
            Documents:
            """
            
            for i, doc in enumerate(documents):
                prompt += f"\n--- Document {i+1}: {doc.metadata.document_id} ---\n"
                prompt += doc.content + "\n"
            
            prompt += """
            Provide a comprehensive analysis that includes:
            1. Key insights from the documents
            2. Connections between different documents
            3. Answers to the specific query
            4. Suggestions for further research
            
            Format your response as JSON with the following structure:
            {
                "insights": [list of key insights],
                "connections": [list of connections between documents],
                "answer": "direct answer to the query",
                "suggestions": [list of suggestions for further research]
            }
            """
            
            # Execute analysis
            if self.trace_manager:
                analysis_span = self.trace_manager.start_span(
                    name="generate_analysis",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "executor"},
                )
            
            result = await self.executor.execute(
                prompt=prompt,
                documents=documents,
                trace_id=trace_id,
            )
            
            if self.trace_manager and analysis_span:
                self.trace_manager.end_span(analysis_span.span_id)
            
            # Parse JSON response
            try:
                analysis = json.loads(result.text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                analysis = {"raw_text": result.text}
            
            # Validate analysis
            if self.trace_manager:
                validation_span = self.trace_manager.start_span(
                    name="validate_analysis",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "validator"},
                )
            
            validation_result = await self.validator.validate(
                input_data={"query": query, "documents": [doc.content for doc in documents]},
                output_data=analysis,
                validation_type="document_analysis",
            )
            
            if self.trace_manager and validation_span:
                validation_span.add_event(
                    name="validation_complete",
                    attributes={"is_valid": validation_result.is_valid},
                )
                self.trace_manager.end_span(validation_span.span_id)
            
            # Add validation result to analysis
            analysis["validation"] = {
                "is_valid": validation_result.is_valid,
                "score": validation_result.score,
                "feedback": validation_result.feedback,
            }
            
            # End root span
            if self.trace_manager and root_span:
                self.trace_manager.end_span(root_span.span_id)
            
            # Process trace for blame graph
            if self.trace_manager and self.blame_graph:
                trace = self.trace_manager.get_trace(trace_id)
                if trace:
                    self.blame_graph.process_trace(trace)
            
            # Save analysis to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"analysis_{timestamp}.json")
            
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Analysis completed and saved to {output_path}")
            
            # Save trace visualization if monitoring is enabled
            if self.trace_manager and self.trace_viewer and trace_id:
                viz_path = os.path.join(
                    self.monitoring_config.visualization_output_dir,
                    f"trace_{timestamp}.html",
                )
                
                self.trace_viewer.view_trace(
                    trace_id=trace_id,
                    output_path=viz_path,
                    show=False,
                )
                
                logger.info(f"Trace visualization saved to {viz_path}")
            
            return {
                "analysis": analysis,
                "trace_id": trace_id,
                "output_path": output_path,
            }
        
        except Exception as e:
            logger.error(f"Error during document analysis: {e}")
            
            # End root span with error
            if self.trace_manager and root_span:
                root_span.set_status("ERROR")
                self.trace_manager.end_span(root_span.span_id)
            
            return {
                "error": str(e),
                "trace_id": trace_id,
            }
    
    async def generate_visualization(self, analysis: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a visualization for the analysis.
        
        Args:
            analysis: Analysis data
            output_path: Path to save the visualization
            
        Returns:
            str: Path to the generated visualization
        """
        logger.info("Generating visualization for analysis")
        
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id
            
            root_span = self.trace_manager.start_span(
                name="generate_visualization",
                trace_id=trace_id,
                attributes={"component": "research_agent"},
            )
        
        try:
            # Create default output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"visualization_{timestamp}.html")
            
            # Create visualization prompt
            prompt = f"""
            Create a visualization for the following analysis:
            
            {json.dumps(analysis, indent=2)}
            
            Generate a Python script that creates an interactive visualization using Plotly.
            The visualization should show the key insights, connections between documents, and other relevant information.
            
            The script should:
            1. Import necessary libraries (plotly, etc.)
            2. Parse the analysis data
            3. Create an appropriate visualization (network graph, bar chart, etc.)
            4. Save the visualization to an HTML file at: {output_path}
            
            Return only the Python code without any explanations.
            """
            
            # Generate visualization code
            if self.trace_manager:
                code_span = self.trace_manager.start_span(
                    name="generate_code",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "executor"},
                )
            
            result = await self.executor.execute(
                prompt=prompt,
                trace_id=trace_id,
            )
            
            if self.trace_manager and code_span:
                self.trace_manager.end_span(code_span.span_id)
            
            # Extract code from response
            code = result.text.strip()
            
            # Create visualization tool
            if self.trace_manager:
                tool_span = self.trace_manager.start_span(
                    name="create_visualization_tool",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "tool_factory"},
                )
            
            visualization_tool = await self.tool_factory.create_tool(
                name="visualization_generator",
                description="Generates an interactive visualization for analysis data",
                code=code,
            )
            
            if self.trace_manager and tool_span:
                self.trace_manager.end_span(tool_span.span_id)
            
            # Execute visualization tool
            if self.trace_manager:
                exec_span = self.trace_manager.start_span(
                    name="execute_visualization_tool",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "tool_factory"},
                )
            
            # Prepare tool input
            tool_input = {
                "analysis": analysis,
                "output_path": output_path,
            }
            
            # Execute tool
            tool_result = await visualization_tool.execute(tool_input)
            
            if self.trace_manager and exec_span:
                self.trace_manager.end_span(exec_span.span_id)
            
            # End root span
            if self.trace_manager and root_span:
                self.trace_manager.end_span(root_span.span_id)
            
            logger.info(f"Visualization generated and saved to {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error during visualization generation: {e}")
            
            # End root span with error
            if self.trace_manager and root_span:
                root_span.set_status("ERROR")
                self.trace_manager.end_span(root_span.span_id)
            
            return str(e)
    
    async def improve_agent(self) -> Dict[str, Any]:
        """
        Improve the agent based on past performance.
        
        Returns:
            Dict[str, Any]: Improvement results
        """
        logger.info("Starting agent self-improvement process")
        
        # Start trace if monitoring is enabled
        trace_id = None
        if self.trace_manager:
            trace = self.trace_manager.create_trace()
            trace_id = trace.trace_id
            
            root_span = self.trace_manager.start_span(
                name="improve_agent",
                trace_id=trace_id,
                attributes={"component": "research_agent"},
            )
        
        try:
            # Get performance data from blame graph
            if self.trace_manager and self.blame_graph:
                if self.trace_manager:
                    perf_span = self.trace_manager.start_span(
                        name="analyze_performance",
                        trace_id=trace_id,
                        parent_id=root_span.span_id if root_span else None,
                        attributes={"component": "blame_graph"},
                    )
                
                # Identify bottlenecks
                bottlenecks = self.blame_graph.identify_bottlenecks(
                    threshold_ms=100.0,
                    min_call_count=1,
                )
                
                # Identify error sources
                error_sources = self.blame_graph.identify_error_sources(
                    min_error_rate=0.1,
                    min_call_count=1,
                )
                
                if self.trace_manager and perf_span:
                    perf_span.add_event(
                        name="performance_analysis_complete",
                        attributes={
                            "bottleneck_count": len(bottlenecks),
                            "error_source_count": len(error_sources),
                        },
                    )
                    self.trace_manager.end_span(perf_span.span_id)
                
                performance_data = {
                    "bottlenecks": bottlenecks,
                    "error_sources": error_sources,
                }
            else:
                performance_data = {
                    "bottlenecks": [],
                    "error_sources": [],
                }
            
            # Get validation data
            validation_data = self.validator.get_validation_history()
            
            # Create improvement prompt
            prompt = f"""
            Analyze the agent's performance and suggest improvements.
            
            Performance Data:
            {json.dumps(performance_data, indent=2)}
            
            Validation History:
            {json.dumps(validation_data, indent=2)}
            
            Based on this data, suggest specific improvements to the agent's:
            1. Prompting strategies
            2. Document analysis approach
            3. Visualization generation
            4. Overall workflow
            
            Format your response as JSON with the following structure:
            {{
                "identified_issues": [list of issues identified],
                "improvement_suggestions": [list of specific improvement suggestions],
                "prompt_templates": {{
                    "analysis_prompt": "improved analysis prompt template",
                    "visualization_prompt": "improved visualization prompt template"
                }},
                "implementation_plan": [list of steps to implement improvements]
            }}
            """
            
            # Execute improvement analysis
            if self.trace_manager:
                improve_span = self.trace_manager.start_span(
                    name="generate_improvements",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "executor"},
                )
            
            result = await self.executor.execute(
                prompt=prompt,
                trace_id=trace_id,
            )
            
            if self.trace_manager and improve_span:
                self.trace_manager.end_span(improve_span.span_id)
            
            # Parse JSON response
            try:
                improvements = json.loads(result.text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                improvements = {"raw_text": result.text}
            
            # Judge improvements
            if self.trace_manager:
                judge_span = self.trace_manager.start_span(
                    name="judge_improvements",
                    trace_id=trace_id,
                    parent_id=root_span.span_id if root_span else None,
                    attributes={"component": "judge"},
                )
            
            judgment = await self.judge.judge(
                input_data={
                    "performance_data": performance_data,
                    "validation_data": validation_data,
                },
                output_data=improvements,
                judgment_type="agent_improvement",
            )
            
            if self.trace_manager and judge_span:
                judge_span.add_event(
                    name="judgment_complete",
                    attributes={"score": judgment.score},
                )
                self.trace_manager.end_span(judge_span.span_id)
            
            # Add judgment to improvements
            improvements["judgment"] = {
                "score": judgment.score,
                "feedback": judgment.feedback,
                "strengths": judgment.strengths,
                "weaknesses": judgment.weaknesses,
            }
            
            # End root span
            if self.trace_manager and root_span:
                self.trace_manager.end_span(root_span.span_id)
            
            # Save improvements to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"improvements_{timestamp}.json")
            
            with open(output_path, "w") as f:
                json.dump(improvements, f, indent=2)
            
            logger.info(f"Improvement analysis completed and saved to {output_path}")
            
            return {
                "improvements": improvements,
                "trace_id": trace_id,
                "output_path": output_path,
            }
        
        except Exception as e:
            logger.error(f"Error during agent improvement: {e}")
            
            # End root span with error
            if self.trace_manager and root_span:
                root_span.set_status("ERROR")
                self.trace_manager.end_span(root_span.span_id)
            
            return {
                "error": str(e),
                "trace_id": trace_id,
            }


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Research Agent Example")
    parser.add_argument("--model", type=str, default="openai://gpt-4",
                        help="Model URI")
    parser.add_argument("--docs-dir", type=str, default="./example_documents",
                        help="Directory containing documents")
    parser.add_argument("--output-dir", type=str, default="./agent_output",
                        help="Output directory")
    parser.add_argument("--query", type=str, default="What are the key insights from these documents?",
                        help="Query for document analysis")
    parser.add_argument("--disable-gasa", action="store_true",
                        help="Disable GASA")
    parser.add_argument("--disable-monitoring", action="store_true",
                        help="Disable monitoring")
    
    args = parser.parse_args()
    
    # Create agent
    agent = ResearchAgent(
        model_uri=args.model,
        output_dir=args.output_dir,
        enable_gasa=not args.disable_gasa,
        enable_monitoring=not args.disable_monitoring,
    )
    
    # Add documents
    if os.path.isdir(args.docs_dir):
        await agent.add_documents_from_directory(args.docs_dir)
    else:
        logger.warning(f"Documents directory not found: {args.docs_dir}")
        logger.info("Creating example documents...")
        
        # Create example directory
        os.makedirs(args.docs_dir, exist_ok=True)
        
        # Create example documents
        example_docs = [
            {
                "filename": "research_paper_1.txt",
                "content": "This research paper discusses the impact of climate change on biodiversity. "
                          "The findings suggest that rising temperatures are causing significant shifts in species distribution. "
                          "Coral reefs are particularly vulnerable, with up to 50% of reefs already damaged.",
            },
            {
                "filename": "research_paper_2.txt",
                "content": "This study examines the economic implications of climate change. "
                          "It estimates that without mitigation, global GDP could decrease by up to 18% by 2050. "
                          "The paper also discusses how renewable energy investments can offset these losses.",
            },
            {
                "filename": "news_article_1.txt",
                "content": "Recent floods in coastal cities have raised concerns about sea level rise. "
                          "Scientists attribute these events to climate change and warn that such incidents will become more frequent. "
                          "Local governments are beginning to implement adaptation strategies.",
            },
        ]
        
        for doc in example_docs:
            with open(os.path.join(args.docs_dir, doc["filename"]), "w") as f:
                f.write(doc["content"])
        
        await agent.add_documents_from_directory(args.docs_dir)
    
    # Analyze documents
    analysis_result = await agent.analyze_documents(args.query)
    
    if "error" in analysis_result:
        logger.error(f"Analysis failed: {analysis_result['error']}")
        return
    
    # Generate visualization
    visualization_path = await agent.generate_visualization(analysis_result["analysis"])
    
    # Improve agent
    improvement_result = await agent.improve_agent()
    
    # Print summary
    print("\n=== Research Agent Example ===")
    print(f"Model: {args.model}")
    print(f"GASA enabled: {not args.disable_gasa}")
    print(f"Monitoring enabled: {not args.disable_monitoring}")
    print(f"Documents directory: {args.docs_dir}")
    print(f"Query: {args.query}")
    print(f"Analysis saved to: {analysis_result.get('output_path', 'N/A')}")
    print(f"Visualization saved to: {visualization_path}")
    print(f"Improvements saved to: {improvement_result.get('output_path', 'N/A')}")
    
    if not args.disable_monitoring and analysis_result.get("trace_id"):
        print(f"Trace ID: {analysis_result['trace_id']}")
        print("Trace visualizations saved to the visualizations directory")


if __name__ == "__main__":
    asyncio.run(main())
