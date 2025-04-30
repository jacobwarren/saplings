"""
End-to-end example of an autonomous agent using Qwen3-32B with vLLM.

This example demonstrates how to:
1. Fetch the top paper from Hugging Face Daily Papers
2. Analyze the paper using a multi-agent system
3. Generate an infographic using React and D3 to visualize the key findings

The example uses:
- vLLM for high-performance inference with Qwen3-32B
- Function calling for tool integration
- GraphRunner for multi-agent orchestration
- JudgeAgent for quality assessment
- GASA for efficient attention patterns
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup

from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig
from saplings.judge.config import Rubric, ScoringDimension
from saplings.orchestration import (
    AgentNode,
    CommunicationChannel,
    GraphRunner,
    GraphRunnerConfig,
    NegotiationStrategy,
)
from saplings.core.function_registry import function_registry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Register tools for the agents to use
@function_registry.register(
    description="Get the most upvoted paper on Hugging Face daily papers",
    group="research"
)
def get_hugging_face_top_daily_paper() -> Dict[str, str]:
    """
    Get the most upvoted paper on Hugging Face daily papers.

    Returns:
        Dict[str, str]: A dictionary containing the title, authors, abstract, and URL of the paper
    """
    try:
        url = "https://huggingface.co/papers"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the paper data from the JSON-like data in the "data-props" attribute
        containers = soup.find_all('div', class_='SVELTE_HYDRATER contents')

        for container in containers:
            data_props = container.get('data-props', '')
            if data_props:
                try:
                    # Parse the JSON-like string
                    json_data = json.loads(data_props.replace('&quot;', '"'))
                    if 'dailyPapers' in json_data and json_data['dailyPapers']:
                        top_paper = json_data['dailyPapers'][0]
                        return {
                            "title": top_paper.get('title', ''),
                            "authors": ", ".join(top_paper.get('authors', [])),
                            "abstract": top_paper.get('abstract', ''),
                            "url": f"https://huggingface.co/papers/{top_paper.get('id', '')}"
                        }
                except json.JSONDecodeError:
                    continue

        return {
            "title": "No paper found",
            "authors": "",
            "abstract": "",
            "url": ""
        }
    except Exception as e:
        logger.error(f"Error fetching top paper: {e}")
        return {
            "title": f"Error: {str(e)}",
            "authors": "",
            "abstract": "",
            "url": ""
        }

@function_registry.register(
    description="Get the paper ID from arXiv URL or paper title",
    group="research"
)
def get_paper_id(url_or_title: str) -> str:
    """
    Extract the paper ID from an arXiv URL or search for it by title.

    Args:
        url_or_title: The URL of the paper or its title

    Returns:
        str: The paper ID
    """
    # Check if it's a URL with an ID
    arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)'
    hf_pattern = r'huggingface\.co/papers/(\d+\.\d+)'

    arxiv_match = re.search(arxiv_pattern, url_or_title)
    hf_match = re.search(hf_pattern, url_or_title)

    if arxiv_match:
        return arxiv_match.group(1)
    elif hf_match:
        return hf_match.group(1)

    # If it's just an ID
    if re.match(r'^\d+\.\d+$', url_or_title):
        return url_or_title

    # If it's a title, we would need to search for it
    # For simplicity, we'll just return an empty string
    return ""

@function_registry.register(
    description="Download a paper from arXiv",
    group="research"
)
def download_paper(paper_id: str) -> Dict[str, str]:
    """
    Download a paper from arXiv and extract its content.

    Args:
        paper_id: The arXiv ID of the paper

    Returns:
        Dict[str, str]: A dictionary containing the paper's content
    """
    try:
        # For this example, we'll simulate downloading by fetching the abstract
        url = f"https://arxiv.org/abs/{paper_id}"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the abstract
        abstract_element = soup.find('blockquote', class_='abstract')
        abstract = abstract_element.text.replace('Abstract: ', '') if abstract_element else "Abstract not found"

        # Extract the title
        title_element = soup.find('h1', class_='title')
        title = title_element.text.replace('Title:', '').strip() if title_element else "Title not found"

        # Extract the authors
        authors_element = soup.find('div', class_='authors')
        authors = authors_element.text.replace('Authors:', '').strip() if authors_element else "Authors not found"

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "content": f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}\n\nNote: This is a simulated download that only includes the abstract. In a real implementation, we would download and parse the full PDF."
        }
    except Exception as e:
        logger.error(f"Error downloading paper: {e}")
        return {
            "title": "",
            "authors": "",
            "abstract": "",
            "content": f"Error downloading paper: {str(e)}"
        }

@function_registry.register(
    description="Get D3.js chart templates for different visualization types",
    group="visualization"
)
def get_d3_chart_template(chart_type: str) -> Dict[str, str]:
    """
    Get D3.js chart templates for different visualization types.

    Args:
        chart_type: Type of chart (bar, line, pie, scatter, heatmap, network, treemap)

    Returns:
        Dict[str, str]: A dictionary containing template code for the requested chart type
    """
    templates = {
        "bar": """
// Bar chart template with D3.js
const BarChart = ({ data, width = 800, height = 500, margin = { top: 20, right: 30, bottom: 40, left: 60 } }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!data || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleBand()
      .domain(data.map(d => d.label))
      .range([0, innerWidth])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .nice()
      .range([innerHeight, 0]);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end");

    g.append("g")
      .call(d3.axisLeft(y));

    g.selectAll(".bar")
      .data(data)
      .enter().append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.label))
        .attr("y", d => y(d.value))
        .attr("width", x.bandwidth())
        .attr("height", d => innerHeight - y(d.value))
        .attr("fill", "#4285F4");
  }, [data, width, height, margin]);

  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};
        """,

        "line": """
// Line chart template with D3.js
const LineChart = ({ data, width = 800, height = 500, margin = { top: 20, right: 30, bottom: 40, left: 60 } }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!data || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scalePoint()
      .domain(data.map(d => d.x))
      .range([0, innerWidth]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.y)])
      .nice()
      .range([innerHeight, 0]);

    const line = d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
      .curve(d3.curveMonotoneX);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x));

    g.append("g")
      .call(d3.axisLeft(y));

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#4285F4")
      .attr("stroke-width", 2)
      .attr("d", line);

    g.selectAll(".dot")
      .data(data)
      .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .attr("r", 4)
        .attr("fill", "#4285F4");
  }, [data, width, height, margin]);

  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};
        """,

        "pie": """
// Pie chart template with D3.js
const PieChart = ({ data, width = 500, height = 500, margin = 40 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!data || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const radius = Math.min(width, height) / 2 - margin;

    const color = d3.scaleOrdinal()
      .domain(data.map(d => d.label))
      .range(d3.schemeCategory10);

    const pie = d3.pie()
      .value(d => d.value);

    const arc = d3.arc()
      .innerRadius(0)
      .outerRadius(radius);

    const g = svg.append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`);

    const arcs = g.selectAll(".arc")
      .data(pie(data))
      .enter().append("g")
        .attr("class", "arc");

    arcs.append("path")
      .attr("d", arc)
      .attr("fill", d => color(d.data.label))
      .attr("stroke", "white")
      .style("stroke-width", "2px");

    arcs.append("text")
      .attr("transform", d => `translate(${arc.centroid(d)})`)
      .attr("dy", ".35em")
      .style("text-anchor", "middle")
      .text(d => d.data.label);

    // Add a legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 100},20)`);

    data.forEach((d, i) => {
      const legendRow = legend.append("g")
        .attr("transform", `translate(0,${i * 20})`);

      legendRow.append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", color(d.label));

      legendRow.append("text")
        .attr("x", 20)
        .attr("y", 10)
        .attr("text-anchor", "start")
        .style("font-size", "12px")
        .text(d.label);
    });
  }, [data, width, height, margin]);

  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};
        """,

        "network": """
// Network graph template with D3.js
const NetworkGraph = ({ nodes, links, width = 800, height = 600 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!nodes || !nodes.length || !links || !links.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.value || 1));

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
        .attr("r", d => d.size || 5)
        .attr("fill", d => d.color || "#69b3a2")
        .call(drag(simulation));

    node.append("title")
      .text(d => d.id);

    const text = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
        .text(d => d.id)
        .attr("font-size", 10)
        .attr("dx", 12)
        .attr("dy", 4);

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      text
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    function drag(simulation) {
      function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }

      function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }

      function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }

      return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    }
  }, [nodes, links, width, height]);

  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};
        """
    }

    if chart_type.lower() not in templates:
        return {
            "error": f"Chart type '{chart_type}' not found. Available types: {', '.join(templates.keys())}",
            "available_types": list(templates.keys())
        }

    return {
        "template": templates[chart_type.lower()],
        "chart_type": chart_type.lower(),
        "usage_example": f"<{chart_type.capitalize()}Chart data={{yourData}} width={{800}} height={{500}} />"
    }

@function_registry.register(
    description="Get React app template for creating an infographic",
    group="visualization"
)
def get_react_app_template() -> Dict[str, str]:
    """
    Get a React app template for creating an infographic.

    Returns:
        Dict[str, str]: A dictionary containing template code for a React app
    """
    return {
        "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scientific Paper Infographic</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div id="root"></div>
  <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/d3@7"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script type="text/babel" src="app.js"></script>
</body>
</html>""",

        "styles.css": """/* Base styles */
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  line-height: 1.6;
  color: #333;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.infographic-container {
  background-color: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 30px;
  margin-bottom: 30px;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
  color: #2c3e50;
}

.header .authors {
  font-style: italic;
  color: #7f8c8d;
  margin-bottom: 20px;
}

.section {
  margin-bottom: 40px;
}

.section h2 {
  font-size: 1.8rem;
  border-bottom: 2px solid #3498db;
  padding-bottom: 10px;
  margin-bottom: 20px;
  color: #2c3e50;
}

.chart-container {
  background-color: white;
  border-radius: 4px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.key-findings {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.finding-card {
  background-color: white;
  border-left: 4px solid #3498db;
  padding: 15px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.finding-card h3 {
  margin-top: 0;
  color: #3498db;
}

.footer {
  text-align: center;
  margin-top: 40px;
  font-size: 0.9rem;
  color: #7f8c8d;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .key-findings {
    grid-template-columns: 1fr;
  }

  .header h1 {
    font-size: 2rem;
  }

  .section h2 {
    font-size: 1.5rem;
  }
}""",

        "app.js": """// Main App Component
const App = () => {
  // Replace with your actual data
  const paperData = {
    title: "Your Paper Title",
    authors: "Author 1, Author 2, Author 3",
    abstract: "Paper abstract goes here...",
    keyFindings: [
      { id: 1, title: "Finding 1", description: "Description of finding 1" },
      { id: 2, title: "Finding 2", description: "Description of finding 2" },
      { id: 3, title: "Finding 3", description: "Description of finding 3" }
    ],
    chartData: {
      // Add your chart data here
    }
  };

  return (
    <div className="infographic-container">
      <Header title={paperData.title} authors={paperData.authors} />
      <Summary abstract={paperData.abstract} />
      <KeyFindings findings={paperData.keyFindings} />
      <Visualization data={paperData.chartData} />
      <Methodology />
      <Conclusion />
      <Footer />
    </div>
  );
};

// Header Component
const Header = ({ title, authors }) => (
  <div className="header">
    <h1>{title}</h1>
    <div className="authors">{authors}</div>
  </div>
);

// Summary Component
const Summary = ({ abstract }) => (
  <div className="section">
    <h2>Abstract</h2>
    <p>{abstract}</p>
  </div>
);

// Key Findings Component
const KeyFindings = ({ findings }) => (
  <div className="section">
    <h2>Key Findings</h2>
    <div className="key-findings">
      {findings.map(finding => (
        <div key={finding.id} className="finding-card">
          <h3>{finding.title}</h3>
          <p>{finding.description}</p>
        </div>
      ))}
    </div>
  </div>
);

// Visualization Component
const Visualization = ({ data }) => (
  <div className="section">
    <h2>Visualization</h2>
    <div className="chart-container">
      {/* Add your D3 chart component here */}
      <p>Chart placeholder - replace with your D3 visualization</p>
    </div>
  </div>
);

// Methodology Component
const Methodology = () => (
  <div className="section">
    <h2>Methodology</h2>
    <p>Description of the methodology used in the paper...</p>
  </div>
);

// Conclusion Component
const Conclusion = () => (
  <div className="section">
    <h2>Conclusion</h2>
    <p>Summary of the paper's conclusions and implications...</p>
  </div>
);

// Footer Component
const Footer = () => (
  <div className="footer">
    <p>Infographic created with React and D3.js</p>
  </div>
);

// Render the App
ReactDOM.render(<App />, document.getElementById('root'));"""
    }

@function_registry.register(
    description="Generate React and D3 code for an infographic",
    group="visualization"
)
def generate_infographic_code(
    title: str,
    data: Dict[str, Any],
    chart_type: str,
    width: int = 800,
    height: int = 600,
    theme: str = "light"
) -> Dict[str, str]:
    """
    Generate React and D3 code for an infographic.

    Args:
        title: Title of the infographic
        data: Data to visualize
        chart_type: Type of chart (bar, line, pie, etc.)
        width: Width of the chart in pixels
        height: Height of the chart in pixels
        theme: Color theme (light or dark)

    Returns:
        Dict[str, str]: A dictionary containing the generated code
    """
    # Get the React app template
    react_template = get_react_app_template()

    # Get the D3 chart template
    chart_template = get_d3_chart_template(chart_type)

    # This would normally be generated by the LLM, but for this example,
    # we'll return templates that the LLM can fill in
    return {
        "html": react_template["index.html"],
        "css": react_template["styles.css"],
        "js": react_template["app.js"],
        "chart": chart_template.get("template", "// Chart template not found"),
        "data": json.dumps(data, indent=2)
    }

async def create_model(model_name: str = "Qwen/Qwen3-32B") -> LLM:
    """
    Create a vLLM model with the specified model name.

    Args:
        model_name: Name of the model to use

    Returns:
        LLM: The created model
    """
    # Create a vLLM model with function calling enabled
    # For Qwen models, we use the hermes parser
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=4096&enable_tool_choice=true&tool_call_parser=hermes"

    logger.info(f"Creating model: {model_uri}")
    model = LLM.from_uri(model_uri)
    logger.info(f"Model created successfully: {model.model_name}")

    return model

async def setup_agents(model: LLM) -> GraphRunner:
    """
    Set up the multi-agent system.

    Args:
        model: The LLM model to use

    Returns:
        GraphRunner: The configured graph runner
    """
    # Create a graph runner with the debate negotiation strategy
    config = GraphRunnerConfig(
        negotiation_strategy=NegotiationStrategy.DEBATE,
        max_rounds=3,
        timeout_seconds=300,
        consensus_threshold=0.8,
    )
    graph_runner = GraphRunner(model=model, config=config)

    # Register agents
    researcher = AgentNode(
        id="researcher",
        name="Research Analyst",
        role="researcher",
        description="""
        You are a Research Analyst specializing in understanding and summarizing scientific papers.
        Your responsibilities include:
        1. Extracting key findings, methodologies, and contributions from research papers
        2. Identifying the most important data points and concepts that should be visualized
        3. Translating complex scientific concepts into clear, accessible explanations
        4. Providing context about why the research is significant and its potential impact

        You have expertise in machine learning, AI, and related fields, allowing you to quickly
        understand technical papers and identify what's truly innovative or important.
        """,
        capabilities=["research", "analysis", "summarization", "scientific_knowledge"],
    )

    visualizer = AgentNode(
        id="visualizer",
        name="Information Designer",
        role="visualizer",
        description="""
        You are an Information Designer specializing in creating clear, compelling visualizations
        of complex data and concepts. Your responsibilities include:
        1. Determining the most effective visualization types for different kinds of data
        2. Creating visual hierarchies that guide viewers through information logically
        3. Designing color schemes, typography, and layouts that enhance understanding
        4. Translating abstract concepts into visual metaphors and diagrams

        You have expertise in information design principles, data visualization best practices,
        and visual communication. You know how to make complex information accessible without
        oversimplifying it.
        """,
        capabilities=["visualization", "design", "information_architecture", "visual_communication"],
    )

    coder = AgentNode(
        id="coder",
        name="Frontend Developer",
        role="coder",
        description="""
        You are a Frontend Developer specializing in creating interactive data visualizations
        with React and D3.js. Your responsibilities include:
        1. Implementing designs as functional, interactive web components
        2. Writing clean, maintainable React code that follows best practices
        3. Using D3.js effectively to create data visualizations
        4. Ensuring visualizations are responsive and accessible

        You have deep expertise in:
        - React (components, hooks, state management)
        - D3.js (scales, axes, transitions, interactions)
        - Modern JavaScript (ES6+)
        - CSS for styling visualizations
        - Web accessibility standards

        You write code that is not just functional but also well-structured, documented,
        and maintainable.
        """,
        capabilities=["coding", "web_development", "react", "d3", "frontend"],
    )

    reviewer = AgentNode(
        id="reviewer",
        name="Technical Reviewer",
        role="reviewer",
        description="""
        You are a Technical Reviewer who evaluates both the content accuracy and technical
        implementation of data visualizations. Your responsibilities include:
        1. Verifying that visualizations accurately represent the underlying data
        2. Checking code for bugs, performance issues, and adherence to best practices
        3. Ensuring visualizations are accessible and work across different devices
        4. Providing constructive feedback on both design and implementation

        You have expertise in both the subject matter (AI/ML research) and the technical
        implementation (React/D3), allowing you to evaluate work from multiple perspectives.
        """,
        capabilities=["code_review", "quality_assurance", "technical_evaluation", "accessibility"],
    )

    # Register the agents
    graph_runner.register_agent(researcher)
    graph_runner.register_agent(visualizer)
    graph_runner.register_agent(coder)
    graph_runner.register_agent(reviewer)

    # Define communication channels
    graph_runner.add_channel(
        CommunicationChannel(
            source_id="researcher",
            target_id="visualizer",
            channel_type="research_results",
            description="Research analysis and key points that should be visualized",
        )
    )

    graph_runner.add_channel(
        CommunicationChannel(
            source_id="visualizer",
            target_id="coder",
            channel_type="visualization_specs",
            description="Detailed visualization specifications including layout, chart types, and design elements",
        )
    )

    graph_runner.add_channel(
        CommunicationChannel(
            source_id="coder",
            target_id="reviewer",
            channel_type="implementation",
            description="Implemented code and visualizations for review",
        )
    )

    graph_runner.add_channel(
        CommunicationChannel(
            source_id="reviewer",
            target_id="coder",
            channel_type="feedback",
            description="Technical feedback and suggestions for improvement",
        )
    )

    graph_runner.add_channel(
        CommunicationChannel(
            source_id="reviewer",
            target_id="visualizer",
            channel_type="design_feedback",
            description="Feedback on visualization design and effectiveness",
        )
    )

    graph_runner.add_channel(
        CommunicationChannel(
            source_id="reviewer",
            target_id="researcher",
            channel_type="accuracy_feedback",
            description="Feedback on the accuracy of represented information",
        )
    )

    return graph_runner

async def create_judge(model: LLM) -> JudgeAgent:
    """
    Create a judge agent for evaluating the quality of the generated infographic.

    Args:
        model: The LLM model to use

    Returns:
        JudgeAgent: The configured judge agent
    """
    try:
        # Try to load the rubric from the JSON file
        from saplings.judge.rubric import RubricLoader
        rubric = RubricLoader.load_from_file("examples/rubrics/infographic_rubric.json")
        logger.info("Loaded infographic rubric from file")
    except Exception as e:
        logger.warning(f"Failed to load rubric from file: {e}")
        logger.info("Creating default rubric")

        # Define a default rubric for evaluating infographics
        dimensions = [
            ScoringDimension(
                name="content_accuracy",
                description="How accurately does the infographic represent the paper's findings and key concepts?",
                weight=0.25,
            ),
            ScoringDimension(
                name="visual_clarity",
                description="How clear and understandable is the infographic?",
                weight=0.2,
            ),
            ScoringDimension(
                name="design_quality",
                description="How visually appealing and well-designed is the infographic?",
                weight=0.15,
            ),
            ScoringDimension(
                name="code_quality",
                description="How well-written and maintainable is the React and D3 code?",
                weight=0.2,
            ),
            ScoringDimension(
                name="technical_implementation",
                description="How well does the implementation use React and D3 features?",
                weight=0.2,
            ),
        ]

        rubric = Rubric(
            name="infographic_rubric",
            description="Rubric for evaluating scientific paper infographics",
            dimensions=dimensions,
            passing_threshold=0.7,
        )

    # Create a judge config with detailed instructions
    judge_config = JudgeConfig(
        rubric=rubric,
        threshold=0.7,
        critique_format="markdown",
        max_tokens_per_judgment=4096,
        instructions="""
        You are evaluating an infographic created from a scientific paper. Your evaluation should be thorough,
        fair, and constructive. Consider both the content accuracy and the technical implementation.

        For each dimension:
        1. Provide specific examples from the submission to justify your score
        2. Highlight both strengths and areas for improvement
        3. Be specific about what would be needed to improve the score

        Your critique should be actionable, helping the creators understand exactly how to improve their work.
        Focus particularly on:
        - Whether the visualization accurately represents the paper's findings
        - How effectively the visualization communicates complex concepts
        - The quality and maintainability of the React and D3 code
        - The visual design and user experience

        Provide at least 3-5 specific, actionable suggestions for improvement.
        """
    )

    # Create the judge agent
    judge_agent = JudgeAgent(model=model, config=judge_config)
    logger.info("Created judge agent with infographic rubric")

    return judge_agent

async def run_infographic_agent():
    """Run the end-to-end infographic agent example."""
    print("=== Qwen3-32B Infographic Agent Example ===")

    try:
        # Create the model
        model = await create_model()

        # Set up the agents
        graph_runner = await setup_agents(model)

        # Create the judge
        judge_agent = await create_judge(model)

        # Step 1: Get the top paper
        print("\nStep 1: Getting the top paper from Hugging Face Daily Papers...")
        paper_info = get_hugging_face_top_daily_paper()

        print(f"Top paper: {paper_info['title']}")
        print(f"Authors: {paper_info['authors']}")
        print(f"Abstract: {paper_info['abstract'][:200]}...")

        # Step 2: Analyze the paper using the multi-agent system
        print("\nStep 2: Analyzing the paper using the multi-agent system...")

        task = f"""
        Analyze the following paper and create an infographic using React and D3:

        Title: {paper_info['title']}
        Authors: {paper_info['authors']}
        Abstract: {paper_info['abstract']}
        URL: {paper_info['url']}

        The infographic should:
        1. Highlight the key findings and contributions
        2. Visualize the most important data or concepts
        3. Be visually appealing and easy to understand
        4. Include appropriate charts or diagrams

        The final output should include:
        1. A summary of the paper
        2. A design for the infographic
        3. React and D3 code to implement the infographic
        """

        # Get the paper ID
        paper_id = get_paper_id(paper_info['url'])
        if paper_id:
            # Download the paper
            print(f"Downloading paper with ID: {paper_id}...")
            paper_content = download_paper(paper_id)

            # Add the paper content to the task
            task += f"\n\nPaper content:\n{paper_content['content'][:1000]}..."

        # First try the contract-net approach for a more structured workflow
        print("Running the multi-agent system using contract-net strategy...")
        try:
            # Configure the graph runner for contract-net
            graph_runner.config.negotiation_strategy = NegotiationStrategy.CONTRACT_NET

            # Use the researcher as the manager
            result = await graph_runner.run_contract_net(
                manager_id="researcher",
                task=task,
                context=f"This is a scientific paper about {paper_info['title']}. The goal is to create an infographic that accurately represents the paper's findings and makes them visually accessible."
            )

            print("Contract-net execution completed successfully")
        except Exception as e:
            logger.warning(f"Contract-net execution failed: {e}")
            logger.info("Falling back to debate strategy")

            # Fall back to debate strategy
            graph_runner.config.negotiation_strategy = NegotiationStrategy.DEBATE
            result = await graph_runner.run_debate(task)

        print("\nMulti-agent system result:")
        print(result[:500] + "..." if len(result) > 500 else result)

        # Step 3: Judge the quality of the result
        print("\nStep 3: Judging the quality of the result...")
        judgment = await judge_agent.judge(
            output=result,
            prompt=task,
        )

        print(f"Overall score: {judgment.overall_score:.2f}")
        print(f"Passed: {judgment.passed}")

        for dimension in judgment.dimension_scores:
            print(f"{dimension.name}: {dimension.score:.2f}")

        print("\nCritique:")
        print(judgment.critique)

        print("\nSuggestions:")
        for suggestion in judgment.suggestions:
            print(f"- {suggestion}")

        # Step 4: Extract and save the infographic code
        print("\nStep 4: Extracting and saving the infographic code...")

        # Extract code blocks from the result
        html_match = re.search(r'```html\n(.*?)\n```', result, re.DOTALL)
        css_match = re.search(r'```css\n(.*?)\n```', result, re.DOTALL)
        js_match = re.search(r'```(?:javascript|js)\n(.*?)\n```', result, re.DOTALL)

        # Create output directory
        os.makedirs("output", exist_ok=True)

        # Save the HTML
        if html_match:
            with open("output/infographic.html", "w") as f:
                f.write(html_match.group(1))
            print("Saved HTML to output/infographic.html")

        # Save the CSS
        if css_match:
            with open("output/infographic.css", "w") as f:
                f.write(css_match.group(1))
            print("Saved CSS to output/infographic.css")

        # Save the JavaScript
        if js_match:
            with open("output/infographic.js", "w") as f:
                f.write(js_match.group(1))
            print("Saved JavaScript to output/infographic.js")

        # Save the full result
        with open("output/full_result.md", "w") as f:
            f.write(result)
        print("Saved full result to output/full_result.md")

        print("\nExample completed successfully!")

        # Clean up
        if hasattr(model, 'cleanup'):
            model.cleanup()

    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main entry point."""
    await run_infographic_agent()

if __name__ == "__main__":
    asyncio.run(main())
