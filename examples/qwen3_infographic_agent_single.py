"""
End-to-end example of an autonomous agent using vLLM with a single Agent.

This example demonstrates how to:
1. Fetch the top paper from Hugging Face Daily Papers
2. Analyze the paper using a single comprehensive Agent
3. Generate an infographic using React and D3 to visualize the key findings

The example uses:
- vLLM for high-performance inference with Qwen/Qwen3-30B-A3B or other available models
- Function calling for tool integration
- Base Agent class with integrated memory, planning, execution, and validation
- GASA for efficient attention patterns
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import arxiv
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, ScoringDimension
from saplings.memory import MemoryStore, Document
from saplings.tool import FunctionRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a function registry for tools
function_registry = FunctionRegistry()

# Paper retrieval functions
def get_hugging_face_top_daily_paper() -> Dict[str, str]:
    """
    Get the top paper from Hugging Face Daily Papers.

    Returns:
        Dict[str, str]: Paper information including title, authors, abstract, and URL
    """
    try:
        # Fetch the daily papers page
        url = "https://huggingface.co/papers"
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first paper
        paper_div = soup.find('div', class_='paper-card')

        if not paper_div:
            logger.warning("No papers found on Hugging Face Daily Papers")
            # Return a default paper as fallback
            return {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
                "url": "https://arxiv.org/abs/1706.03762"
            }

        # Extract paper information
        title_elem = paper_div.find('h3')
        title = title_elem.text.strip() if title_elem else "Unknown Title"

        authors_elem = paper_div.find('div', class_='authors')
        authors = authors_elem.text.strip() if authors_elem else "Unknown Authors"

        abstract_elem = paper_div.find('div', class_='abstract')
        abstract = abstract_elem.text.strip() if abstract_elem else "No abstract available"

        # Get the paper URL
        url_elem = paper_div.find('a', href=True)
        paper_url = url_elem['href'] if url_elem else None

        # If the URL is relative, make it absolute
        if paper_url and not paper_url.startswith('http'):
            paper_url = f"https://huggingface.co{paper_url}"

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": paper_url or "https://huggingface.co/papers"
        }
    except Exception as e:
        logger.error(f"Error fetching paper from Hugging Face: {e}")
        # Return a default paper as fallback
        return {
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
            "url": "https://arxiv.org/abs/1706.03762"
        }

def get_paper_id(url: str) -> Optional[str]:
    """
    Extract the paper ID from a URL.

    Args:
        url: URL of the paper

    Returns:
        Optional[str]: Paper ID if found, None otherwise
    """
    # Try to extract arXiv ID
    arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)'
    match = re.search(arxiv_pattern, url)
    if match:
        return match.group(1)

    # Try to extract arXiv ID from the URL path
    path_pattern = r'/(\d+\.\d+)(?:v\d+)?$'
    match = re.search(path_pattern, url)
    if match:
        return match.group(1)

    return None

def read_pdf_file(file_path: str, max_pages: int = 5) -> str:
    """
    Read the first few pages of a PDF file.

    Args:
        file_path: Path to the PDF file
        max_pages: Maximum number of pages to read

    Returns:
        str: Content of the PDF file
    """
    try:
        reader = PdfReader(file_path)
        content = ""

        # Read the first few pages
        for i in range(min(max_pages, len(reader.pages))):
            page = reader.pages[i]
            content += page.extract_text() + "\n\n"

        return content
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return "Error reading PDF file"

async def get_paper_content(paper_id: str) -> Dict[str, Any]:
    """
    Get the content of a paper from arXiv.

    Args:
        paper_id: arXiv ID of the paper

    Returns:
        Dict[str, Any]: Paper content including title, authors, abstract, and full text
    """
    try:
        # Search for the paper
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))

        if not results:
            logger.warning(f"Paper with ID {paper_id} not found")
            return {
                "title": "Unknown",
                "authors": "Unknown",
                "abstract": "Not available",
                "content": "Paper not found"
            }

        paper = results[0]

        # Extract metadata
        title = paper.title
        authors = ", ".join(author.name for author in paper.authors)
        abstract = paper.summary

        # Download the PDF
        pdf_path = f"paper_{paper_id.replace('.', '_')}.pdf"
        paper.download_pdf(filename=pdf_path)
        logger.info(f"Downloaded PDF to {pdf_path}")

        # Read the first few pages of the PDF
        pdf_content = read_pdf_file(pdf_path)

        # Combine metadata and content
        content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}\n\nPaper Content:\n{pdf_content}"

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "content": content,
            "pdf_path": pdf_path
        }
    except Exception as e:
        logger.error(f"Error getting paper content: {e}")
        return {
            "title": "Error",
            "authors": "Error",
            "abstract": "Error getting paper content",
            "content": f"Error: {str(e)}",
            "pdf_path": None
        }

# Tool functions for infographic generation
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

def get_react_app_template() -> Dict[str, str]:
    """
    Get a template for a React app.

    Returns:
        Dict[str, str]: A dictionary containing the HTML, CSS, and JS templates
    """
    return {
        "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{title}}</title>
  <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div id="root"></div>
  <script type="text/babel" src="app.js"></script>
</body>
</html>""",
        "styles.css": """body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  margin: 0;
  padding: 0;
  background-color: {{background_color}};
  color: {{text_color}};
}

.infographic-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 28px;
  margin-bottom: 10px;
}

.authors {
  font-style: italic;
  color: #666;
}

.section {
  margin-bottom: 40px;
}

h2 {
  border-bottom: 2px solid #ddd;
  padding-bottom: 10px;
  margin-bottom: 20px;
}

.visualization {
  display: flex;
  justify-content: center;
  margin: 30px 0;
}

.key-findings {
  background-color: {{highlight_color}};
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.key-findings ul {
  margin: 0;
  padding-left: 20px;
}

.key-findings li {
  margin-bottom: 10px;
}

.footer {
  text-align: center;
  margin-top: 50px;
  font-size: 14px;
  color: #666;
}""",
        "app.js": """// Data for the infographic
const paperData = {
  title: "{{title}}",
  authors: "{{authors}}",
  abstract: "{{abstract}}",
  keyFindings: {{key_findings}},
  chartData: {{chart_data}}
};

// App Component
const App = () => {
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
      <ul>
        {findings.map((finding, index) => (
          <li key={index}>{finding}</li>
        ))}
      </ul>
    </div>
  </div>
);

// Visualization Component
const Visualization = ({ data }) => {
  const svgRef = React.useRef(null);
  const width = 600;
  const height = 400;
  const margin = 50;

  React.useEffect(() => {
    if (!data || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    {{chart_code}}
  }, [data]);

  return (
    <div className="section">
      <h2>Visualization</h2>
      <div className="visualization">
        <svg ref={svgRef} width={width} height={height}></svg>
      </div>
    </div>
  );
};

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

def get_d3_chart_template(chart_type: str) -> Dict[str, str]:
    """
    Get a template for a D3 chart.

    Args:
        chart_type: Type of chart (bar, line, pie, etc.)

    Returns:
        Dict[str, str]: A dictionary containing the chart template
    """
    templates = {
        "bar": {
            "template": """const margin = { top: 20, right: 30, bottom: 40, left: 40 };
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
    .attr("fill", "steelblue");"""
        },
        "line": {
            "template": """const margin = { top: 20, right: 30, bottom: 30, left: 40 };
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const x = d3.scalePoint()
  .domain(data.map(d => d.label))
  .range([0, innerWidth]);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.value)])
  .nice()
  .range([innerHeight, 0]);

const line = d3.line()
  .x(d => x(d.label))
  .y(d => y(d.value))
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
  .attr("stroke", "steelblue")
  .attr("stroke-width", 2)
  .attr("d", line);

g.selectAll(".dot")
  .data(data)
  .enter().append("circle")
    .attr("class", "dot")
    .attr("cx", d => x(d.label))
    .attr("cy", d => y(d.value))
    .attr("r", 4)
    .attr("fill", "steelblue");"""
        },
        "pie": {
            "template": """const radius = Math.min(width, height) / 2 - margin;

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
  .attr("dy", "0.35em")
  .text(d => d.data.label)
  .style("text-anchor", "middle")
  .style("font-size", "12px")
  .style("fill", "white");"""
        },
        "scatter": {
            "template": """const margin = { top: 20, right: 30, bottom: 30, left: 40 };
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const x = d3.scaleLinear()
  .domain(d3.extent(data, d => d.x))
  .nice()
  .range([0, innerWidth]);

const y = d3.scaleLinear()
  .domain(d3.extent(data, d => d.y))
  .nice()
  .range([innerHeight, 0]);

const g = svg.append("g")
  .attr("transform", `translate(${margin.left},${margin.top})`);

g.append("g")
  .attr("transform", `translate(0,${innerHeight})`)
  .call(d3.axisBottom(x));

g.append("g")
  .call(d3.axisLeft(y));

g.selectAll(".dot")
  .data(data)
  .enter().append("circle")
    .attr("class", "dot")
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y))
    .attr("r", 5)
    .attr("fill", "steelblue");"""
        }
    }

    return templates.get(chart_type.lower(), {"template": "// Custom chart code goes here"})

@function_registry.register(
    description="Save the infographic files to disk",
    group="file_operations"
)
def save_infographic_files(
    title: str,
    html: str,
    css: str,
    js: str,
    output_dir: str = "./infographic"
) -> Dict[str, str]:
    """
    Save the infographic files to disk.

    Args:
        title: Title of the infographic
        html: HTML content
        css: CSS content
        js: JavaScript content
        output_dir: Output directory

    Returns:
        Dict[str, str]: A dictionary containing the file paths
    """
    try:
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create sanitized filename from title
        filename_base = re.sub(r'[^\w\s-]', '', title).strip().lower()
        filename_base = re.sub(r'[-\s]+', '-', filename_base)

        # Save the files
        html_path = os.path.join(output_dir, "index.html")
        css_path = os.path.join(output_dir, "styles.css")
        js_path = os.path.join(output_dir, "app.js")

        with open(html_path, "w") as f:
            f.write(html)

        with open(css_path, "w") as f:
            f.write(css)

        with open(js_path, "w") as f:
            f.write(js)

        return {
            "html_path": html_path,
            "css_path": css_path,
            "js_path": js_path,
            "output_dir": output_dir
        }
    except Exception as e:
        logger.error(f"Error saving infographic files: {e}")
        return {
            "error": str(e)
        }

# Model creation function
async def create_model(model_name: str = "Qwen/Qwen3-30B-A3B") -> LLM:
    """
    Create a vLLM model with the specified model name.

    Args:
        model_name: Name of the model to use

    Returns:
        LLM: The created model
    """
    # We'll skip the model existence check and let vLLM handle it
    logger.info(f"Using model: {model_name}")

    # Create a vLLM model with function calling enabled
    try:
        # Create model parameters
        model_params = {
            "temperature": 0.7,
            "max_tokens": 4096
        }

        # Only add function calling parameters if we're using a model that supports it
        if "gpt2" not in model_name.lower():
            model_params["enable_tool_choice"] = True

        logger.info(f"Creating model: vllm/{model_name} with parameters: {model_params}")

        try:
            # Try with the parameters using the new approach
            model = LLM.create(
                provider="vllm",
                model=model_name,
                **model_params
            )
            logger.info(f"Model created successfully: {model_name}")
            return model
        except RuntimeError as e:
            # If there's any parameter issue, try with minimal parameters
            if "parameter" in str(e).lower() or "argument" in str(e).lower():
                logger.warning(f"Parameter error: {e}. Trying with minimal parameters.")
                # Use minimal parameters
                model = LLM.create(
                    provider="vllm",
                    model=model_name,
                    temperature=0.7,
                    max_tokens=4096
                )
                logger.info(f"Model created successfully with minimal parameters: {model_name}")
                return model
            # If the model doesn't exist, try with a local model if available
            elif "Invalid repository ID" in str(e) or "Repository Not Found" in str(e):
                # Check if there's a local model directory we can use
                local_models_dir = os.environ.get("LOCAL_MODELS_DIR", "./models")
                if os.path.exists(local_models_dir):
                    local_models = [d for d in os.listdir(local_models_dir)
                                   if os.path.isdir(os.path.join(local_models_dir, d))]
                    if local_models:
                        local_model_path = os.path.join(local_models_dir, local_models[0])
                        logger.warning(f"Model {model_name} not found. Trying with local model: {local_model_path}")
                        return await create_model(local_model_path)

                # If no local model is available, try with a different Hugging Face model
                logger.warning(f"Model {model_name} not found. Trying with mistralai/Mistral-7B-v0.1.")
                return await create_model("mistralai/Mistral-7B-v0.1")
            else:
                # If it's a different error, re-raise it
                raise
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

# Create judge agent for evaluation
async def create_judge() -> JudgeAgent:
    """
    Create a judge agent for evaluating the quality of the generated infographic.

    Returns:
        JudgeAgent: The configured judge agent
    """
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

        Content Accuracy:
        - Does the infographic accurately represent the paper's findings?
        - Are the key concepts explained correctly?
        - Is the information presented in a way that maintains scientific integrity?

        Visual Clarity:
        - Is the information organized in a logical, easy-to-follow manner?
        - Are the visualizations clear and easy to understand?
        - Does the design help or hinder understanding of the content?

        Design Quality:
        - Is the infographic visually appealing?
        - Does it use color, typography, and layout effectively?
        - Is there a consistent visual style throughout?

        Code Quality:
        - Is the React code well-structured and maintainable?
        - Are components properly organized and named?
        - Is the code DRY (Don't Repeat Yourself) and following best practices?

        Technical Implementation:
        - Does the implementation make good use of React features?
        - Is D3 integrated effectively with React?
        - Are there any performance concerns or bugs?
        """,
    )

    # Create the judge agent with its own model
    judge_agent = JudgeAgent(
        provider="vllm",
        model_name="Qwen/Qwen3-30B-A3B",
        config=judge_config,
        temperature=0.4,  # Lower temperature for more consistent evaluations
        max_tokens=4096,
        enable_tool_choice=True
    )
    logger.info("Created judge agent with infographic rubric and dedicated model")

    return judge_agent

# Main function
async def run_infographic_agent():
    """
    Run the infographic agent to analyze a paper and create an infographic.
    """
    try:
        # Create the agent configuration using the new approach
        config = AgentConfig(
            provider="vllm",
            model_name="Qwen/Qwen3-30B-A3B",
            memory_path="./agent_memory",
            output_dir="./agent_output",
            enable_gasa=True,
            enable_monitoring=True,
            enable_self_healing=True,
            enable_tool_factory=True,
            max_tokens=4096,
            temperature=0.7,
            gasa_max_hops=2,
            retrieval_entropy_threshold=0.1,
            retrieval_max_documents=10,
            planner_budget_strategy="token_count",
            executor_verification_strategy="judge",
            tool_factory_sandbox_enabled=True,
            allowed_imports=["json", "re", "os"],
            enable_tool_choice=True  # Enable function calling for vLLM
        )

        # Create the agent
        agent = Agent(config=config)

        # No need to set the model explicitly, the Agent class will create it from the config

        # Register tools with the agent
        for tool in function_registry.get_all_tools():
            agent.executor.register_tool(tool)

        # Create the judge with its own model
        judge_agent = await create_judge()

        # Step 1: Get the top paper
        print("\nStep 1: Getting the top paper from Hugging Face Daily Papers...")
        paper_info = get_hugging_face_top_daily_paper()

        print(f"Top paper: {paper_info['title']}")
        print(f"Authors: {paper_info['authors']}")
        print(f"Abstract: {paper_info['abstract'][:200]}...")

        # Step 2: Get the paper ID and download the full paper
        print("\nStep 2: Getting the paper ID and downloading the full paper...")

        # First try to get the ID from the URL
        paper_id = get_paper_id(paper_info['url'])

        # If we couldn't get the ID, use a default paper
        if not paper_id:
            print("Could not extract paper ID from URL. Using default paper.")
            paper_id = "1706.03762"  # Attention Is All You Need

        print(f"Paper ID: {paper_id}")

        # Get the paper content
        paper_content = await get_paper_content(paper_id)

        # Initialize memory store with paper content
        memory_store = MemoryStore()

        # Add paper content as a document
        doc = Document(
            id=f"paper_{paper_id}",
            content=paper_content['content'],
            metadata={
                "title": paper_content['title'],
                "authors": paper_content['authors'],
                "abstract": paper_content['abstract'],
                "source": "arxiv",
                "paper_id": paper_id,
            }
        )
        memory_store.add_document(doc)

        # Set the memory store
        agent.memory_store = memory_store

        # Step 3: Analyze the paper and create an infographic
        print("\nStep 3: Analyzing the paper and creating an infographic...")

        task = f"""
        Analyze the following paper and create an infographic using React and D3:

        Title: {paper_content['title']}
        Authors: {paper_content['authors']}
        Abstract: {paper_content['abstract']}

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

        # Run the agent
        print("\nRunning the agent to analyze the paper and create an infographic...")
        result = await agent.run(task)

        # Extract the final result
        final_result = result.get("final_result", "No result generated")

        print("\nInfographic generation completed!")
        print(f"Output: {final_result[:500]}...")

        # Step 4: Judge the result
        print("\nStep 4: Judging the quality of the infographic...")
        judgment = await judge_agent.judge(
            output=final_result,
            prompt=task,
        )

        print(f"Judgment score: {judgment.score}")
        print(f"Passed: {judgment.passed}")
        print(f"Critique: {judgment.critique[:500]}...")

        return {
            "paper": paper_content,
            "infographic": final_result,
            "judgment": judgment,
        }

    except Exception as e:
        logger.error(f"Error running infographic agent: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    asyncio.run(run_infographic_agent())
