"""
Tests for the Qwen3 infographic agent example.

This test file verifies the functionality of the Qwen3 infographic agent example
by mocking external dependencies and testing the core components.
"""

import asyncio
import json
import os
import sys
import pytest
import requests
from unittest.mock import AsyncMock, MagicMock, patch

from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent
from saplings.judge.config import Rubric, ScoringDimension
from saplings.orchestration import GraphRunner, NegotiationStrategy

# Import the example module
import examples.qwen3_infographic_agent as agent_example

# Mock BeautifulSoup
bs4_mock = MagicMock()
bs4_soup_mock = MagicMock()
bs4_mock.BeautifulSoup = MagicMock(return_value=bs4_soup_mock)
agent_example.bs4 = bs4_mock


@pytest.fixture
def mock_paper_info():
    """Mock paper information returned from Hugging Face."""
    return {
        "title": "Test Paper Title",
        "authors": "Test Author 1, Test Author 2",
        "abstract": "This is a test abstract for the mock paper.",
        "url": "https://huggingface.co/papers/2401.12345"
    }


@pytest.fixture
def mock_paper_content():
    """Mock paper content returned from arXiv."""
    return {
        "title": "Test Paper Title",
        "authors": "Test Author 1, Test Author 2",
        "abstract": "This is a test abstract for the mock paper.",
        "content": "Title: Test Paper Title\n\nAuthors: Test Author 1, Test Author 2\n\nAbstract: This is a test abstract for the mock paper.\n\nNote: This is a simulated download that only includes the abstract."
    }


@pytest.fixture
def mock_model():
    """Mock LLM model."""
    model = AsyncMock(spec=LLM)
    model.model_name = "Qwen/Qwen3-32B"
    model.cleanup = MagicMock()
    return model


@pytest.fixture
def mock_graph_runner():
    """Mock GraphRunner."""
    runner = AsyncMock(spec=GraphRunner)
    runner.config = MagicMock()
    runner.config.negotiation_strategy = NegotiationStrategy.CONTRACT_NET
    runner.register_agent = MagicMock()
    runner.add_channel = MagicMock()
    runner.run_contract_net = AsyncMock(return_value="Mock multi-agent result with infographic code")
    runner.run_debate = AsyncMock(return_value="Mock debate result with infographic code")
    return runner


@pytest.fixture
def mock_judge_agent():
    """Mock JudgeAgent."""
    judge = AsyncMock(spec=JudgeAgent)

    # Create a mock judgment result
    judgment = MagicMock()
    judgment.overall_score = 0.85
    judgment.passed = True
    judgment.dimension_scores = [
        MagicMock(name="content_accuracy", score=0.9),
        MagicMock(name="visual_clarity", score=0.8),
        MagicMock(name="design_quality", score=0.85),
        MagicMock(name="code_quality", score=0.8),
        MagicMock(name="technical_implementation", score=0.85)
    ]
    judgment.critique = "Mock critique of the infographic"
    judgment.suggestions = ["Suggestion 1", "Suggestion 2", "Suggestion 3"]

    judge.judge = AsyncMock(return_value=judgment)
    return judge


@pytest.mark.asyncio
async def test_get_hugging_face_top_daily_paper(mock_paper_info):
    """Test the get_hugging_face_top_daily_paper function."""
    # Direct test of the function
    get_hugging_face_top_daily_paper = agent_example.get_hugging_face_top_daily_paper

    # Mock requests.get directly
    original_get = requests.get
    try:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = """
        <div class="SVELTE_HYDRATER contents" data-props='{"dailyPapers":[
            {"title":"Test Paper Title","authors":["Test Author 1","Test Author 2"],"abstract":"This is a test abstract for the mock paper.","id":"2401.12345"}
        ]}'>
        </div>
        """
        mock_response.raise_for_status = MagicMock()

        # Replace requests.get with a mock
        requests.get = MagicMock(return_value=mock_response)

        # Configure the BeautifulSoup mock
        contents_mock = MagicMock()
        contents_mock.get.return_value = '{"dailyPapers":[{"title":"Test Paper Title","authors":["Test Author 1","Test Author 2"],"abstract":"This is a test abstract for the mock paper.","id":"2401.12345"}]}'

        bs4_soup_mock.find.return_value = contents_mock

        # Manually create the expected result
        expected_result = {
            "title": "Test Paper Title",
            "authors": "Test Author 1, Test Author 2",
            "abstract": "This is a test abstract for the mock paper.",
            "url": "https://huggingface.co/papers/2401.12345"
        }

        # Call the function
        result = get_hugging_face_top_daily_paper()

        # Verify the result
        assert result == expected_result
    finally:
        # Restore the original function
        requests.get = original_get


@pytest.mark.asyncio
async def test_get_paper_id():
    """Test the get_paper_id function."""
    # Test with arXiv URL
    arxiv_url = "https://arxiv.org/abs/2401.12345"
    assert agent_example.get_paper_id(arxiv_url) == "2401.12345"

    # Test with Hugging Face URL
    hf_url = "https://huggingface.co/papers/2401.12345"
    assert agent_example.get_paper_id(hf_url) == "2401.12345"

    # Test with just an ID
    paper_id = "2401.12345"
    assert agent_example.get_paper_id(paper_id) == "2401.12345"

    # Test with a title (should return empty string)
    title = "This is a paper title"
    assert agent_example.get_paper_id(title) == ""


@pytest.mark.asyncio
async def test_download_paper(mock_paper_content):
    """Test the download_paper function."""
    # Direct test of the function
    download_paper = agent_example.download_paper

    # Mock requests.get directly
    original_get = requests.get
    try:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = """
        <h1 class="title">Title: Test Paper Title</h1>
        <div class="authors">Authors: Test Author 1, Test Author 2</div>
        <blockquote class="abstract">Abstract: This is a test abstract for the mock paper.</blockquote>
        """
        mock_response.raise_for_status = MagicMock()

        # Replace requests.get with a mock
        requests.get = MagicMock(return_value=mock_response)

        # Configure the BeautifulSoup mock
        title_mock = MagicMock()
        title_mock.text = "Title: Test Paper Title"

        authors_mock = MagicMock()
        authors_mock.text = "Authors: Test Author 1, Test Author 2"

        abstract_mock = MagicMock()
        abstract_mock.text = "Abstract: This is a test abstract for the mock paper."

        bs4_soup_mock.find.side_effect = lambda selector, **kwargs: {
            ".title": title_mock,
            ".authors": authors_mock,
            ".abstract": abstract_mock
        }.get(selector, MagicMock())

        # Call the function
        result = download_paper("2401.12345")

        # Manually create the expected result
        expected_result = {
            "title": "Test Paper Title",
            "authors": "Test Author 1, Test Author 2",
            "abstract": "This is a test abstract for the mock paper.",
            "content": "Title: Test Paper Title\n\nAuthors: Test Author 1, Test Author 2\n\nAbstract: This is a test abstract for the mock paper.\n\nNote: This is a simulated download that only includes the abstract. In a real implementation, we would download and parse the full PDF."
        }

        # Verify the result
        assert result == expected_result
    finally:
        # Restore the original function
        requests.get = original_get


@pytest.mark.asyncio
async def test_get_d3_chart_template():
    """Test the get_d3_chart_template function."""
    # Test with valid chart type
    bar_result = agent_example.get_d3_chart_template("bar")
    assert "template" in bar_result
    assert "chart_type" in bar_result
    assert "usage_example" in bar_result
    assert bar_result["chart_type"] == "bar"
    assert "BarChart" in bar_result["template"]

    # Test with invalid chart type
    invalid_result = agent_example.get_d3_chart_template("invalid_type")
    assert "error" in invalid_result
    assert "available_types" in invalid_result


@pytest.mark.asyncio
async def test_get_react_app_template():
    """Test the get_react_app_template function."""
    result = agent_example.get_react_app_template()
    assert "index.html" in result
    assert "styles.css" in result
    assert "app.js" in result
    assert "<!DOCTYPE html>" in result["index.html"]
    assert "body {" in result["styles.css"]
    assert "const App = () =>" in result["app.js"]


@pytest.mark.asyncio
async def test_generate_infographic_code():
    """Test the generate_infographic_code function."""
    # Test data
    title = "Test Infographic"
    data = {"key1": "value1", "key2": "value2"}
    chart_type = "bar"

    # Call the function
    result = agent_example.generate_infographic_code(
        title=title,
        data=data,
        chart_type=chart_type
    )

    # Verify the result
    assert "html" in result
    assert "css" in result
    assert "js" in result
    assert "chart" in result
    assert "data" in result
    assert "<!DOCTYPE html>" in result["html"]
    assert "body {" in result["css"]
    assert "const App = () =>" in result["js"]
    assert "BarChart" in result["chart"]
    assert json.loads(result["data"]) == data


@pytest.mark.asyncio
async def test_create_model(mock_model):
    """Test the create_model function."""
    with patch('examples.qwen3_infographic_agent.LLM.from_uri', return_value=mock_model):
        model = await agent_example.create_model()
        assert model == mock_model
        assert model.model_name == "Qwen/Qwen3-32B"


@pytest.mark.asyncio
async def test_setup_agents(mock_model, mock_graph_runner):
    """Test the setup_agents function."""
    with patch('examples.qwen3_infographic_agent.GraphRunner', return_value=mock_graph_runner):
        runner = await agent_example.setup_agents(mock_model)

        # Verify the graph runner was configured correctly
        assert runner == mock_graph_runner
        assert runner.register_agent.call_count == 4  # 4 agents should be registered
        assert runner.add_channel.call_count == 6  # 6 channels should be added


@pytest.mark.asyncio
async def test_create_judge(mock_model, mock_judge_agent):
    """Test the create_judge function."""
    # Mock the ScoringDimension class to allow custom dimensions
    mock_scoring_dimension = MagicMock()

    with patch('examples.qwen3_infographic_agent.JudgeAgent', return_value=mock_judge_agent), \
         patch('examples.qwen3_infographic_agent.ScoringDimension', mock_scoring_dimension), \
         patch('saplings.judge.rubric.RubricLoader.load_from_file', side_effect=Exception("File not found")):

        judge = await agent_example.create_judge(mock_model)

        # Verify the judge was created correctly
        assert judge == mock_judge_agent


@pytest.mark.asyncio
async def test_run_infographic_agent(
    mock_model,
    mock_graph_runner,
    mock_judge_agent,
    mock_paper_info,
    mock_paper_content
):
    """Test the run_infographic_agent function."""
    # Set up all the mocks
    with patch('examples.qwen3_infographic_agent.create_model', return_value=mock_model), \
         patch('examples.qwen3_infographic_agent.setup_agents', return_value=mock_graph_runner), \
         patch('examples.qwen3_infographic_agent.create_judge', return_value=mock_judge_agent), \
         patch('examples.qwen3_infographic_agent.get_hugging_face_top_daily_paper', return_value=mock_paper_info), \
         patch('examples.qwen3_infographic_agent.get_paper_id', return_value="2401.12345"), \
         patch('examples.qwen3_infographic_agent.download_paper', return_value=mock_paper_content), \
         patch('builtins.print'), \
         patch('os.makedirs'), \
         patch('builtins.open', create=True):

        # Mock the regex searches for code extraction
        with patch('examples.qwen3_infographic_agent.re.search') as mock_search:
            # Configure mock_search to return different values for different patterns
            def mock_search_side_effect(pattern, text, flags=0):
                if '```html' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "<html>Mock HTML</html>"
                    return mock_match
                elif '```css' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "body { color: black; }"
                    return mock_match
                elif '```(?:javascript|js)' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "const app = () => {};"
                    return mock_match
                return None

            mock_search.side_effect = mock_search_side_effect

            # Run the function
            await agent_example.run_infographic_agent()

            # Verify the model was cleaned up
            mock_model.cleanup.assert_called_once()

            # Verify the contract_net was called
            mock_graph_runner.run_contract_net.assert_called_once()


@pytest.mark.asyncio
async def test_run_infographic_agent_fallback_to_debate(
    mock_model,
    mock_graph_runner,
    mock_judge_agent,
    mock_paper_info,
    mock_paper_content
):
    """Test the run_infographic_agent function with fallback to debate strategy."""
    # Configure the contract_net to fail
    mock_graph_runner.run_contract_net.side_effect = Exception("Contract-net failed")

    # Set up all the mocks
    with patch('examples.qwen3_infographic_agent.create_model', return_value=mock_model), \
         patch('examples.qwen3_infographic_agent.setup_agents', return_value=mock_graph_runner), \
         patch('examples.qwen3_infographic_agent.create_judge', return_value=mock_judge_agent), \
         patch('examples.qwen3_infographic_agent.get_hugging_face_top_daily_paper', return_value=mock_paper_info), \
         patch('examples.qwen3_infographic_agent.get_paper_id', return_value="2401.12345"), \
         patch('examples.qwen3_infographic_agent.download_paper', return_value=mock_paper_content), \
         patch('builtins.print'), \
         patch('os.makedirs'), \
         patch('builtins.open', create=True):

        # Mock the regex searches for code extraction
        with patch('examples.qwen3_infographic_agent.re.search') as mock_search:
            # Configure mock_search to return different values for different patterns
            def mock_search_side_effect(pattern, text, flags=0):
                if '```html' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "<html>Mock HTML</html>"
                    return mock_match
                elif '```css' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "body { color: black; }"
                    return mock_match
                elif '```(?:javascript|js)' in pattern:
                    mock_match = MagicMock()
                    mock_match.group.return_value = "const app = () => {};"
                    return mock_match
                return None

            mock_search.side_effect = mock_search_side_effect

            # Run the function
            await agent_example.run_infographic_agent()

            # Verify the model was cleaned up
            mock_model.cleanup.assert_called_once()

            # Verify the debate was called as a fallback
            mock_graph_runner.run_debate.assert_called_once()


@pytest.mark.asyncio
async def test_main():
    """Test the main function."""
    with patch('examples.qwen3_infographic_agent.run_infographic_agent', new_callable=AsyncMock) as mock_run:
        await agent_example.main()
        mock_run.assert_called_once()
