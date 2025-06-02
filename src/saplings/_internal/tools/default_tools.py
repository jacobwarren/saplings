from __future__ import annotations

"""
Default tools for Saplings.

This module provides a set of default tools that can be used by agents.
These tools provide common functionality like Python code execution,
web search, webpage visiting, and more.
"""


import re
import sys
from typing import Any

from saplings._internal.tools.base import Tool
from saplings.core.config_service import config_service

# Type checking imports


class PythonInterpreterTool(Tool):
    """
    Tool for evaluating Python code.

    This tool provides a sandboxed environment for executing Python code,
    with restrictions on which modules can be imported.
    """

    # Base built-in modules that are safe to import
    BASE_BUILTIN_MODULES = {
        "math",
        "random",
        "re",
        "json",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "uuid",
        "copy",
        "string",
        "time",
        "calendar",
        "fractions",
        "decimal",
        "bisect",
        "heapq",
        "array",
        "enum",
        "dataclasses",
    }

    def __init__(
        self,
        authorized_imports: list[str] | None = None,
        name: str = "python_interpreter",
        description: str = "Evaluates Python code in a sandboxed environment.",
    ) -> None:
        """
        Initialize the Python interpreter tool.

        Args:
        ----
            authorized_imports: Additional modules that can be imported
            name: Name of the tool
            description: Description of the tool

        """
        # Set up authorized imports
        if authorized_imports is None:
            self.authorized_imports = list(self.BASE_BUILTIN_MODULES)
        else:
            self.authorized_imports = list(set(self.BASE_BUILTIN_MODULES) | set(authorized_imports))

        # Create a more detailed description with authorized imports
        detailed_description = (
            f"{description} All variables used must be defined in the same snippet. "
            f"This code can only import the following Python libraries: {self.authorized_imports}."
        )

        # Define the function that will execute the code
        def execute_python(code: str) -> str:
            """
            Execute Python code in a sandboxed environment.

            Args:
            ----
                code: Python code to execute

            Returns:
            -------
                String representation of the result and any printed output

            """
            # Create a dictionary to store the execution state
            state = {"_print_outputs": ""}

            # Redirect stdout to capture print statements
            original_stdout = sys.stdout
            from io import StringIO

            captured_output = StringIO()
            sys.stdout = captured_output

            result = None
            try:
                # Compile the code to check for syntax errors
                compiled_code = compile(code, "<string>", "exec")

                # Check for unauthorized imports
                for node in compiled_code.co_names:
                    if node in {"import", "__import__"}:
                        # We'll do a more detailed check during execution
                        pass

                # Execute the code
                exec_globals = {"__builtins__": __builtins__}
                exec_locals = {}

                # Add a custom __import__ function to restrict imports
                def restricted_import(name: str, *args, **kwargs):
                    if name not in self.authorized_imports:
                        msg = f"Import of '{name}' is not allowed. Authorized imports: {self.authorized_imports}"
                        raise ImportError(msg)
                    return __import__(name, *args, **kwargs)

                exec_globals["__import__"] = restricted_import

                # Execute the code
                exec(compiled_code, exec_globals, exec_locals)

                # Get the last expression's value if it exists
                if "_" in exec_locals:
                    result = exec_locals["_"]
                else:
                    # Try to find a reasonable result from the locals
                    for var_name, var_value in exec_locals.items():
                        if not var_name.startswith("_") and var_name != "builtins":
                            result = var_value
                            break
            except Exception as e:
                result = f"Error: {e!s}"
            finally:
                # Restore stdout
                state["_print_outputs"] = captured_output.getvalue()
                sys.stdout = original_stdout

            # Return the result and any printed output
            return f"Stdout:\n{state['_print_outputs']}\nOutput: {result!s}"

        # Initialize the Tool with our execute_python function
        super().__init__(func=execute_python, name=name, description=detailed_description)

        # Update parameter information
        self.parameters = {
            "code": {
                "type": "string",
                "description": (
                    "The Python code to execute. All variables must be defined in this snippet. "
                    f"Only the following modules can be imported: {self.authorized_imports}"
                ),
                "required": True,
            }
        }


class FinalAnswerTool(Tool):
    """Tool for providing a final answer to a problem."""

    def __init__(
        self,
        name: str = "final_answer",
        description: str = "Provides a final answer to the given problem.",
    ) -> None:
        """
        Initialize the final answer tool.

        Args:
        ----
            name: Name of the tool
            description: Description of the tool

        """

        def final_answer(answer: Any) -> Any:
            """
            Provide a final answer to a problem.

            Args:
            ----
                answer: The final answer

            Returns:
            -------
                The final answer

            """
            return answer

        super().__init__(func=final_answer, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "answer": {
                "type": "string",
                "description": "The final answer to the problem",
                "required": True,
            }
        }


class UserInputTool(Tool):
    """Tool for getting input from the user."""

    def __init__(
        self,
        name: str = "user_input",
        description: str = "Asks for user's input on a specific question.",
    ) -> None:
        """
        Initialize the user input tool.

        Args:
        ----
            name: Name of the tool
            description: Description of the tool

        """

        def user_input(question: str) -> str:
            """
            Get input from the user.

            Args:
            ----
                question: The question to ask the user

            Returns:
            -------
                The user's input

            """
            return input(f"{question} => Type your answer here: ")

        super().__init__(func=user_input, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
                "required": True,
            }
        }


class DuckDuckGoSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo."""

    def __init__(
        self,
        max_results: int = 5,
        name: str = "web_search_duckduckgo",
        description: str = "Searches the web using DuckDuckGo and returns the top search results.",
    ) -> None:
        """
        Initialize the DuckDuckGo search tool.

        Args:
        ----
            max_results: Maximum number of results to return
            name: Name of the tool
            description: Description of the tool

        """
        self.max_results = max_results

        def duckduckgo_search(query: str) -> str:
            """
            Search the web using DuckDuckGo.

            Args:
            ----
                query: The search query

            Returns:
            -------
                Formatted search results

            """
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                msg = (
                    "You must install package `duckduckgo_search` to use this tool. "
                    "Run `pip install duckduckgo-search`."
                )
                raise ImportError(msg)

            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            if not results:
                return "No results found. Try a less restrictive/shorter query."

            formatted_results = [
                f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
            ]

            return "## Search Results\n\n" + "\n\n".join(formatted_results)

        super().__init__(func=duckduckgo_search, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True,
            }
        }


class GoogleSearchTool(Tool):
    """Tool for searching the web using Google."""

    def __init__(
        self,
        provider: str = "serpapi",
        name: str = "web_search",
        description: str = "Performs a Google web search and returns the top search results.",
    ) -> None:
        """
        Initialize the Google search tool.

        Args:
        ----
            provider: The search provider to use ('serpapi' or 'serper')
            name: Name of the tool
            description: Description of the tool

        """
        self.provider = provider

        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"

        self.api_key = config_service.get_value(api_key_env_name)
        if self.api_key is None:
            msg = f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables."
            raise ValueError(msg)

        def google_search(query: str) -> str:
            """
            Search the web using Google.

            Args:
            ----
                query: The search query

            Returns:
            -------
                Formatted search results

            """
            try:
                import requests
            except ImportError:
                msg = (
                    "You must install package `requests` to use this tool. "
                    "Run `pip install requests`."
                )
                raise ImportError(msg)

            if self.provider == "serpapi":
                url = "https://serpapi.com/search"
                params = {
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                }
            else:  # serper
                url = "https://google.serper.dev/search"
                params = None
                headers = {
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                }
                payload = {"q": query}

            try:
                if self.provider == "serpapi":
                    response = requests.get(url, params=params)
                else:  # serper
                    response = requests.post(url, headers=headers, json=payload)

                response.raise_for_status()
                results = response.json()

                if self.organic_key not in results:
                    return "No results found. Try a less restrictive/shorter query."

                organic_results = results[self.organic_key]

                if not organic_results:
                    return "No results found. Try a less restrictive/shorter query."

                formatted_results = []
                for result in organic_results[:5]:
                    if self.provider == "serpapi":
                        title = result.get("title", "No title")
                        link = result.get("link", "No link")
                        snippet = result.get("snippet", "No description")
                    else:  # serper
                        title = result.get("title", "No title")
                        link = result.get("link", "No link")
                        snippet = result.get("snippet", "No description")

                    formatted_results.append(f"[{title}]({link})\n{snippet}")

                return "## Search Results\n\n" + "\n\n".join(formatted_results)
            except requests.exceptions.RequestException as e:
                return f"Error performing search: {e!s}"

        super().__init__(func=google_search, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True,
            }
        }


class VisitWebpageTool(Tool):
    """Tool for visiting a webpage and reading its content."""

    def __init__(
        self,
        max_output_length: int = 40000,
        name: str = "visit_webpage",
        description: str = "Visits a webpage at the given URL and reads its content as markdown.",
    ) -> None:
        """
        Initialize the visit webpage tool.

        Args:
        ----
            max_output_length: Maximum length of the output
            name: Name of the tool
            description: Description of the tool

        """
        self.max_output_length = max_output_length

        def visit_webpage(url: str) -> str:
            """
            Visit a webpage and read its content.

            Args:
            ----
                url: URL of the webpage to visit

            Returns:
            -------
                Markdown representation of the webpage content

            """
            try:
                import requests
                from requests.exceptions import RequestException

                # Import markdownify
                try:
                    from markdownify import (  # type: ignore[import-not-found]
                        markdownify as md_convert,  # type: ignore[import-not-found]
                    )
                except ImportError:
                    msg = (
                        "You must install the `markdownify` package to use this tool. "
                        "Run `pip install saplings[tools]`."
                    )
                    raise ImportError(msg)
            except ImportError:
                msg = (
                    "You must install required packages to use this tool. "
                    "Run `pip install saplings[tools]`."
                )
                raise ImportError(msg)

            try:
                # Add a user agent to avoid being blocked
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }

                # Make the request
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                # Convert HTML to markdown
                html_content = response.text
                markdown_content = md_convert(html_content)

                # Truncate if too long
                if len(markdown_content) > self.max_output_length:
                    markdown_content = markdown_content[: self.max_output_length] + "...[truncated]"

                # Clean up the markdown
                # Remove excessive newlines
                markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

                return markdown_content
            except requests.exceptions.Timeout:
                return "The request timed out. Please try again later or check the URL."
            except RequestException as e:
                return f"Error fetching the webpage: {e!s}"
            except Exception as e:
                return f"An unexpected error occurred: {e!s}"

        super().__init__(func=visit_webpage, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to visit",
                "required": True,
            }
        }


class WikipediaSearchTool(Tool):
    """Tool for searching Wikipedia and returning article content."""

    def __init__(
        self,
        user_agent: str = "Saplings (https://github.com/yourusername/saplings)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
        name: str = "wikipedia_search",
        description: str = "Searches Wikipedia and returns a summary or full text of the given topic.",
    ) -> None:
        """
        Initialize the Wikipedia search tool.

        Args:
        ----
            user_agent: User agent string for Wikipedia API
            language: Language code for Wikipedia
            content_type: Type of content to return ('summary' or 'text')
            extract_format: Format of the extract ('WIKI' or 'HTML')
            name: Name of the tool
            description: Description of the tool

        """
        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type
        self.extract_format = extract_format

        def wikipedia_search(query: str) -> str:
            """
            Search Wikipedia and return article content.

            Args:
            ----
                query: The topic to search on Wikipedia

            Returns:
            -------
                Formatted Wikipedia content

            """
            try:
                import wikipediaapi
            except ImportError:
                msg = (
                    "You must install the `wikipedia-api` package to use this tool. "
                    "Run `pip install saplings[tools]`."
                )
                raise ImportError(msg)

            wiki = wikipediaapi.Wikipedia(
                user_agent=self.user_agent,
                language=self.language,
                extract_format=self.extract_format,
            )

            try:
                page = wiki.page(query)

                if not page.exists():
                    return f"No Wikipedia page found for '{query}'. Try a different query."

                title = page.title
                url = page.fullurl

                if self.content_type == "summary":
                    text = page.summary
                elif self.content_type == "text":
                    text = page.text
                else:
                    return "Invalid `content_type`. Use either 'summary' or 'text'."

                return f"**Wikipedia Page:** {title}\n\n**Content:** {text}\n\n**Read more:** {url}"
            except Exception as e:
                return f"Error fetching Wikipedia content: {e!s}"

        super().__init__(func=wikipedia_search, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "query": {
                "type": "string",
                "description": "The topic to search on Wikipedia",
                "required": True,
            }
        }


class SpeechToTextTool(Tool):
    """Tool for transcribing speech to text."""

    def __init__(
        self,
        name: str = "transcribe",
        description: str = "Transcribes speech from an audio file to text.",
    ) -> None:
        """
        Initialize the speech-to-text tool.

        Args:
        ----
            name: Name of the tool
            description: Description of the tool

        """

        def transcribe_audio(audio_path: str) -> str:
            """
            Transcribe speech from an audio file to text.

            Args:
            ----
                audio_path: Path to the audio file

            Returns:
            -------
                Transcribed text

            """
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            except ImportError:
                msg = (
                    "You must install the required packages to use this tool. "
                    "Run `pip install saplings[speech]`."
                )
                raise ImportError(msg)

            try:
                # Check if audio file exists
                import os

                if not os.path.exists(audio_path):
                    return f"Audio file not found: {audio_path}"

                # Load the model and processor
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                model_id = "openai/whisper-base"

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                )
                model.to(device)

                processor = AutoProcessor.from_pretrained(model_id)

                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                )

                # Transcribe the audio
                result = pipe(audio_path)
                return result["text"]
            except Exception as e:
                return f"Error transcribing audio: {e!s}"

        super().__init__(func=transcribe_audio, name=name, description=description)

        # Update parameter information
        self.parameters = {
            "audio_path": {
                "type": "string",
                "description": "Path to the audio file to transcribe",
                "required": True,
            }
        }


# Dictionary mapping tool names to tool classes
TOOL_MAPPING = {
    "python_interpreter": PythonInterpreterTool,
    "final_answer": FinalAnswerTool,
    "user_input": UserInputTool,
    "web_search_duckduckgo": DuckDuckGoSearchTool,
    "web_search_google": GoogleSearchTool,
    "visit_webpage": VisitWebpageTool,
    "wikipedia_search": WikipediaSearchTool,
    "transcriber": SpeechToTextTool,
}


def get_default_tool(tool_name: str, **kwargs) -> Tool:
    """
    Get a default tool by name.

    Args:
    ----
        tool_name: Name of the tool to get
        **kwargs: Additional arguments to pass to the tool constructor

    Returns:
    -------
        Tool: The requested tool

    Raises:
    ------
        ValueError: If the tool name is not recognized

    """
    if tool_name not in TOOL_MAPPING:
        msg = f"Unknown tool: {tool_name}. Available tools: {list(TOOL_MAPPING.keys())}"
        raise ValueError(msg)

    tool_class = TOOL_MAPPING[tool_name]
    return tool_class(**kwargs)


def get_all_default_tools():
    """
    Get all default tools.

    Returns
    -------
        Dict[str, Tool]: Dictionary mapping tool names to tool instances

    """
    return {name: tool_class() for name, tool_class in TOOL_MAPPING.items()}
