from __future__ import annotations

"""Anthropic adapter for Saplings.

This module provides an implementation of the LLM interface for Anthropic's Claude models.
"""

# Standard library imports
import asyncio  # noqa: E402
import collections.abc
import logging  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

# Local imports
from saplings.core.config_service import config_service  # noqa: E402
from saplings.models._internal.interfaces import (  # noqa: E402
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Import Anthropic if available
ANTHROPIC_AVAILABLE = False
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic not installed. Please install it with: pip install anthropic")
    Anthropic = None

    # Create a mock class for testing - this won't be used in production
    class MockContentBlock:
        """Mock content block for testing."""

        def __init__(self, text=None):
            self.text = text
            self.type = "text"

    class MockMessage:
        """Mock Message class for testing."""

        def __init__(self):
            self.content = [MockContentBlock(text="Mock response")]
            self.usage = MockUsage()
            self.stop_reason = "end_turn"
            self.model = "claude-mock"
            self.id = "msg_mock"
            self.type = "message"
            self.role = "assistant"

    class MockUsage:
        """Mock Usage class for testing."""

        def __init__(self):
            self.input_tokens = 0
            self.output_tokens = 0


class AnthropicAdapter(LLM):
    """
    Anthropic adapter for Saplings.

    This adapter provides access to Anthropic's Claude models.
    """

    def __init__(self, provider: str, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the Anthropic adapter.

        Args:
        ----
            provider: The model provider (e.g., 'anthropic')
            model_name: The model name (e.g., 'claude-3-opus-20240229')
            **kwargs: Additional arguments for the adapter

        """
        if not ANTHROPIC_AVAILABLE:
            msg = "Anthropic not installed. Please install it with: pip install anthropic"
            raise ImportError(msg)

        # Store provider and model name
        self.provider = provider
        self.model_name = model_name

        # Extract parameters from kwargs
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.max_tokens = int(kwargs.get("max_tokens", 1024))

        # Get API key from kwargs or config service
        api_key = kwargs.get("api_key")

        if api_key is None:
            api_key = config_service.get_value("ANTHROPIC_API_KEY")

        # Create the Anthropic client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        # Create the client
        if Anthropic is None:
            msg = "Anthropic not installed. Please install it with: pip install anthropic"
            raise ImportError(msg)
        self.client = Anthropic(**client_kwargs)

        # Cache for model metadata
        self._metadata = None

        logger.info("Initialized Anthropic adapter for model: %s", self.model_name)

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        *,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate text from the model.

        Args:
        ----
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The generated response

        """
        # Set parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        messages = self._process_prompt(prompt)

        # Prepare completion arguments
        completion_args = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        # Add tools/functions if provided
        if functions:
            # Convert to Anthropic's tool format
            tools = [{"type": "function", "function": func} for func in functions]
            completion_args["tools"] = tools

            # Handle tool choice
            if function_call:
                if function_call == "auto":
                    completion_args["tool_choice"] = "auto"
                elif function_call == "none":
                    completion_args["tool_choice"] = "none"
                elif isinstance(function_call, dict) and "name" in function_call:
                    completion_args["tool_choice"] = {
                        "type": "function",
                        "function": {"name": function_call["name"]},
                    }

        # Add system prompt for JSON mode if requested
        if json_mode:
            # Anthropic doesn't have a native JSON mode, so we use a system prompt
            system_prompt = "Always respond with valid JSON. Do not include any explanatory text or markdown formatting."

            # Add or update system message
            has_system = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt + " " + msg.get("content", "")
                    has_system = True
                    break

            if not has_system:
                # Insert system message at the beginning
                messages.insert(0, {"role": "system", "content": system_prompt})

            # Update messages in completion args
            completion_args["messages"] = messages

        # Create the completion
        response = await asyncio.to_thread(self.client.messages.create, **completion_args)

        # Process the response
        function_call_result = None
        tool_calls_result = None
        generated_text = None

        # Check if the model returned a tool call
        # Note: This is a simplified implementation that may need to be updated
        # based on the actual Anthropic API response structure
        tool_calls_result = None

        # Extract content from the response
        if response and hasattr(response, "content") and response.content:
            # Regular text response - handle different content block types
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "text":
                    if hasattr(content_block, "text"):
                        generated_text = content_block.text
                        break

        # Get token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if response and hasattr(response, "usage"):
            prompt_tokens = getattr(response.usage, "input_tokens", 0)
            completion_tokens = getattr(response.usage, "output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

        # Create response
        return LLMResponse(
            text=generated_text,
            provider=self.provider,
            model_name=self.model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            function_call=function_call_result,
            tool_calls=tool_calls_result,
        )

    def _process_prompt(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process the prompt into a list of messages.

        Args:
        ----
            prompt: The prompt to process

        Returns:
        -------
            List[Dict[str, Any]]: List of message dictionaries

        """
        if isinstance(prompt, str):
            # Convert string prompt to a message
            return [{"role": "user", "content": prompt}]

        # Already a list of messages
        return prompt

    async def generate_streaming(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        *,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate text from the model with streaming output.

        Args:
        ----
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            chunk_size: Number of tokens per chunk
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Yields:
        ------
            Union[str, Dict[str, Any]]: Text chunks or function call chunks as they are generated

        """
        # Set parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        messages = self._process_prompt(prompt)

        # Prepare completion arguments
        completion_args = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        # Add tools/functions if provided
        if functions:
            # Convert to Anthropic's tool format
            tools = [{"type": "function", "function": func} for func in functions]
            completion_args["tools"] = tools

            # Handle tool choice
            if function_call:
                if function_call == "auto":
                    completion_args["tool_choice"] = "auto"
                elif function_call == "none":
                    completion_args["tool_choice"] = "none"
                elif isinstance(function_call, dict) and "name" in function_call:
                    completion_args["tool_choice"] = {
                        "type": "function",
                        "function": {"name": function_call["name"]},
                    }

        # Add system prompt for JSON mode if requested
        if json_mode:
            # Anthropic doesn't have a native JSON mode, so we use a system prompt
            system_prompt = "Always respond with valid JSON. Do not include any explanatory text or markdown formatting."

            # Add or update system message
            has_system = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = system_prompt + " " + msg.get("content", "")
                    has_system = True
                    break

            if not has_system:
                # Insert system message at the beginning
                messages.insert(0, {"role": "system", "content": system_prompt})

            # Update messages in completion args
            completion_args["messages"] = messages

        # Create the completion with streaming
        try:
            # Create streaming completion
            stream = await asyncio.to_thread(self.client.messages.create, **completion_args)

            current_text = ""
            async for chunk in self._stream_response(stream):
                current_text += chunk
                yield LLMResponse(
                    text=current_text,
                    provider=self.provider,
                    model_name=self.model_name,
                    usage={
                        "prompt_tokens": 0,  # Updated in final chunk
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    function_call=None,
                    tool_calls=None,
                )

        except Exception as e:
            # Log the error with details and yield a final response
            logger.exception("Error in streaming response: %s", str(e))
            yield LLMResponse(
                text=f"Error in streaming response: {e!s}",
                provider=self.provider,
                model_name=self.model_name,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                function_call=None,
                tool_calls=None,
            )

    async def _stream_response(self, response: Any) -> AsyncGenerator[str, None]:
        """
        Stream the response from Anthropic.

        Args:
        ----
            response: The streaming response from Anthropic

        Yields:
        ------
            str: Text chunks as they are generated

        Raises:
        ------
            ValueError: If the response is not iterable or has an invalid format

        """
        if not isinstance(response, collections.abc.Iterable):
            msg = "Response must be iterable"
            raise ValueError(msg)

        for chunk in response:
            if not chunk:
                continue

            # Handle different response formats
            if chunk and hasattr(chunk, "delta") and chunk.delta:
                # Handle delta format (newer API)
                delta = chunk.delta
                if hasattr(delta, "text") and delta.text:
                    yield delta.text
            elif chunk and hasattr(chunk, "content") and chunk.content:
                # Handle content blocks format
                for block in chunk.content:
                    if hasattr(block, "type") and block.type == "text":
                        if hasattr(block, "text") and block.text:
                            yield block.text

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Metadata about the model

        """
        if self._metadata is None:
            # Get model information
            model_info = self._get_model_info()

            self._metadata = ModelMetadata(
                name=self.model_name,
                provider=self.provider,
                version="1.0",
                description=f"Anthropic {self.model_name}",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                ],
                roles=[ModelRole.GENERAL],
                context_window=model_info.get("context_window", 100000),
                max_tokens_per_request=model_info.get("max_tokens", 4096),
                cost_per_1k_tokens_input=model_info.get("cost_input", 0.0),
                cost_per_1k_tokens_output=model_info.get("cost_output", 0.0),
            )
        return self._metadata

    def _get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns
        -------
            Dict[str, Any]: Model information

        """
        # Model information for common Anthropic models
        model_info = {
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "max_tokens": 4096,
                "cost_input": 0.015,
                "cost_output": 0.075,
            },
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "max_tokens": 4096,
                "cost_input": 0.003,
                "cost_output": 0.015,
            },
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "max_tokens": 4096,
                "cost_input": 0.00025,
                "cost_output": 0.00125,
            },
            "claude-3-5-sonnet-20240620": {
                "context_window": 200000,
                "max_tokens": 4096,
                "cost_input": 0.003,
                "cost_output": 0.015,
            },
        }

        # Get the model info or use default values
        return model_info.get(
            self.model_name,
            {
                "context_window": 100000,
                "max_tokens": 4096,
                "cost_input": 0.0,
                "cost_output": 0.0,
            },
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
        ----
            text: The text to estimate tokens for

        Returns:
        -------
            int: Estimated number of tokens

        Note:
        ----
            This is a more accurate estimation based on Claude's tokenization patterns:
            - Each word is roughly 1.3 tokens
            - Each character in code or URLs is about 0.3 tokens
            - Special characters and spaces are roughly 0.3 tokens
            - Newlines count as 2 tokens

        """
        # Count basic components
        words = len(text.split())
        chars = len(text)
        newlines = text.count("\n")

        # Estimate tokens:
        # - Words: 1.3 tokens per word
        # - Special chars: 0.3 tokens per char
        # - Newlines: 2 tokens each
        estimated_tokens = (
            words * 1.3  # Words
            + chars * 0.3  # Characters
            + newlines * 2.0  # Newlines
        )

        return max(1, int(estimated_tokens))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
        ----
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
        -------
            float: Estimated cost in USD

        """
        model_info = self._get_model_info()
        cost_input = model_info.get("cost_input", 0.0)
        cost_output = model_info.get("cost_output", 0.0)

        return (prompt_tokens / 1000.0) * cost_input + (completion_tokens / 1000.0) * cost_output

    # Plugin interface implementation

    # No longer implementing the Plugin interface
