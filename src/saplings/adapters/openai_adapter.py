"""
OpenAI adapter for Saplings.

This module provides an OpenAI-based implementation of the LLM interface.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from saplings.core.model_adapter import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
    ModelURI,
)
from saplings.core.plugin import ModelAdapterPlugin, PluginType

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Please install it with: pip install openai")


class OpenAIAdapter(LLM, ModelAdapterPlugin):
    """
    OpenAI adapter for Saplings.

    This adapter provides access to OpenAI's models.
    """

    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        """
        Initialize the OpenAI adapter.

        Args:
            model_uri: URI of the model to use
            **kwargs: Additional arguments for the adapter
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Please install it with: pip install openai")

        # Parse the model URI
        if isinstance(model_uri, str):
            self.model_uri = ModelURI.parse(model_uri)
        else:
            self.model_uri = model_uri

        # Extract parameters from URI
        self.model_name = self.model_uri.model_name
        self.temperature = float(self.model_uri.parameters.get("temperature", 0.7))
        self.max_tokens = int(self.model_uri.parameters.get("max_tokens", 1024))

        # Get API key from kwargs, URI parameters, or environment variable
        api_key = kwargs.get("api_key")
        if api_key is None:
            api_key = self.model_uri.parameters.get("api_key")
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        # Get API base from kwargs, URI parameters, or default
        api_base = kwargs.get("api_base")
        if api_base is None:
            api_base = self.model_uri.parameters.get("api_base")
        if api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE")

        # Create the OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base

        # Add organization if provided
        organization = kwargs.get("organization")
        if organization is None:
            organization = self.model_uri.parameters.get("organization")
        if organization is None:
            organization = os.environ.get("OPENAI_ORGANIZATION")
        if organization:
            client_kwargs["organization"] = organization

        # Create the client
        self.client = OpenAI(**client_kwargs)

        # Cache for model metadata
        self._metadata = None

        logger.info(f"Initialized OpenAI adapter for model: {self.model_name}")

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from the model.

        Args:
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Returns:
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

        # Add functions if provided
        if functions:
            completion_args["tools"] = [
                {"type": "function", "function": func} for func in functions
            ]

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

        # Add JSON mode if requested
        if json_mode:
            completion_args["response_format"] = {"type": "json_object"}

        # Create the completion
        response = await asyncio.to_thread(self.client.chat.completions.create, **completion_args)

        # Get token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Process the response
        function_call_result = None
        tool_calls_result = None
        generated_text = None

        # Check if the model returned a function call
        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            tool_calls = response.choices[0].message.tool_calls
            tool_calls_result = []

            for tool_call in tool_calls:
                if tool_call.type == "function":
                    tool_calls_result.append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
        elif (
            hasattr(response.choices[0].message, "function_call")
            and response.choices[0].message.function_call
        ):
            # Legacy function calling format
            function_call_obj = response.choices[0].message.function_call
            function_call_result = {
                "name": function_call_obj.name,
                "arguments": function_call_obj.arguments,
            }
        else:
            # Regular text response
            generated_text = response.choices[0].message.content

        # Create response
        return LLMResponse(
            text=generated_text,
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata={
                "model": self.model_name,
                "provider": "openai",
            },
            function_call=function_call_result,
            tool_calls=tool_calls_result,
        )

    def _process_prompt(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process the prompt into a list of messages.

        Args:
            prompt: The prompt to process

        Returns:
            List[Dict[str, Any]]: List of message dictionaries
        """
        if isinstance(prompt, str):
            # Convert string prompt to a message
            return [{"role": "user", "content": prompt}]

        # Already a list of messages
        return prompt

    async def generate_streaming(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate text from the model with streaming output.

        Args:
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

        # Add functions if provided
        if functions:
            completion_args["tools"] = [
                {"type": "function", "function": func} for func in functions
            ]

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

        # Add JSON mode if requested
        if json_mode:
            completion_args["response_format"] = {"type": "json_object"}

        # Create the completion with streaming
        response = await asyncio.to_thread(self.client.chat.completions.create, **completion_args)

        # For tracking function calls
        current_tool_calls = {}

        # Stream the response
        async for chunk in response:
            # Check for tool calls
            if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    tool_id = tool_call.index

                    # Initialize the tool call if it's new
                    if tool_id not in current_tool_calls:
                        current_tool_calls[tool_id] = {
                            "id": tool_call.id
                            if hasattr(tool_call, "id") and tool_call.id
                            else f"call_{tool_id}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # Update the tool call with new data
                    if hasattr(tool_call, "function"):
                        if hasattr(tool_call.function, "name") and tool_call.function.name:
                            current_tool_calls[tool_id]["function"][
                                "name"
                            ] += tool_call.function.name

                        if (
                            hasattr(tool_call.function, "arguments")
                            and tool_call.function.arguments
                        ):
                            current_tool_calls[tool_id]["function"][
                                "arguments"
                            ] += tool_call.function.arguments

                    # Yield the updated tool call
                    yield {"tool_call": current_tool_calls[tool_id]}

            # Check for function calls (legacy format)
            elif (
                hasattr(chunk.choices[0].delta, "function_call")
                and chunk.choices[0].delta.function_call
            ):
                function_call_delta = chunk.choices[0].delta.function_call

                # Initialize the function call if it's new
                if "function_call" not in current_tool_calls:
                    current_tool_calls["function_call"] = {"name": "", "arguments": ""}

                # Update the function call with new data
                if hasattr(function_call_delta, "name") and function_call_delta.name:
                    current_tool_calls["function_call"]["name"] += function_call_delta.name

                if hasattr(function_call_delta, "arguments") and function_call_delta.arguments:
                    current_tool_calls["function_call"][
                        "arguments"
                    ] += function_call_delta.arguments

                # Yield the updated function call
                yield {"function_call": current_tool_calls["function_call"]}

            # Regular text content
            elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_response(self, response):
        """
        Stream the response from OpenAI.

        Args:
            response: The streaming response from OpenAI

        Yields:
            str: Text chunks as they are generated
        """
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns:
            ModelMetadata: Metadata about the model
        """
        if self._metadata is None:
            # Get model information
            model_info = self._get_model_info()

            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="openai",
                version=self.model_uri.version,
                description=f"OpenAI {self.model_name}",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                ],
                roles=[ModelRole.GENERAL],
                context_window=model_info.get("context_window", 4096),
                max_tokens_per_request=model_info.get("max_tokens", 4096),
                cost_per_1k_tokens_input=model_info.get("cost_input", 0.0),
                cost_per_1k_tokens_output=model_info.get("cost_output", 0.0),
            )
        return self._metadata

    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information
        """
        # Model information for common OpenAI models
        model_info = {
            "gpt-4o": {
                "context_window": 128000,
                "max_tokens": 4096,
                "cost_input": 0.005,
                "cost_output": 0.015,
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "max_tokens": 4096,
                "cost_input": 0.01,
                "cost_output": 0.03,
            },
            "gpt-4": {
                "context_window": 8192,
                "max_tokens": 4096,
                "cost_input": 0.03,
                "cost_output": 0.06,
            },
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "max_tokens": 4096,
                "cost_input": 0.0005,
                "cost_output": 0.0015,
            },
        }

        # Get the model info or use default values
        return model_info.get(
            self.model_name,
            {
                "context_window": 4096,
                "max_tokens": 2048,
                "cost_input": 0.0,
                "cost_output": 0.0,
            },
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: The text to estimate tokens for

        Returns:
            int: Estimated number of tokens
        """
        # Simple estimation based on words
        # In practice, this varies by model and tokenizer
        return len(text.split()) * 1.3

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            float: Estimated cost in USD
        """
        model_info = self._get_model_info()
        cost_input = model_info.get("cost_input", 0.0)
        cost_output = model_info.get("cost_output", 0.0)

        return (prompt_tokens / 1000.0) * cost_input + (completion_tokens / 1000.0) * cost_output

    # Plugin interface implementation

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "openai"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "OpenAI adapter for Saplings"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MODEL_ADAPTER
