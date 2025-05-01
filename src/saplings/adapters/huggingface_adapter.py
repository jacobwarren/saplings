"""
HuggingFace adapter for Saplings.

This module provides a HuggingFace-based implementation of the LLM interface.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

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
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers not installed. Please install it with: pip install transformers torch"
    )


class HuggingFaceAdapter(LLM, ModelAdapterPlugin):
    """
    HuggingFace adapter for Saplings.

    This adapter provides access to HuggingFace's models.
    """

    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        """
        Initialize the HuggingFace adapter.

        Args:
            model_uri: URI of the model to use
            **kwargs: Additional arguments for the adapter
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers not installed. Please install it with: pip install transformers torch"
            )

        # Parse the model URI
        if isinstance(model_uri, str):
            self.model_uri = ModelURI.parse(model_uri)
        else:
            self.model_uri = model_uri

        # Extract parameters from URI
        self.model_name = self.model_uri.model_name
        self.temperature = float(self.model_uri.parameters.get("temperature", 0.7))
        self.max_tokens = int(self.model_uri.parameters.get("max_tokens", 1024))

        # Get device from kwargs, URI parameters, or default
        device = kwargs.get("device")
        if device is None:
            device = self.model_uri.parameters.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Get torch dtype from kwargs, URI parameters, or default
        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None:
            torch_dtype_str = self.model_uri.parameters.get("torch_dtype")
            if torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif torch_dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

        # Create a streamer for streaming generation
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Cache for model metadata
        self._metadata = None

        logger.info(f"Initialized HuggingFace adapter for model: {self.model_name}")

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
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Tokenize the prompt
        inputs = self.tokenizer(processed_prompt, return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        # Generate text
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            **kwargs,
        }

        # Run the generation in a separate thread to avoid blocking
        outputs = await asyncio.to_thread(self.model.generate, **inputs, **generation_kwargs)

        # Decode the generated text
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Count tokens
        completion_tokens = len(outputs[0]) - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens

        # Process function calls if needed
        function_call_result = None
        tool_calls_result = None

        if functions and (function_call == "auto" or isinstance(function_call, dict)):
            # Try to parse function calls from the generated text
            function_call_result, tool_calls_result = self._extract_function_calls(generated_text)

        # Create response
        return LLMResponse(
            text=generated_text if not function_call_result and not tool_calls_result else None,
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata={
                "model": self.model_name,
                "provider": "huggingface",
            },
            function_call=function_call_result,
            tool_calls=tool_calls_result,
        )

    def _process_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Process the prompt to include functions and other features.

        Args:
            prompt: The prompt to process
            functions: List of function definitions
            function_call: Function call control
            json_mode: Whether to force JSON output

        Returns:
            str: The processed prompt
        """
        if isinstance(prompt, str):
            # Simple string prompt
            processed_prompt = prompt
        else:
            # Handle message list
            if hasattr(self.tokenizer, "apply_chat_template"):
                # Use the tokenizer's chat template if available
                chat_template_args = {
                    "messages": prompt,
                    "add_generation_prompt": True,
                    "tokenize": False,
                }
                processed_prompt = self.tokenizer.apply_chat_template(**chat_template_args)
            else:
                # Convert to a simple prompt
                processed_prompt = self._messages_to_prompt(prompt)

        # Add function calling instructions if needed
        if functions:
            # Format functions as a string
            functions_str = json.dumps(functions, indent=2)

            # Add instructions for function calling
            if function_call == "auto":
                function_instructions = (
                    "\nYou have access to the following functions. Use them when appropriate:\n"
                    f"{functions_str}\n\n"
                    "To call a function, respond with a JSON object with 'name' and 'arguments' keys.\n"
                    'Example: {"name": "function_name", "arguments": {"arg1": "value1"}}\n'
                )
            elif isinstance(function_call, dict) and "name" in function_call:
                function_name = function_call["name"]
                function_def = next((f for f in functions if f["name"] == function_name), None)
                if function_def:
                    function_instructions = (
                        f"\nYou must call the function '{function_name}'.\n"
                        f"Function definition: {json.dumps(function_def, indent=2)}\n\n"
                        "Respond with a JSON object with 'name' and 'arguments' keys.\n"
                        f'Example: {{"name": "{function_name}", "arguments": {{"arg1": "value1"}}}}\n'
                    )
                else:
                    function_instructions = ""
            else:
                function_instructions = ""

            processed_prompt += function_instructions

        # Add JSON mode instructions if needed
        if json_mode:
            json_instructions = "\nYou must respond with valid JSON only. Do not include any explanatory text or markdown formatting.\n"
            processed_prompt += json_instructions

        return processed_prompt

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert a list of messages to a simple prompt string.

        Args:
            messages: List of message dictionaries

        Returns:
            str: The prompt string
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Handle content that might be a list of content parts
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = " ".join(text_parts)

            # Format based on role
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role in ["function", "tool"]:
                name = message.get("name", "")
                prompt_parts.append(f"Function {name}: {content}")

        # Add a final assistant prompt
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)

    def _extract_function_calls(
        self, text: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Extract function calls from generated text.

        Args:
            text: The generated text

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
                The function call and tool calls if found
        """
        # Try to find function calls in the format:
        # ```json
        # {"name": "function_name", "arguments": {...}}
        # ```
        function_pattern = r"```(?:json)?\s*({[\s\S]*?})\s*```"
        matches = re.findall(function_pattern, text)

        if matches:
            try:
                # Try to parse the first match as JSON
                func_data = json.loads(matches[0])

                # Check if it has the expected structure
                if "name" in func_data and "arguments" in func_data:
                    # Single function call
                    return {
                        "name": func_data["name"],
                        "arguments": json.dumps(func_data["arguments"])
                        if isinstance(func_data["arguments"], dict)
                        else func_data["arguments"],
                    }, None
                elif "tool_calls" in func_data:
                    # Multiple tool calls
                    return None, func_data["tool_calls"]
            except json.JSONDecodeError:
                pass

        # Try to find function calls in a more relaxed format
        function_pattern = r"function\s*:\s*(\w+).*?arguments\s*:\s*({[\s\S]*?})(?=\n\w+\s*:|$)"
        matches = re.findall(function_pattern, text, re.IGNORECASE)

        if matches:
            try:
                name, args_str = matches[0]
                args = json.loads(args_str)
                return {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                }, None
            except (json.JSONDecodeError, ValueError):
                pass

        return None, None

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
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Tokenize the prompt
        inputs = self.tokenizer(processed_prompt, return_tensors="pt").to(self.device)

        # Set up generation parameters
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": self.streamer,
            **kwargs,
        }

        # Start generation in a separate thread
        thread = asyncio.to_thread(self.model.generate, **generation_kwargs)

        # For function calling, we need to accumulate the text
        accumulated_text = ""

        # Stream the output
        for new_text in self.streamer:
            # If we're using function calling, we need to check for function calls
            if functions and (function_call == "auto" or isinstance(function_call, dict)):
                accumulated_text += new_text
                function_call_result, tool_calls_result = self._extract_function_calls(
                    accumulated_text
                )

                if function_call_result:
                    yield {"function_call": function_call_result}
                    return
                elif tool_calls_result:
                    for tool_call in tool_calls_result:
                        yield {"tool_call": tool_call}
                    return
                else:
                    # No function call detected yet, yield the chunk as text
                    yield new_text
            else:
                # Normal text generation
                yield new_text

        # Wait for the generation to complete
        await thread

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
                provider="huggingface",
                version=self.model_uri.version,
                description=f"HuggingFace {self.model_name}",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                ],
                roles=[ModelRole.GENERAL],
                context_window=model_info.get("context_window", 4096),
                max_tokens_per_request=model_info.get("max_tokens", 2048),
                cost_per_1k_tokens_input=0.0,  # Local models have no API cost
                cost_per_1k_tokens_output=0.0,  # Local models have no API cost
            )
        return self._metadata

    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information
        """
        # Try to get context window from model config
        context_window = 4096  # Default
        max_tokens = 2048  # Default

        try:
            if hasattr(self.model.config, "max_position_embeddings"):
                context_window = self.model.config.max_position_embeddings
            elif hasattr(self.model.config, "n_positions"):
                context_window = self.model.config.n_positions
            elif hasattr(self.model.config, "max_sequence_length"):
                context_window = self.model.config.max_sequence_length

            # Set max_tokens to half the context window by default
            max_tokens = min(context_window // 2, 2048)
        except Exception as e:
            logger.warning(f"Failed to get context window from model config: {e}")

        return {
            "context_window": context_window,
            "max_tokens": max_tokens,
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: The text to estimate tokens for

        Returns:
            int: Estimated number of tokens
        """
        # Use the tokenizer to get the exact token count
        return len(self.tokenizer.encode(text))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            float: Estimated cost in USD
        """
        # Local models have no API cost
        return 0.0

    def cleanup(self) -> None:
        """
        Clean up resources used by the model.

        This method should be called when the model is no longer needed.
        """
        if hasattr(self, "model") and self.model is not None:
            import gc

            # Delete the model
            del self.model
            self.model = None

            # Run garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Cleaned up HuggingFace adapter for model: {self.model_name}")

    # Plugin interface implementation

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "huggingface"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "HuggingFace adapter for Saplings"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MODEL_ADAPTER
