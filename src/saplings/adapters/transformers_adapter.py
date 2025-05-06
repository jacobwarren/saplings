from __future__ import annotations

"""
Transformers Adapter for Saplings.

This module provides an adapter for using Hugging Face Transformers models directly,
without requiring vLLM or external APIs. This is particularly useful for environments
without Triton support, like Apple Silicon Macs.
"""


import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import torch

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class TransformersAdapter(LLM):
    """
    Adapter for Hugging Face Transformers models.

    This adapter allows using Transformers models directly without vLLM or external APIs.
    It's particularly useful for environments without Triton support.
    """

    def __init__(self, provider: str, model_name: str, **kwargs) -> None:
        """
        Initialize the Transformers adapter.

        Args:
        ----
            provider: Provider of the model (e.g., 'transformers')
            model_name: Name of the model
            **kwargs: Additional parameters for the model

        """
        super().__init__(provider, model_name, **kwargs)

        # Store provider and model name
        self.provider = provider
        self.model_name = model_name

        # Default parameters
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.7)
        self.device = kwargs.get("device", "auto")
        self.torch_dtype = kwargs.get("torch_dtype", "auto")
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.use_fast_tokenizer = kwargs.get("use_fast_tokenizer", True)
        self.trust_remote_code = kwargs.get("trust_remote_code", True)

        # Import transformers here to avoid import errors if not installed
        try:
            import transformers
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            self.transformers = transformers
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            msg = "Transformers not installed. Please install it with: pip install transformers"
            raise ImportError(msg)

        # Determine the device to use
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Check for Apple Silicon MPS
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        # Determine the torch dtype to use
        if self.torch_dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = torch.float16
            elif self.device == "mps":
                # MPS works better with float16 on newer models
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.torch_dtype == "float32":
            self.torch_dtype = torch.float32

        logger.info(f"Using torch dtype: {self.torch_dtype}")

        # Load the tokenizer
        try:
            self.tokenizer = self.AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast_tokenizer,
                trust_remote_code=self.trust_remote_code,
            )
            logger.info(f"Loaded tokenizer for model: {self.model_name}")
        except Exception as e:
            logger.exception(f"Error loading tokenizer: {e}")
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg)

        # Load the model
        try:
            # Prepare quantization parameters
            quantization_config = None
            if self.load_in_4bit:
                try:
                    from transformers.utils.quantization_config import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning(
                        "BitsAndBytesConfig not available. Disabling 4-bit quantization."
                    )
                    self.load_in_4bit = False
            elif self.load_in_8bit:
                try:
                    from transformers.utils.quantization_config import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Using 8-bit quantization")
                except ImportError:
                    logger.warning(
                        "BitsAndBytesConfig not available. Disabling 8-bit quantization."
                    )
                    self.load_in_8bit = False

            # Load the model with the appropriate configuration
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device
                if self.device != "mps"
                else None,  # device_map doesn't work well with MPS
                quantization_config=quantization_config,
                trust_remote_code=self.trust_remote_code,
            )

            # If using MPS, move the model to the device manually
            if self.device == "mps":
                self.model = self.model.to(self.device)

            logger.info(f"Loaded model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            msg = f"Error loading model: {e}"
            raise RuntimeError(msg)

        # Check if the model supports function calling
        self.supports_function_calling = hasattr(self.model, "function_call") or hasattr(
            self.model.config, "tool_use"
        )

        # Check if the model supports chat templates
        self.supports_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        # Get model metadata
        self.context_window = getattr(self.model.config, "max_position_embeddings", 2048)
        logger.info(f"Model context window: {self.context_window}")

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: int | None = 3600,
        attention_mask: np.ndarray | sp.spmatrix | list[dict[str, Any]] | None = None,
        **kwargs,
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
            use_cache: Whether to use caching for this request
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The generated response

        """
        # Process the prompt
        input_text = self._process_prompt(prompt, functions, function_call, json_mode)

        # Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_tokens = inputs.input_ids.shape[1]

        # Apply the attention mask if provided
        if attention_mask is not None:
            try:
                logger.info("Using provided GASA attention mask")

                if isinstance(attention_mask, np.ndarray):
                    # Validate shape matches input
                    if attention_mask.shape != (
                        inputs.input_ids.shape[1],
                        inputs.input_ids.shape[1],
                    ):
                        msg = f"Attention mask shape {attention_mask.shape} does not match input shape {(inputs.input_ids.shape[1], inputs.input_ids.shape[1])}"
                        raise ValueError(msg)
                    # Convert dense numpy array to torch tensor
                    attention_mask_tensor = torch.tensor(attention_mask, device=self.device)
                    inputs["attention_mask"] = attention_mask_tensor

                elif isinstance(attention_mask, sp.spmatrix):
                    # Validate shape matches input
                    if attention_mask.shape != (
                        inputs.input_ids.shape[1],
                        inputs.input_ids.shape[1],
                    ):
                        msg = f"Attention mask shape {attention_mask.shape} does not match input shape {(inputs.input_ids.shape[1], inputs.input_ids.shape[1])}"
                        raise ValueError(msg)
                    # Convert sparse matrix to dense torch tensor
                    csr_mask = sp.csr_matrix(attention_mask)
                    attention_mask_tensor = torch.tensor(csr_mask.toarray(), device=self.device)
                    inputs["attention_mask"] = attention_mask_tensor

                elif (
                    isinstance(attention_mask, list)
                    and attention_mask
                    and isinstance(attention_mask[0], dict)
                ):
                    # Handle block-sparse format
                    if (
                        hasattr(self.model, "supports_block_sparse_attention")
                        and self.model.supports_block_sparse_attention
                    ):
                        inputs["block_sparse_mask"] = attention_mask
                    else:
                        logger.warning(
                            "Model does not support block-sparse attention masks, ignoring provided mask"
                        )

                else:
                    msg = f"Unsupported attention mask format: {type(attention_mask)}"
                    raise ValueError(msg)

            except Exception as e:
                logger.error(f"Error applying attention mask: {e!s}")
                raise ValueError(f"Failed to apply attention mask: {e!s}") from e

        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "do_sample": (temperature or self.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in gen_kwargs:
                gen_kwargs[key] = value

        # Generate the response
        with torch.no_grad():
            # Use the inputs dictionary which now contains our custom attention mask if provided
            outputs = self.model.generate(
                inputs.input_ids, attention_mask=inputs.get("attention_mask", None), **gen_kwargs
            )

        # Decode the output
        output_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Count tokens
        output_tokens = len(outputs[0]) - input_tokens

        # Check for function calls
        function_call_result = None
        tool_calls_result = None

        if functions and (function_call == "auto" or isinstance(function_call, dict)):
            # Try to parse function calls from the output
            function_call_result = self._extract_function_call(output_text, functions)

        # Create the response
        return LLMResponse(
            text=output_text if not function_call_result and not tool_calls_result else None,
            provider=self.provider,
            model_name=self.model_name,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            metadata={
                "model": self.model_name,
                "provider": "transformers",
            },
            function_call=function_call_result,
            tool_calls=tool_calls_result,
        )

    async def generate_streaming(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        chunk_size: int | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
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
        # Process the prompt
        input_text = self._process_prompt(prompt, functions, function_call, json_mode)

        # Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "do_sample": (temperature or self.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in gen_kwargs:
                gen_kwargs[key] = value

        # Generate the response with streaming
        try:
            import threading

            from transformers.generation.streamers import TextIteratorStreamer

            # Create synchronization primitives
            generation_done = asyncio.Event()
            generation_error = None

            def generate_in_thread():
                nonlocal generation_error
                try:
                    # Run generation in thread
                    self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_kwargs,
                    )
                except Exception as e:
                    generation_error = e
                finally:
                    # Signal completion
                    asyncio.get_event_loop().call_soon_threadsafe(generation_done.set)

            # Create streamer with skip_prompt to avoid duplicating input text
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_special_tokens=True, skip_prompt=True
            )
            gen_kwargs["streamer"] = streamer

            # Start generation in separate thread
            thread = threading.Thread(target=generate_in_thread)
            thread.start()

            try:
                # Yield chunks as they are generated
                for chunk in streamer:
                    if generation_error:
                        raise generation_error
                    yield chunk

                # Wait for generation to complete
                await generation_done.wait()
                if generation_error:
                    raise generation_error

            finally:
                # Ensure thread is cleaned up
                thread.join(timeout=1.0)
                if thread.is_alive():
                    logger.warning("Generation thread did not complete in time")
        except ImportError:
            # Fallback to non-streaming if TextIteratorStreamer is not available
            logger.warning(
                "TextIteratorStreamer not available. Falling back to non-streaming generation."
            )
            response = await self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                **kwargs,
            )

            if response.text:
                yield response.text
            elif response.function_call:
                yield {"function_call": response.function_call}
            elif response.tool_calls:
                for tool_call in response.tool_calls:
                    yield {"tool_call": tool_call}

    def _process_prompt(
        self,
        prompt: str | list[dict[str, Any]],
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Process the prompt to include functions and other features.

        Args:
        ----
            prompt: The prompt to process
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
            json_mode: Whether to force the model to output valid JSON

        Returns:
        -------
            str: The processed prompt

        """
        # If the prompt is a string, use it directly
        if isinstance(prompt, str):
            return prompt

        # If the prompt is a list of messages, use the chat template if available
        if self.supports_chat_template:
            # Add function definitions to the messages if needed
            messages = list(prompt)  # Make a copy to avoid modifying the original

            if functions:
                # Add function definitions to the messages
                # This depends on the model's chat template format
                if hasattr(self.tokenizer, "apply_chat_template"):
                    # Use the tokenizer's chat template
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                # No functions, just use the chat template
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        # Fallback to a simple format if chat template is not available
        formatted_prompt = ""
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            else:
                formatted_prompt += f"{role.capitalize()}: {content}\n\n"

        # Add a final assistant prompt
        formatted_prompt += "Assistant: "

        return formatted_prompt

    def _extract_function_call(
        self, text: str, functions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Extract function calls from the generated text.

        Args:
        ----
            text: The generated text
            functions: List of function definitions

        Returns:
        -------
            Optional[Dict[str, Any]]: The extracted function call, if any

        """
        # Try different function call formats
        function_call = self._extract_parentheses_function_call(
            text, functions
        ) or self._extract_json_function_call(text, functions)
        return function_call

    def _extract_parentheses_function_call(
        self, text: str, functions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Extract function calls in the format: function_name(arg1=value1, arg2=value2)"""
        # More precise pattern that requires word boundaries
        function_call_pattern = r"\b(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        match = re.search(function_call_pattern, text)

        if not match:
            return None

        function_name = match.group(1)
        args_str = match.group(2)

        # Verify function name
        function_names = [f["name"] for f in functions]
        if function_name not in function_names:
            return None

        try:
            args_dict = self._parse_function_arguments(args_str)
            return {"name": function_name, "arguments": json.dumps(args_dict)}
        except Exception as e:
            logger.warning(f"Error parsing function call arguments: {e}")
            return None

    def _parse_function_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse function arguments string into a dictionary."""
        args_dict = {}
        if not args_str.strip():
            return args_dict

        args_parts = self._split_args_handling_nesting(args_str)

        for part in args_parts:
            if "=" not in part:
                continue

            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse as JSON, fallback to string
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                # Remove any surrounding quotes
                value = value.strip("\"'")

            args_dict[key] = value

        return args_dict

    def _split_args_handling_nesting(self, args_str: str) -> list[str]:
        """Split arguments string handling nested structures."""
        args_parts = []
        current_part = ""
        nesting_level = 0

        for char in args_str:
            if char == "," and nesting_level == 0:
                if current_part.strip():
                    args_parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
                if char in "({[":
                    nesting_level += 1
                elif char in ")}]":
                    nesting_level -= 1

        if current_part.strip():
            args_parts.append(current_part.strip())

        return args_parts

    def _extract_json_function_call(
        self, text: str, functions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Extract function calls in JSON format."""
        # Look for JSON object containing function call
        try:
            # Find the first occurrence of a JSON object
            start = text.find("{")
            if start == -1:
                return None

            # Track nesting to find the matching closing brace
            nesting = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    nesting += 1
                elif text[i] == "}":
                    nesting -= 1
                    if nesting == 0:
                        # Found complete JSON object
                        json_str = text[start : i + 1]
                        json_obj = json.loads(json_str)

                        if not isinstance(json_obj, dict):
                            return None

                        # Check for function call format
                        if "function" in json_obj and isinstance(json_obj["function"], dict):
                            function_name = json_obj["function"].get("name")
                            if not function_name:
                                return None

                            # Verify function name
                            function_names = [f["name"] for f in functions]
                            if function_name not in function_names:
                                return None

                            # Extract arguments
                            arguments = json_obj.get("arguments", {})
                            arguments_str = (
                                arguments if isinstance(arguments, str) else json.dumps(arguments)
                            )

                            return {"name": function_name, "arguments": arguments_str}

            return None
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Error parsing JSON function call: {e}")
            return None

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Metadata about the model

        """
        return ModelMetadata(
            name=self.model_name,
            provider=self.provider,
            version="latest",
            description=f"Transformers model {self.model_name}",
            context_window=self.context_window,
            max_tokens_per_request=self.max_tokens,
            roles=[ModelRole.GENERAL],
            cost_per_1k_tokens_input=0.0,
            cost_per_1k_tokens_output=0.0,
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

        """
        # Use the tokenizer to get the exact token count
        return len(self.tokenizer.encode(text))

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
        # Local models have no API cost
        return 0.0
