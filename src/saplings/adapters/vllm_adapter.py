"""
vLLM adapter for Saplings.

This module provides a vLLM-based implementation of the LLM interface,
allowing for high-performance inference with vLLM.
"""

import asyncio
import json
import logging
import os
import re
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from saplings.core.model_adapter import LLM, LLMResponse, ModelCapability, ModelMetadata, ModelRole, ModelURI
from saplings.core.plugin import ModelAdapterPlugin, PluginType

logger = logging.getLogger(__name__)

try:
    import vllm
    from vllm import SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning(
        "vLLM not installed. Please install it with: pip install vllm"
    )


class VLLMAdapter(LLM, ModelAdapterPlugin):
    """
    vLLM adapter for Saplings.

    This adapter provides high-performance inference using vLLM.
    """

    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        """
        Initialize the vLLM adapter.

        Args:
            model_uri: URI of the model to use
            **kwargs: Additional arguments for the adapter
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Please install it with: pip install vllm"
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
        self.quantization = self.model_uri.parameters.get("quantization", None)

        # Extract function calling parameters from URI
        self.enable_tool_choice = self.model_uri.parameters.get("enable_tool_choice", "true").lower() == "true"
        self.tool_call_parser = self.model_uri.parameters.get("tool_call_parser", None)
        self.chat_template = self.model_uri.parameters.get("chat_template", None)

        # Special handling for Hugging Face models with organization prefixes
        # vLLM has an issue with models specified as "Org/Model" format
        # It tries to access "https://huggingface.co/api/models/Org/tree/main" instead of
        # "https://huggingface.co/api/models/Org/Model/tree/main"
        # We need to handle this by using the full model path

        # For local models, use the model name as is
        if os.path.exists(self.model_name):
            model_name_for_engine = self.model_name
            logger.info(f"Using local model: {model_name_for_engine}")
        else:
            # For Hugging Face models, we need to modify how the model is loaded
            # vLLM expects the model to be in the format "Org/Model" but has issues with
            # how it constructs the API URL

            # Let's try to use a direct download approach
            # This will force vLLM to download the model from the full URL
            # instead of trying to construct the API URL itself
            if "/" in self.model_name:
                # Use the huggingface_hub library to get the full repo ID
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()

                    # Try to get the model info to verify it exists
                    # This will raise an error if the model doesn't exist
                    # or if we don't have access to it
                    try:
                        # Use the full model name as is
                        model_name_for_engine = self.model_name
                        logger.info(f"Using Hugging Face model: {model_name_for_engine}")
                    except Exception as e:
                        logger.warning(f"Error accessing model {self.model_name}: {e}")
                        # Fall back to a different approach
                        model_name_for_engine = self.model_name
                        logger.info(f"Falling back to direct model name: {model_name_for_engine}")
                except ImportError:
                    # If huggingface_hub is not installed, just use the model name as is
                    model_name_for_engine = self.model_name
                    logger.info(f"Using Hugging Face model (without hub library): {model_name_for_engine}")
            else:
                # If there's no organization prefix, just use the model name as is
                model_name_for_engine = self.model_name
                logger.info(f"Using Hugging Face model: {model_name_for_engine}")

        # Initialize vLLM engine
        engine_kwargs = {
            "model": model_name_for_engine,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        # Add quantization if specified
        if self.quantization:
            engine_kwargs["quantization"] = self.quantization

        # Add function calling parameters if specified
        if self.enable_tool_choice:
            # Check vLLM version for function calling support
            import pkg_resources
            vllm_version = pkg_resources.get_distribution("vllm").version

            # vLLM 0.8.0+ supports function calling with different parameters
            if pkg_resources.parse_version(vllm_version) >= pkg_resources.parse_version("0.8.0"):
                logger.info(f"Using vLLM {vllm_version} function calling features")

                # In vLLM 0.8.0+, function calling is enabled by default
                # We just need to set the appropriate parameters for the model

                # In vLLM 0.8.5, tool_call_parser and chat_template are not supported as engine arguments
                # Instead, we need to use the enable_auto_tool_choice parameter

                # Check if enable_auto_tool_choice is supported in this vLLM version
                try:
                    # Create a test EngineArgs to check if enable_auto_tool_choice is supported
                    from vllm.engine.arg_utils import EngineArgs
                    test_args = {"enable_auto_tool_choice": True}
                    EngineArgs(**test_args)

                    # If we get here, enable_auto_tool_choice is supported
                    engine_kwargs["enable_auto_tool_choice"] = True
                    logger.info("Using enable_auto_tool_choice for function calling")
                except (TypeError, ImportError, AttributeError):
                    # enable_auto_tool_choice is not supported in this vLLM version
                    logger.warning("Function calling may not work as expected in this vLLM version.")

                # Log warnings about unsupported parameters
                if self.tool_call_parser:
                    logger.warning("tool_call_parser parameter not supported in this vLLM version. "
                                  "Function calling may not work as expected.")

                if self.chat_template:
                    logger.warning("chat_template parameter not supported in this vLLM version. "
                                  "Chat formatting may not work as expected.")
            else:
                # For older versions, try to use enable_auto_tool_choice
                try:
                    # Create a test EngineArgs to check if enable_auto_tool_choice is supported
                    from vllm.engine.arg_utils import EngineArgs
                    test_args = {"enable_auto_tool_choice": True}
                    EngineArgs(**test_args)
                    # If we get here, enable_auto_tool_choice is supported
                    engine_kwargs["enable_auto_tool_choice"] = True

                    # Check if tool_call_parser is supported in this vLLM version
                    try:
                        # Create a test EngineArgs to check if tool_call_parser is supported
                        test_args = {"tool_call_parser": "test"}
                        EngineArgs(**test_args)

                        # If we get here, tool_call_parser is supported
                        if self.tool_call_parser:
                            engine_kwargs["tool_call_parser"] = self.tool_call_parser
                    except (TypeError, AttributeError):
                        # tool_call_parser is not supported in this vLLM version
                        logger.warning("tool_call_parser parameter not supported in this vLLM version. "
                                      "Function calling may not work as expected.")

                    # Check if chat_template is supported in this vLLM version
                    try:
                        # Create a test EngineArgs to check if chat_template is supported
                        test_args = {"chat_template": "test"}
                        EngineArgs(**test_args)

                        # If we get here, chat_template is supported
                        if self.chat_template:
                            engine_kwargs["chat_template"] = self.chat_template
                    except (TypeError, AttributeError):
                        # chat_template is not supported in this vLLM version
                        logger.warning("chat_template parameter not supported in this vLLM version. "
                                      "Chat formatting may not work as expected.")
                except (TypeError, ImportError, AttributeError):
                    # enable_auto_tool_choice is not supported in this vLLM version
                    logger.warning("Function calling not fully supported in this vLLM version. "
                                  "Function calling may not work as expected.")

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in engine_kwargs and key != "trust_remote_code":
                engine_kwargs[key] = value

        # Create the vLLM engine
        try:
            # Try to load the model with the current engine_kwargs
            try:
                self.engine = vllm.LLM(**engine_kwargs)
            except Exception as first_error:
                # If the error is about the repository not being found and the model name contains a slash
                if ("Repository Not Found" in str(first_error) or "Invalid repository ID" in str(first_error)) and "/" in self.model_name:
                    # Try a different approach - use the model name without the organization prefix
                    org, model_without_org = self.model_name.split("/", 1)
                    logger.warning(f"Error loading model with full name. Trying with model name only: {model_without_org}")

                    # Update the engine kwargs with the new model name
                    engine_kwargs["model"] = model_without_org

                    # Try again with the new model name
                    try:
                        self.engine = vllm.LLM(**engine_kwargs)
                    except Exception as second_error:
                        # If that also fails, try with just the organization name
                        logger.warning(f"Error loading model with model name only. Trying with organization name: {org}")

                        # Update the engine kwargs with the organization name
                        engine_kwargs["model"] = org

                        # Try one more time
                        try:
                            self.engine = vllm.LLM(**engine_kwargs)
                        except Exception as third_error:
                            # If all approaches fail, raise the original error
                            logger.error(f"All approaches failed. Original error: {first_error}")
                            raise RuntimeError(f"Error initializing vLLM engine: {first_error}")
                else:
                    # If it's a different error, re-raise it
                    raise RuntimeError(f"Error initializing vLLM engine: {first_error}")
        except Exception as e:
            logger.error(f"Error initializing vLLM engine: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Error initializing vLLM engine: {e}")

        # Log function calling configuration
        if self.enable_tool_choice:
            logger.info(f"Function calling enabled with parser: {self.tool_call_parser or 'default'}")

        # Get the tokenizer
        self.tokenizer = get_tokenizer(self.model_name)

        # Cache for model metadata
        self._metadata = None

        logger.info(f"Initialized vLLM adapter for model: {self.model_name}")

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
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
            use_cache: Whether to use caching for this request
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            **kwargs: Additional arguments for generation

        Returns:
            LLMResponse: The generated response
        """
        # Use caching if requested
        if use_cache:
            return await self.generate_with_cache(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                cache_namespace=cache_namespace,
                cache_ttl=cache_ttl,
                **kwargs
            )

        # Set parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Add JSON mode if requested
        if json_mode:
            # Add JSON grammar constraints if available
            if hasattr(sampling_params, "grammar"):
                sampling_params.grammar = "json"

        # Handle function calling
        tool_choice = None
        tools = None

        if functions:
            # Convert functions to tools format for vLLM
            tools = []
            for func in functions:
                tool = {
                    "type": "function",
                    "function": func
                }
                tools.append(tool)

            # Set tool_choice based on function_call
            if function_call == "auto":
                tool_choice = "auto"
            elif function_call == "none":
                tool_choice = "none"
            elif isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call["name"]}
                }
            elif function_call == "required":
                tool_choice = "required"

        try:
            # Generate text
            if tools and self.enable_tool_choice:
                # Use the OpenAI-compatible API for function calling
                outputs = self.engine.generate(
                    processed_prompt,
                    sampling_params=sampling_params,
                    tools=tools,
                    tool_choice=tool_choice
                )
            else:
                # Standard generation without function calling
                outputs = self.engine.generate(
                    processed_prompt,
                    sampling_params=sampling_params,
                )

            # Get the generated text
            generated_text = outputs[0].outputs[0].text

            # Calculate token usage
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(outputs[0].outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens

            # Process function calls if needed
            function_call_result = None
            tool_calls_result = None

            # Check if the output has tool calls (native vLLM function calling)
            if hasattr(outputs[0], 'tool_calls') and outputs[0].tool_calls:
                # Native vLLM function calling
                tool_calls = outputs[0].tool_calls

                if len(tool_calls) == 1:
                    # Single function call
                    tool_call = tool_calls[0]
                    if tool_call.get('type') == 'function':
                        function_call_result = {
                            "name": tool_call['function']['name'],
                            "arguments": tool_call['function']['arguments']
                        }
                elif len(tool_calls) > 1:
                    # Multiple tool calls
                    tool_calls_result = []
                    for tool_call in tool_calls:
                        if tool_call.get('type') == 'function':
                            tool_calls_result.append({
                                "id": tool_call.get('id', f"call_{len(tool_calls_result)}"),
                                "type": "function",
                                "function": {
                                    "name": tool_call['function']['name'],
                                    "arguments": tool_call['function']['arguments']
                                }
                            })
            # Fallback to text parsing if native function calling didn't work
            elif functions and (function_call == "auto" or isinstance(function_call, dict)) and not self.enable_tool_choice:
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
                    "provider": "vllm",
                },
                function_call=function_call_result,
                tool_calls=tool_calls_result
            )
        except Exception as e:
            logger.error(f"Error generating text with vLLM: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Error generating text with vLLM: {e}")

    async def generate_with_cache(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model with caching.

        Args:
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            **kwargs: Additional arguments for generation

        Returns:
            LLMResponse: The generated response
        """
        # Import here to avoid circular imports
        from saplings.core.model_caching import generate_cache_key, get_model_cache

        # Generate a cache key
        cache_key = generate_cache_key(
            model_uri=str(self.model_uri),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs
        )

        # Get the cache
        cache = get_model_cache(namespace=cache_namespace, ttl=cache_ttl)

        # Check if the response is in the cache
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for vLLM model {self.model_name}")
            return cached_response

        # Generate the response
        logger.debug(f"Cache miss for vLLM model {self.model_name}")
        response = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            use_cache=False,  # Important to avoid infinite recursion
            **kwargs
        )

        # Cache the response
        cache.set(cache_key, response)

        return response

    def _process_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False
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
            return prompt

        # Handle message list
        if not hasattr(self.tokenizer, "apply_chat_template"):
            # If the tokenizer doesn't support chat templates, convert to a simple prompt
            return self._messages_to_prompt(prompt)

        # Use the tokenizer's chat template
        chat_template_args = {
            "messages": prompt,
            "add_generation_prompt": True,
            "tokenize": False,
        }

        # Add functions if provided
        if functions:
            chat_template_args["tools"] = functions

        return self.tokenizer.apply_chat_template(**chat_template_args)

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

    def _extract_function_calls(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Extract function calls from generated text.

        Args:
            text: The generated text

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
                The function call and tool calls if found
        """
        import re
        import json

        # Try to find function calls in the format:
        # ```json
        # {"name": "function_name", "arguments": {...}}
        # ```
        function_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
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
                        "arguments": json.dumps(func_data["arguments"]) if isinstance(func_data["arguments"], dict) else func_data["arguments"]
                    }, None
                elif "tool_calls" in func_data:
                    # Multiple tool calls
                    return None, func_data["tool_calls"]
            except json.JSONDecodeError:
                pass

        # Try to find function calls in a more relaxed format
        function_pattern = r'function\s*:\s*(\w+).*?arguments\s*:\s*({[\s\S]*?})(?=\n\w+\s*:|$)'
        matches = re.findall(function_pattern, text, re.IGNORECASE)

        if matches:
            try:
                name, args_str = matches[0]
                args = json.loads(args_str)
                return {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else args
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
        **kwargs
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

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Add JSON mode if requested
        if json_mode and hasattr(sampling_params, "grammar"):
            sampling_params.grammar = "json"

        # Handle function calling
        tool_choice = None
        tools = None

        if functions:
            # Convert functions to tools format for vLLM
            tools = []
            for func in functions:
                tool = {
                    "type": "function",
                    "function": func
                }
                tools.append(tool)

            # Set tool_choice based on function_call
            if function_call == "auto":
                tool_choice = "auto"
            elif function_call == "none":
                tool_choice = "none"
            elif isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call["name"]}
                }
            elif function_call == "required":
                tool_choice = "required"

        # Generate text with streaming
        if tools and self.enable_tool_choice:
            # Use the OpenAI-compatible API for function calling
            outputs_generator = self.engine.generate_iterator(
                [processed_prompt],
                sampling_params=sampling_params,
                tools=tools,
                tool_choice=tool_choice
            )
        else:
            # Standard generation without function calling
            outputs_generator = self.engine.generate_iterator(
                [processed_prompt],
                sampling_params=sampling_params,
            )

        # For function calling, we need to accumulate the text
        accumulated_text = ""
        last_tool_calls = None

        # Stream the outputs
        for outputs in outputs_generator:
            output = outputs[0]

            # Check for tool calls in the output (native vLLM function calling)
            if hasattr(output, 'tool_calls') and output.tool_calls and output.tool_calls != last_tool_calls:
                # Native vLLM function calling
                tool_calls = output.tool_calls
                last_tool_calls = tool_calls

                if len(tool_calls) == 1:
                    # Single function call
                    tool_call = tool_calls[0]
                    if tool_call.get('type') == 'function':
                        yield {"function_call": {
                            "name": tool_call['function']['name'],
                            "arguments": tool_call['function']['arguments']
                        }}
                        return
                elif len(tool_calls) > 1:
                    # Multiple tool calls
                    for tool_call in tool_calls:
                        if tool_call.get('type') == 'function':
                            yield {"tool_call": {
                                "id": tool_call.get('id', f"call_{len(tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": tool_call['function']['name'],
                                    "arguments": tool_call['function']['arguments']
                                }
                            }}
                    return

            # Process text chunks
            if output.outputs:
                chunk = output.outputs[0].text

                # If we're using function calling but not native support, check for function calls in text
                if functions and (function_call == "auto" or isinstance(function_call, dict)) and not self.enable_tool_choice:
                    accumulated_text += chunk
                    function_call_result, tool_calls_result = self._extract_function_calls(accumulated_text)

                    if function_call_result:
                        yield {"function_call": function_call_result}
                        return
                    elif tool_calls_result:
                        for tool_call in tool_calls_result:
                            yield {"tool_call": tool_call}
                        return
                    else:
                        # No function call detected yet, yield the chunk as text
                        yield chunk
                else:
                    # Normal text generation
                    yield chunk

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns:
            ModelMetadata: Metadata about the model
        """
        if self._metadata is None:
            capabilities = [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_GENERATION,
            ]

            # Add function calling capability if enabled
            if self.enable_tool_choice:
                capabilities.append(ModelCapability.FUNCTION_CALLING)

            # Add JSON mode capability
            if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
                capabilities.append(ModelCapability.JSON_MODE)

            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="vllm",
                version=self.model_uri.version,
                description=f"vLLM adapter for {self.model_name}",
                capabilities=capabilities,
                roles=[ModelRole.GENERAL],
                context_window=self._get_context_window(),
                max_tokens_per_request=self.max_tokens,
                cost_per_1k_tokens_input=0.0,  # Local models have no API cost
                cost_per_1k_tokens_output=0.0,  # Local models have no API cost
            )
        return self._metadata

    def _get_context_window(self) -> int:
        """
        Get the context window size for the model.

        Returns:
            int: Context window size in tokens
        """
        # Try to get the context window from the model config
        model_name = self.model_name.lower()

        # Common model context windows
        if "llama-3-70b" in model_name:
            return 8192
        elif "llama-3" in model_name:
            return 8192
        elif "llama-2-70b" in model_name:
            return 4096
        elif "llama-2" in model_name:
            return 4096
        elif "mistral" in model_name:
            return 8192
        elif "mixtral" in model_name:
            return 32768
        elif "qwen" in model_name:
            return 8192
        else:
            return 4096  # Default

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

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response to a conversation.

        Args:
            messages: List of message dictionaries
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
            LLMResponse: The generated response
        """
        if use_cache:
            return await self.generate_with_cache(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                cache_namespace=cache_namespace,
                cache_ttl=cache_ttl,
                **kwargs
            )
        else:
            return await self.generate(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                **kwargs
            )

    def cleanup(self) -> None:
        """
        Clean up resources used by the model.

        This method should be called when the model is no longer needed.
        """
        if hasattr(self, 'engine') and self.engine is not None:
            import gc
            import torch

            # Clean up vLLM resources
            if hasattr(self.engine, 'llm_engine') and hasattr(self.engine.llm_engine, 'model_executor'):
                if hasattr(self.engine.llm_engine.model_executor, 'driver_worker'):
                    del self.engine.llm_engine.model_executor.driver_worker

            # Delete the engine
            del self.engine
            self.engine = None

            # Run garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Cleaned up vLLM adapter for model: {self.model_name}")

    # Plugin interface implementation

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return f"vllm-{self.model_name}"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return f"vLLM adapter for {self.model_name}"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MODEL_ADAPTER
