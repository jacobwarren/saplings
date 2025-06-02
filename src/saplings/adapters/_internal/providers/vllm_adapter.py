from __future__ import annotations

"""
vLLM adapter for Saplings.

This module provides a vLLM-based implementation of the LLM interface,
allowing for high-performance inference with vLLM.
"""


import json
import logging
import os
import re
import traceback
from typing import TYPE_CHECKING, Any

from saplings.models._internal.interfaces import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Use centralized lazy import system
from saplings._internal.optional_deps import OPTIONAL_DEPENDENCIES

VLLM_AVAILABLE = OPTIONAL_DEPENDENCIES["vllm"].available


def _get_vllm_imports():
    """Lazy import vLLM components using centralized system."""
    vllm_module = OPTIONAL_DEPENDENCIES["vllm"].require()

    from vllm import SamplingParams  # type: ignore
    from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

    # Log vLLM version
    logger.debug(f"vLLM version: {getattr(vllm_module, '__version__', 'unknown')}")

    return vllm_module, SamplingParams, get_tokenizer


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer
    from vllm import LLM as VLLM_LLM


class VLLMAdapter(LLM):
    """
    vLLM adapter for Saplings.

    This adapter provides high-performance inference using vLLM.
    """

    tokenizer: Optional["PreTrainedTokenizer"]
    model: Optional["PreTrainedModel"]
    engine: Optional["VLLM_LLM"]

    def __init__(self, provider: str, model_name: str, **kwargs) -> None:
        """
        Initialize the vLLM adapter.

        Args:
        ----
            provider: The model provider (e.g., 'vllm')
            model_name: The model name
            **kwargs: Additional arguments for the adapter

        """
        if not VLLM_AVAILABLE:
            msg = "vLLM not installed. Please install it with: pip install vllm"
            raise ImportError(msg)

        # Store provider and model information
        self.provider = provider
        self.model_name = model_name

        # Extract parameters from kwargs
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.max_tokens = int(kwargs.get("max_tokens", 1024))
        self.quantization = kwargs.get("quantization")

        # Extract function calling parameters
        self.enable_tool_choice = kwargs.get("enable_tool_choice", True)
        if isinstance(self.enable_tool_choice, str):
            self.enable_tool_choice = self.enable_tool_choice.lower() == "true"

        self.tool_call_parser = kwargs.get("tool_call_parser")
        self.chat_template = kwargs.get("chat_template")

        # Special handling for Hugging Face models with organization prefixes
        # vLLM has an issue with models specified as "Org/Model" format
        # It tries to access "https://huggingface.co/api/models/Org/tree/main" instead of
        # "https://huggingface.co/api/models/Org/Model/tree/main"
        # We need to handle this by using the full model path

        # For local models, use the model name as is
        if os.path.exists(self.model_name):
            model_name_for_engine = self.model_name
            logger.info("Using local model: %s", model_name_for_engine)
        # For Hugging Face models, we need to modify how the model is loaded
        # vLLM expects the model to be in the format "Org/Model" but has issues with
        # how it constructs the API URL

        # Let's try to use a direct download approach
        # This will force vLLM to download the model from the full URL
        # instead of trying to construct the API URL itself
        elif "/" in self.model_name:
            # Use the huggingface_hub library to get the full repo ID
            try:
                from huggingface_hub import HfApi

                HfApi()

                # Try to get the model info to verify it exists
                # This will raise an error if the model doesn't exist
                # or if we don't have access to it
                try:
                    # Use the full model name as is
                    model_name_for_engine = self.model_name
                    logger.info("Using Hugging Face model: %s", model_name_for_engine)
                except Exception as e:
                    logger.warning("Error accessing model %s: %s", self.model_name, str(e))
                    # Fall back to a different approach
                    model_name_for_engine = self.model_name
                    logger.info("Falling back to direct model name: %s", model_name_for_engine)
            except ImportError:
                # If huggingface_hub is not installed, just use the model name as is
                model_name_for_engine = self.model_name
                logger.info(
                    "Using Hugging Face model (without hub library): %s", model_name_for_engine
                )
        else:
            # If there's no organization prefix, just use the model name as is
            model_name_for_engine = self.model_name
            logger.info("Using Hugging Face model: %s", model_name_for_engine)

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
                logger.info("Using vLLM %s function calling features", vllm_version)

                # In vLLM 0.8.0+, function calling is enabled by default
                # We just need to set the appropriate parameters for the model

                # In vLLM 0.8.5, tool_call_parser and chat_template are not supported as engine arguments
                # Instead, we need to use the enable_auto_tool_choice parameter

                # Check if enable_auto_tool_choice or enable_tool_choice is supported in this vLLM version
                try:
                    # Create a test EngineArgs to check if enable_auto_tool_choice is supported
                    from vllm.engine.arg_utils import EngineArgs  # type: ignore

                    # Check if enable_auto_tool_choice is supported
                    try:
                        # Instead of directly creating EngineArgs, check if the parameter exists in the class
                        import inspect

                        engine_args_params = inspect.signature(EngineArgs.__init__).parameters
                        if "enable_auto_tool_choice" in engine_args_params:
                            engine_kwargs["enable_auto_tool_choice"] = True
                            logger.info("Using enable_auto_tool_choice for function calling")
                        else:
                            logger.warning(
                                "enable_auto_tool_choice parameter not supported in this vLLM version. "
                                "Function calling may not work as expected."
                            )
                    except (TypeError, AttributeError):
                        # If there's an error checking parameters, don't add it to engine_kwargs
                        logger.warning(
                            "enable_auto_tool_choice parameter not supported in this vLLM version. "
                            "Function calling may not work as expected."
                        )
                except (ImportError, AttributeError):
                    # EngineArgs is not available or has a different structure
                    logger.warning(
                        "Function calling may not work as expected in this vLLM version."
                    )

                # Log warnings about unsupported parameters
                if self.tool_call_parser:
                    logger.warning(
                        "tool_call_parser parameter not supported in this vLLM version. "
                        "Function calling may not work as expected."
                    )

                if self.chat_template:
                    logger.warning(
                        "chat_template parameter not supported in this vLLM version. "
                        "Chat formatting may not work as expected."
                    )
            else:
                # For older versions, try to use enable_auto_tool_choice
                try:
                    # Create a test EngineArgs to check if enable_auto_tool_choice is supported
                    from vllm.engine.arg_utils import EngineArgs  # type: ignore

                    # Check if enable_auto_tool_choice is supported
                    try:
                        # Instead of directly creating EngineArgs, check if the parameter exists in the class
                        import inspect

                        engine_args_params = inspect.signature(EngineArgs.__init__).parameters
                        if "enable_auto_tool_choice" in engine_args_params:
                            engine_kwargs["enable_auto_tool_choice"] = True
                        else:
                            logger.warning(
                                "enable_auto_tool_choice parameter not supported in this vLLM version. "
                                "Function calling may not work as expected."
                            )
                    except (TypeError, AttributeError):
                        # If there's an error checking parameters, don't add it to engine_kwargs
                        logger.warning(
                            "enable_auto_tool_choice parameter not supported in this vLLM version. "
                            "Function calling may not work as expected."
                        )

                    # Check if tool_call_parser is supported in this vLLM version
                    try:
                        # Check if the parameter exists in the class
                        import inspect

                        engine_args_params = inspect.signature(EngineArgs.__init__).parameters
                        if "tool_call_parser" in engine_args_params and self.tool_call_parser:
                            engine_kwargs["tool_call_parser"] = self.tool_call_parser
                        elif self.tool_call_parser:
                            logger.warning(
                                "tool_call_parser parameter not supported in this vLLM version. "
                                "Function calling may not work as expected."
                            )
                    except (TypeError, AttributeError):
                        # Error checking parameters
                        logger.warning(
                            "tool_call_parser parameter not supported in this vLLM version. "
                            "Function calling may not work as expected."
                        )

                    # Check if chat_template is supported in this vLLM version
                    try:
                        # Check if the parameter exists in the class
                        import inspect

                        engine_args_params = inspect.signature(EngineArgs.__init__).parameters
                        if "chat_template" in engine_args_params and self.chat_template:
                            engine_kwargs["chat_template"] = self.chat_template
                        elif self.chat_template:
                            logger.warning(
                                "chat_template parameter not supported in this vLLM version. "
                                "Chat formatting may not work as expected."
                            )
                    except (TypeError, AttributeError):
                        # Error checking parameters
                        logger.warning(
                            "chat_template parameter not supported in this vLLM version. "
                            "Chat formatting may not work as expected."
                        )
                except (TypeError, ImportError, AttributeError):
                    # enable_auto_tool_choice is not supported in this vLLM version
                    logger.warning(
                        "Function calling not fully supported in this vLLM version. "
                        "Function calling may not work as expected."
                    )

        # Filter out generation parameters that shouldn't be passed to the engine
        generation_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
        ]

        # Parameters specific to Saplings that shouldn't be passed to vLLM
        saplings_params = ["gasa_cache", "use_cache", "cache_namespace", "cache_ttl", "trace_id"]

        # Add any additional kwargs, filtering out generation parameters and Saplings-specific parameters
        for key, value in kwargs.items():
            if (
                key not in engine_kwargs
                and key != "trust_remote_code"
                and key not in generation_params
                and key not in saplings_params
            ):
                engine_kwargs[key] = value

        # Initialize these before creating the engine
        # Cache for model metadata
        self._metadata = None
        self.tokenizer = None

        # Log function calling configuration
        if self.enable_tool_choice:
            logger.info(
                "Function calling enabled with parser: %s", self.tool_call_parser or "default"
            )

        # Create the vLLM engine
        try:
            # Get vLLM imports lazily
            vllm, SamplingParams, get_tokenizer = _get_vllm_imports()

            # First attempt: try with original model name
            try:
                self.engine = vllm.LLM(**engine_kwargs)
                logger.info(f"Successfully loaded model: {self.model_name}")
                # Get the tokenizer after successful engine creation
                # Use a type ignore to handle the tokenizer type mismatch
                self.tokenizer = get_tokenizer(self.model_name)  # type: ignore
                logger.info(f"Initialized vLLM adapter for model: {self.model_name}")
                return
            except Exception as first_error:
                error_msg = str(first_error).lower()
                is_repo_error = any(
                    msg in error_msg
                    for msg in [
                        "repository not found",
                        "invalid repository id",
                        "model not found",
                        "unable to find model",
                    ]
                )

                if is_repo_error and "/" in self.model_name:
                    # Second attempt: try with model name only
                    org, model_without_org = self.model_name.split("/", 1)
                    engine_kwargs["model"] = model_without_org
                    logger.warning(f"Retrying with model name only: {model_without_org}")

                    try:
                        self.engine = vllm.LLM(**engine_kwargs)
                        # Get the tokenizer after successful engine creation
                        # Use a type ignore to handle the tokenizer type mismatch
                        self.tokenizer = get_tokenizer(model_without_org)  # type: ignore
                        logger.info(
                            f"Successfully loaded model using name only: {model_without_org}"
                        )
                        return
                    except Exception as second_error:
                        logger.warning(f"Failed to load with model name only: {second_error}")

                        # Third attempt: try with organization name
                        engine_kwargs["model"] = org
                        logger.warning(f"Retrying with organization name: {org}")

                        try:
                            self.engine = vllm.LLM(**engine_kwargs)
                            # Get the tokenizer after successful engine creation
                            # Use a type ignore to handle the tokenizer type mismatch
                            self.tokenizer = get_tokenizer(org)  # type: ignore
                            logger.info(f"Successfully loaded model using org name: {org}")
                            return
                        except Exception as third_error:
                            logger.error(f"Failed to load with organization name: {third_error}")
                            # Raise detailed error with all attempts
                            msg = (
                                f"Failed to initialize vLLM engine after multiple attempts:\n"
                                f"1. Full name ({self.model_name}): {first_error}\n"
                                f"2. Model name ({model_without_org}): {second_error}\n"
                                f"3. Org name ({org}): {third_error}"
                            )
                            raise RuntimeError(msg)
                else:
                    # Not a repository error or no organization prefix
                    msg = f"Error initializing vLLM engine: {first_error}"
                    raise RuntimeError(msg)

        except Exception as e:
            logger.error(f"Fatal error initializing vLLM engine: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Fatal error initializing vLLM engine: {e}") from e

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
                **kwargs,
            )

        # Set parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Get vLLM imports lazily
        vllm, SamplingParams, get_tokenizer = _get_vllm_imports()

        # Create sampling parameters
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **kwargs)

        # Add JSON mode if requested
        if json_mode:
            # vLLM may support JSON mode in different ways depending on version
            try:
                # Check if the SamplingParams class has a grammar attribute
                import inspect

                sampling_params_attrs = dir(sampling_params)

                if "grammar" in sampling_params_attrs:
                    # For older vLLM versions that support grammar attribute
                    logger.debug("Using grammar attribute for JSON mode")
                    # Use a safer approach that doesn't directly set the attribute
                    sampling_params.grammar = "json"
                elif (
                    "guided_decoding" in sampling_params_attrs
                    or "guided_json" in sampling_params_attrs
                ):
                    # For newer vLLM versions that might use guided_decoding or guided_json
                    if "guided_json" in sampling_params_attrs:
                        logger.debug("Using guided_json attribute for JSON mode")
                        sampling_params.guided_json = True
                    elif "guided_decoding" in sampling_params_attrs:
                        logger.debug("Using guided_decoding attribute for JSON mode")
                        sampling_params.guided_decoding = "json"
                else:
                    logger.warning(
                        "JSON mode requested but no compatible attribute available in SamplingParams"
                    )
            except Exception as e:
                logger.warning(f"Failed to set JSON mode: {e}")

        # Handle function calling
        tool_choice = None
        tools = None

        if functions:
            # Convert functions to tools format for vLLM
            tools = []
            for func in functions:
                tool = {"type": "function", "function": func}
                tools.append(tool)

            # Set tool_choice based on function_call
            if function_call == "auto":
                tool_choice = "auto"
            elif function_call == "none":
                tool_choice = "none"
            elif isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {"type": "function", "function": {"name": function_call["name"]}}
            elif function_call == "required":
                tool_choice = "required"

        try:
            # Generate text
            assert self.engine is not None

            # Prepare generation arguments
            generate_args = {
                "prompt": processed_prompt,
                "sampling_params": sampling_params,
            }

            # Add tools and tool_choice if supported
            if tools and self.enable_tool_choice:
                # Try to use the OpenAI-compatible API for function calling
                try:
                    # Check if the generate method accepts tools and tool_choice parameters
                    import inspect

                    generate_params = inspect.signature(self.engine.generate).parameters

                    if "tools" in generate_params:
                        generate_args["tools"] = tools

                    if "tool_choice" in generate_params:
                        generate_args["tool_choice"] = tool_choice
                except Exception as e:
                    logger.warning(f"Failed to add tools to generation: {e}")

            # Generate text
            # Handle deprecation warning by using the current API
            # The generate method is deprecated but still works
            # In future versions, prompt_token_ids will be part of prompts
            outputs = self.engine.generate(**generate_args)  # type: ignore

            # Get the generated text
            generated_text = outputs[0].outputs[0].text

            # Calculate token usage
            prompt_tokens = 0
            completion_tokens = 0

            # Safely get token counts
            if hasattr(outputs[0], "prompt_token_ids") and outputs[0].prompt_token_ids is not None:
                prompt_tokens = len(outputs[0].prompt_token_ids)

            if (
                hasattr(outputs[0], "outputs")
                and outputs[0].outputs
                and hasattr(outputs[0].outputs[0], "token_ids")
                and outputs[0].outputs[0].token_ids is not None
            ):
                completion_tokens = len(outputs[0].outputs[0].token_ids)

            total_tokens = prompt_tokens + completion_tokens

            # Process function calls if needed
            function_call_result = None
            tool_calls_result = None

            # Check if the output has tool calls (native vLLM function calling)
            # Use getattr to avoid static type checking errors
            tool_calls = None
            if hasattr(outputs[0], "tool_calls"):
                tool_calls = getattr(outputs[0], "tool_calls", None)

            if tool_calls:
                # Native vLLM function calling
                if len(tool_calls) == 1:
                    # Single function call
                    tool_call = tool_calls[0]
                    if tool_call.get("type") == "function":
                        function_call_result = {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        }
                elif len(tool_calls) > 1:
                    # Multiple tool calls
                    tool_calls_result = []
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            tool_calls_result.append(
                                {
                                    "id": tool_call.get("id", f"call_{len(tool_calls_result)}"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["function"]["name"],
                                        "arguments": tool_call["function"]["arguments"],
                                    },
                                }
                            )
            # Fallback to text parsing if native function calling didn't work
            elif (
                functions
                and (function_call == "auto" or isinstance(function_call, dict))
                and not self.enable_tool_choice
            ):
                # Try to parse function calls from the generated text
                function_call_result, tool_calls_result = self._extract_function_calls(
                    generated_text
                )

            # Create response
            return LLMResponse(
                text=generated_text if not function_call_result and not tool_calls_result else None,
                provider=self.provider,
                model_name=self.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                metadata={
                    "model": self.model_name,
                    "provider": self.provider,
                },
                function_call=function_call_result,
                tool_calls=tool_calls_result,
            )
        except Exception as e:
            error_msg = f"Error generating text with vLLM: {e!s}"
            logger.exception(error_msg)
            logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    async def generate_with_cache(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        cache_namespace: str = "default",
        cache_ttl: int | None = 3600,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from the model with caching.

        Args:
        ----
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
        -------
            LLMResponse: The generated response

        """
        # Import here to avoid circular imports
        from saplings.core.caching import generate_with_cache_async
        from saplings.core.caching.interface import CacheStrategy

        # Convert strategy string to enum if provided
        strategy = None
        if cache_strategy := kwargs.pop("cache_strategy", None):
            strategy = CacheStrategy(cache_strategy)

        # Define the generate function
        async def _generate(p, **kw):
            return await self.generate(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                use_cache=False,  # Important to avoid infinite recursion
                **kw,
            )

        # Use the unified caching system
        logger.debug(
            f"Using cache with namespace: {cache_namespace} for vLLM model {self.model_name}"
        )
        if strategy is not None:
            return await generate_with_cache_async(
                generate_func=_generate,
                model_uri=f"{self.provider}:{self.model_name}",
                prompt=prompt,
                namespace=cache_namespace,
                ttl=cache_ttl,
                provider=kwargs.pop("cache_provider", "memory"),
                strategy=strategy,
                **kwargs,
            )
        return await generate_with_cache_async(
            generate_func=_generate,
            model_uri=f"{self.provider}:{self.model_name}",
            prompt=prompt,
            namespace=cache_namespace,
            ttl=cache_ttl,
            provider=kwargs.pop("cache_provider", "memory"),
            **kwargs,
        )

    def _process_prompt(
        self,
        prompt: str | list[dict[str, Any]],
        functions: list[dict[str, Any]] | None = None,
        # Unused parameters kept for API compatibility
        # pylint: disable=unused-argument
        _function_call: str | dict[str, str] | None = None,
        _json_mode: bool = False,
    ) -> str:
        """
        Process the prompt to include functions and other features.

        Args:
        ----
            prompt: The prompt to process
            functions: List of function definitions
            function_call: Function call control
            json_mode: Whether to force JSON output

        Returns:
        -------
            str: The processed prompt

        """
        if isinstance(prompt, str):
            # Simple string prompt
            return prompt

        # Handle message list
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            # If the tokenizer doesn't support chat templates or is None, convert to a simple prompt
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

        # Apply chat template and ensure we return a string
        result = self.tokenizer.apply_chat_template(**chat_template_args)

        # Convert to string if it's not already
        if not isinstance(result, str):
            # Handle different return types
            if hasattr(result, "text"):
                # Some tokenizers might return an object with a text attribute
                return str(getattr(result, "text", ""))
            if hasattr(result, "__str__"):
                # Try to convert to string
                return str(result)
            # Last resort: convert to JSON string if possible
            try:
                import json

                return json.dumps(result)
            except Exception:
                # If all else fails, force string conversion
                return str(result)

        return result

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """
        Convert a list of messages to a simple prompt string.

        Args:
        ----
            messages: List of message dictionaries

        Returns:
        -------
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
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        """
        Extract function calls from generated text.

        Args:
        ----
            text: The generated text

        Returns:
        -------
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
                if "tool_calls" in func_data:
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
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        # Unused parameters kept for API compatibility
        # pylint: disable=unused-argument
        _chunk_size: int | None = None,
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
            _chunk_size: Number of tokens per chunk (unused in this implementation)
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
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Get vLLM imports lazily
        vllm, SamplingParams, get_tokenizer = _get_vllm_imports()

        # Create sampling parameters
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **kwargs)

        # Add JSON mode if requested
        if json_mode:
            # vLLM may support JSON mode in different ways depending on version
            try:
                # Check if the SamplingParams class has a grammar attribute
                sampling_params_attrs = dir(sampling_params)

                if "grammar" in sampling_params_attrs:
                    # For older vLLM versions that support grammar attribute
                    logger.debug("Using grammar attribute for JSON mode")
                    # Use a safer approach that doesn't directly set the attribute
                    sampling_params.grammar = "json"
                elif (
                    "guided_decoding" in sampling_params_attrs
                    or "guided_json" in sampling_params_attrs
                ):
                    # For newer vLLM versions that might use guided_decoding or guided_json
                    if "guided_json" in sampling_params_attrs:
                        logger.debug("Using guided_json attribute for JSON mode")
                        sampling_params.guided_json = True
                    elif "guided_decoding" in sampling_params_attrs:
                        logger.debug("Using guided_decoding attribute for JSON mode")
                        sampling_params.guided_decoding = "json"
                else:
                    logger.warning(
                        "JSON mode requested but no compatible attribute available in SamplingParams"
                    )
            except Exception as e:
                logger.warning(f"Failed to set JSON mode: {e}")

        # Handle function calling
        tool_choice = None
        tools = None

        if functions:
            # Convert functions to tools format for vLLM
            tools = []
            for func in functions:
                tool = {"type": "function", "function": func}
                tools.append(tool)

            # Set tool_choice based on function_call
            if function_call == "auto":
                tool_choice = "auto"
            elif function_call == "none":
                tool_choice = "none"
            elif isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {"type": "function", "function": {"name": function_call["name"]}}
            elif function_call == "required":
                tool_choice = "required"

        # Generate text with streaming
        assert self.engine is not None

        # Prepare generation arguments
        generate_args = {
            "prompts": [processed_prompt],
            "sampling_params": sampling_params,
        }

        # Add tools and tool_choice if supported
        if tools and self.enable_tool_choice:
            # Try to use the OpenAI-compatible API for function calling
            try:
                # Check if the generate_iterator method accepts tools and tool_choice parameters
                import inspect

                # Check if generate_iterator exists
                if hasattr(self.engine, "generate_iterator"):
                    # Use getattr to avoid static type checking errors
                    generate_iterator_method = self.engine.generate_iterator

                    try:
                        generate_params = inspect.signature(generate_iterator_method).parameters

                        if "tools" in generate_params:
                            generate_args["tools"] = tools

                        if "tool_choice" in generate_params:
                            generate_args["tool_choice"] = tool_choice
                    except Exception as e:
                        logger.warning(f"Failed to inspect generate_iterator parameters: {e}")
            except Exception as e:
                logger.warning(f"Failed to add tools to streaming generation: {e}")

        # Check if generate_iterator exists
        if not hasattr(self.engine, "generate_iterator"):
            logger.warning(
                "generate_iterator not available in this vLLM version. Falling back to non-streaming generation."
            )
            # Fall back to non-streaming generation
            # Handle deprecation warning by using the current API
            outputs = self.engine.generate(**generate_args)  # type: ignore
            yield outputs[0].outputs[0].text
            return

        # Use generate_iterator for streaming
        # Type ignore is needed because the static type checker doesn't know about this method
        # Use getattr to avoid static type checking errors
        generate_iterator_method = self.engine.generate_iterator
        outputs_generator = generate_iterator_method(**generate_args)  # type: ignore

        # For function calling, we need to accumulate the text
        accumulated_text = ""
        last_tool_calls = None

        # Stream the outputs
        for outputs in outputs_generator:
            output = outputs[0]

            # Check for tool calls in the output (native vLLM function calling)
            if (
                hasattr(output, "tool_calls")
                and output.tool_calls
                and output.tool_calls != last_tool_calls
            ):
                # Native vLLM function calling
                tool_calls = output.tool_calls
                last_tool_calls = tool_calls

                if len(tool_calls) == 1:
                    # Single function call
                    tool_call = tool_calls[0]
                    if tool_call.get("type") == "function":
                        yield {
                            "function_call": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            }
                        }
                        return
                elif len(tool_calls) > 1:
                    # Multiple tool calls
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            yield {
                                "tool_call": {
                                    "id": tool_call.get("id", f"call_{len(tool_calls)}"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["function"]["name"],
                                        "arguments": tool_call["function"]["arguments"],
                                    },
                                }
                            }
                    return

            # Process text chunks
            if output.outputs:
                chunk = output.outputs[0].text

                # If we're using function calling but not native support, check for function calls in text
                if (
                    functions
                    and (function_call == "auto" or isinstance(function_call, dict))
                    and not self.enable_tool_choice
                ):
                    accumulated_text += chunk
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
                        yield chunk
                else:
                    # Normal text generation
                    yield chunk

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
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
                provider=self.provider,
                version="latest",  # Default version
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

        Returns
        -------
            int: Context window size in tokens

        """
        # Try to get the context window from the model config
        model_name = self.model_name.lower()

        # Try to get context window from model config first
        if (
            self.model is not None
            and hasattr(self.model, "config")
            and hasattr(self.model.config, "max_position_embeddings")
        ):
            return self.model.config.max_position_embeddings

        # Common model context windows
        if "llama-3-70b" in model_name or "llama-3" in model_name:
            return 128000  # Updated for Llama 3
        if "llama-2-70b" in model_name:
            return 4096
        if "llama-2-13b" in model_name:
            return 4096
        if "llama-2-7b" in model_name:
            return 4096
        if "mistral-7b" in model_name:
            return 32768
        if "mixtral-8x7b" in model_name:
            return 32768
        if "qwen-7b" in model_name:
            return 32768
        if "qwen-14b" in model_name:
            return 32768
        if "qwen-72b" in model_name:
            return 32768
        if "yi-34b" in model_name:
            return 32768
        if "yi-6b" in model_name:
            return 4096
        if "phi-2" in model_name:
            return 2048
        if "gemma-7b" in model_name:
            return 8192
        if "gemma-2b" in model_name:
            return 8192
        if "mamba" in model_name:
            return 2048

        # Try to infer from model name
        for size in ["32k", "16k", "8k", "4k", "2k"]:
            if size in model_name:
                return int(size.replace("k", "000"))

        return 4096  # Conservative default

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
        # Use the tokenizer to get the exact token count if available
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))

        # Fallback to a simple estimation if tokenizer is not available
        words = len(text.split())
        return int(words * 1.3)  # Rough estimation

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
        ----
            _prompt_tokens: Number of tokens in the prompt (unused for local models)
            _completion_tokens: Number of tokens in the completion (unused for local models)

        Returns:
        -------
            float: Estimated cost in USD

        """
        # Local models have no API cost
        # Suppress unused parameter warnings
        _ = prompt_tokens
        _ = completion_tokens
        return 0.0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: int | None = 3600,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response to a conversation.

        Args:
        ----
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
        -------
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
                **kwargs,
            )
        return await self.generate(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs,
        )

    def cleanup(self) -> None:
        """
        Clean up resources used by the model.

        This method should be called when the model is no longer needed.
        """
        if hasattr(self, "engine") and self.engine is not None:
            import gc

            import torch

            # Clean up vLLM resources
            # Try to clean up any resources that might be available
            try:
                if hasattr(self.engine, "llm_engine"):
                    llm_engine = self.engine.llm_engine
                    if hasattr(llm_engine, "model_executor"):
                        model_executor = llm_engine.model_executor
                        if hasattr(model_executor, "driver_worker"):
                            # Use delattr to avoid static type checking errors
                            delattr(model_executor, "driver_worker")
            except Exception as e:
                logger.warning(f"Error during cleanup of vLLM resources: {e}")

            # Delete the engine
            del self.engine
            self.engine = None

            # Run garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Cleaned up vLLM adapter for model: {self.model_name}")

    # No longer implementing the Plugin interface
