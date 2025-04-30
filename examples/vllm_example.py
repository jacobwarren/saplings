"""
Example of using vLLM for inference in Saplings.

This example demonstrates how to use the vLLM adapter for high-performance inference,
including advanced features like function calling, JSON mode, and caching.
"""

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Dict, List, Optional

from saplings.core.model_adapter import LLM, ModelURI
from saplings.core.model_caching import clear_model_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def run_vllm_example(model_name: Optional[str] = None):
    """
    Run an example of using vLLM for inference.

    Args:
        model_name: Name of the model to use, or None to use the default
    """
    print("=== vLLM Inference Example ===")

    # Use the provided model name or a default
    model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_name}")

    # Create a vLLM model
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=1024"
    model = LLM.from_uri(model_uri)

    print("Model loaded successfully")

    # Generate text
    prompt = "Explain the concept of self-improving AI in simple terms."

    print(f"Generating response for prompt: {prompt}")
    start_time = time.time()

    response = await model.generate(prompt)

    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")

    # Print the response
    print("\nResponse:")
    print(response.text)

    # Print token usage
    print("\nToken usage:")
    print(f"  Prompt tokens: {response.usage['prompt_tokens']}")
    print(f"  Completion tokens: {response.usage['completion_tokens']}")
    print(f"  Total tokens: {response.usage['total_tokens']}")

    # Generate text with streaming
    prompt = "What are the key benefits of using vLLM for inference?"

    print(f"\nStreaming response for prompt: {prompt}")
    print("\nResponse (streaming):")

    start_time = time.time()

    async for chunk in model.generate_streaming(prompt):
        print(chunk, end="", flush=True)

    end_time = time.time()
    print(f"\n\nStreaming completed in {end_time - start_time:.2f} seconds")

    # Clean up
    if hasattr(model, 'cleanup'):
        model.cleanup()

    print("\nExample completed successfully")


async def run_provider_comparison():
    """Run a comparison between different model providers."""
    print("=== Model Provider Comparison ===")

    # Define the prompt
    prompt = "Explain the concept of self-improving AI in simple terms."

    # Test vLLM (local inference)
    try:
        print("\n--- vLLM Provider ---")
        model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")
        start_time = time.time()
        response = await model.generate(prompt)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Token usage: {response.usage['total_tokens']} tokens")
        print(f"Response: {response.text[:100]}...")
        if hasattr(model, 'cleanup'):
            model.cleanup()
    except Exception as e:
        print(f"Error with vLLM: {e}")

    # Test OpenAI
    try:
        print("\n--- OpenAI Provider ---")
        model = LLM.from_uri("openai://gpt-3.5-turbo")
        start_time = time.time()
        response = await model.generate(prompt)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Token usage: {response.usage['total_tokens']} tokens")
        print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error with OpenAI: {e}")

    # Test Anthropic
    try:
        print("\n--- Anthropic Provider ---")
        model = LLM.from_uri("anthropic://claude-3-haiku-20240307")
        start_time = time.time()
        response = await model.generate(prompt)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Token usage: {response.usage['total_tokens']} tokens")
        print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error with Anthropic: {e}")

    # Test HuggingFace
    try:
        print("\n--- HuggingFace Provider ---")
        model = LLM.from_uri("huggingface://HuggingFaceTB/SmolLM-1.7B-Instruct")
        start_time = time.time()
        response = await model.generate(prompt)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Token usage: {response.usage['total_tokens']} tokens")
        print(f"Response: {response.text[:100]}...")
        if hasattr(model, 'cleanup'):
            model.cleanup()
    except Exception as e:
        print(f"Error with HuggingFace: {e}")

    print("\nComparison completed")


async def run_function_calling_example(model_name: Optional[str] = None):
    """
    Run an example of function calling with vLLM.

    Args:
        model_name: Name of the model to use, or None to use the default
    """
    print("=== Function Calling Example ===")

    # Use the provided model name or a default
    model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_name}")

    # Create a vLLM model with native function calling enabled
    # For Llama 3.1, we use the llama3_json tool parser
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=1024&enable_tool_choice=true&tool_call_parser=llama3_json&chat_template=tool_use"
    model = LLM.from_uri(model_uri)

    # Define a function
    get_weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to use",
                }
            },
            "required": ["location"]
        }
    }

    # Generate a response with function calling
    print("Generating response with function calling...")
    response = await model.generate(
        "What's the weather like in San Francisco?",
        functions=[get_weather_function],
        function_call="auto"
    )

    # Check if the model decided to call a function
    if response.function_call:
        print(f"Function call: {response.function_call['name']}")
        print(f"Arguments: {response.function_call['arguments']}")

        # Simulate executing the function
        location = json.loads(response.function_call['arguments']).get('location', 'unknown')
        unit = json.loads(response.function_call['arguments']).get('unit', 'celsius')
        print(f"\nExecuting function get_weather({location}, {unit})...")
        weather_result = f"The weather in {location} is currently 72°F (22°C) and sunny."

        # Create a follow-up message with the function result
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco?"},
            {"role": "assistant", "content": None, "function_call": {
                "name": "get_weather",
                "arguments": response.function_call['arguments']
            }},
            {"role": "function", "name": "get_weather", "content": weather_result}
        ]

        # Get the assistant's response to the function result
        print("\nGetting assistant's response to function result...")
        final_response = await model.chat(messages)
        print(f"Final response: {final_response.text}")
    else:
        print(f"Response: {response.text}")

    # Clean up
    if hasattr(model, 'cleanup'):
        model.cleanup()

    print("\nFunction calling example completed")


async def run_json_mode_example(model_name: Optional[str] = None):
    """
    Run an example of JSON mode with vLLM.

    Args:
        model_name: Name of the model to use, or None to use the default
    """
    print("=== JSON Mode Example ===")

    # Use the provided model name or a default
    model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_name}")

    # Create a vLLM model
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=1024"
    model = LLM.from_uri(model_uri)

    # Generate text with JSON mode
    print("Generating JSON response...")
    response = await model.generate(
        "List the three primary colors as a JSON array",
        json_mode=True
    )

    print(f"Response: {response.text}")

    # Try to parse the response as JSON
    try:
        colors = json.loads(response.text)
        print(f"Parsed JSON: {colors}")
    except json.JSONDecodeError:
        print("Response is not valid JSON")

    # Clean up
    if hasattr(model, 'cleanup'):
        model.cleanup()

    print("\nJSON mode example completed")


async def run_caching_example(model_name: Optional[str] = None):
    """
    Run an example of caching with vLLM.

    Args:
        model_name: Name of the model to use, or None to use the default
    """
    print("=== Caching Example ===")

    # Use the provided model name or a default
    model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_name}")

    # Create a vLLM model
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=1024"
    model = LLM.from_uri(model_uri)

    # Generate text without caching
    prompt = "What is the capital of Italy?"
    print(f"Generating first response for prompt: {prompt} (no cache)...")
    start_time = time.time()
    response1 = await model.generate(prompt)
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    print(f"Response: {response1.text}")

    # Generate the same text with caching
    print(f"\nGenerating second response for prompt: {prompt} (with cache)...")
    start_time = time.time()
    response2 = await model.generate(
        prompt,
        use_cache=True
    )
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    print(f"Response: {response2.text}")

    # Clear the cache
    clear_model_cache()

    # Clean up
    if hasattr(model, 'cleanup'):
        model.cleanup()

    print("\nCaching example completed")


async def run_required_function_calling_example(model_name: Optional[str] = None):
    """
    Run an example of required function calling with vLLM.

    Args:
        model_name: Name of the model to use, or None to use the default
    """
    print("=== Required Function Calling Example ===")

    # Use the provided model name or a default
    model_name = model_name or "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_name}")

    # Create a vLLM model with native function calling enabled
    model_uri = f"vllm://{model_name}?temperature=0.7&max_tokens=1024&enable_tool_choice=true&tool_call_parser=llama3_json&chat_template=tool_use"
    model = LLM.from_uri(model_uri)

    # Define multiple functions
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to use",
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "get_restaurant",
            "description": "Find a restaurant in a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Type of food, e.g. Italian, Chinese, etc.",
                    },
                    "price_range": {
                        "type": "string",
                        "enum": ["$", "$$", "$$$", "$$$$"],
                        "description": "Price range from $ (cheap) to $$$$ (expensive)",
                    }
                },
                "required": ["location", "cuisine"]
            }
        }
    ]

    # Generate a response with required function calling
    print("Generating response with required function calling...")
    response = await model.generate(
        "I'm visiting San Francisco next week and I'm wondering what the weather will be like.",
        functions=functions,
        function_call="required"  # Force the model to call a function
    )

    # The model must call a function
    if response.function_call:
        print(f"Function call: {response.function_call['name']}")
        print(f"Arguments: {response.function_call['arguments']}")
    else:
        print("Error: Model did not call a function as required")

    # Clean up
    if hasattr(model, 'cleanup'):
        model.cleanup()

    print("\nRequired function calling example completed")


async def main():
    """Run the examples."""
    try:
        # Check if a model name was provided as an environment variable
        model_name = os.environ.get("VLLM_MODEL")

        # Run the examples
        await run_vllm_example(model_name)

        # Uncomment to run additional examples
        # await run_function_calling_example(model_name)
        # await run_required_function_calling_example(model_name)
        # await run_json_mode_example(model_name)
        # await run_caching_example(model_name)
        # await run_provider_comparison()
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure vLLM is installed: pip install vllm")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Stack trace: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())
