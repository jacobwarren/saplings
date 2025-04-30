"""
Example of using advanced features in Saplings.

This example demonstrates how to use advanced features like function calling,
vision models, and structured output.
"""

import asyncio
import base64
import json
import os
from typing import Dict, List, Optional, Union

from saplings.core.model_adapter import LLM, LLMResponse, ModelURI


async def run_function_calling_example():
    """
    Run an example of function calling.
    """
    print("=== Function Calling Example ===")
    
    # Define a function that the model can call
    get_weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
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
    
    # Create an OpenAI model
    model = LLM.from_uri("openai://gpt-4")
    
    # Create a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can check the weather."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    print("Sending request to model...")
    
    # Generate a response with function calling
    response = await model.generate(
        prompt=messages,
        functions=[get_weather_function],
        function_call="auto"
    )
    
    # Check if the model decided to call a function
    if response.function_call:
        print("\nModel decided to call a function:")
        print(f"Function: {response.function_call['name']}")
        print(f"Arguments: {response.function_call['arguments']}")
        
        # In a real application, you would call the actual function here
        # For this example, we'll just simulate a response
        function_name = response.function_call["name"]
        function_args = json.loads(response.function_call["arguments"])
        
        if function_name == "get_weather":
            # Simulate getting the weather
            weather_result = {
                "location": function_args["location"],
                "temperature": 22 if function_args.get("unit") == "celsius" else 72,
                "unit": function_args.get("unit", "celsius"),
                "condition": "sunny"
            }
            
            # Add the function result to the conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": response.function_call["arguments"]
                }
            })
            
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(weather_result)
            })
            
            # Get the final response
            final_response = await model.generate(prompt=messages)
            
            print("\nFinal response:")
            print(final_response.text)
    else:
        print("\nResponse:")
        print(response.text)


async def run_vision_example():
    """
    Run an example of using vision models.
    """
    print("\n=== Vision Model Example ===")
    
    # Check if we have an OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping vision example: OPENAI_API_KEY not set")
        return
    
    # Create a vision model
    model = LLM.from_uri("openai://gpt-4-vision-preview")
    
    # Create a tiny 1x1 transparent PNG for the example
    image_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    image_base64 = base64.b64encode(image_data).decode()
    
    # Create a message with text and an image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]
    
    print("Sending request to vision model...")
    
    # Generate a response
    response = await model.generate(prompt=messages)
    
    print("\nResponse:")
    print(response.text)


async def run_json_mode_example():
    """
    Run an example of using JSON mode.
    """
    print("\n=== JSON Mode Example ===")
    
    # Create an OpenAI model
    model = LLM.from_uri("openai://gpt-4")
    
    # Create a prompt that asks for JSON
    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns data in JSON format."},
        {"role": "user", "content": "Give me information about the top 3 programming languages in 2023."}
    ]
    
    print("Sending request with JSON mode...")
    
    # Generate a response with JSON mode
    response = await model.generate(
        prompt=messages,
        json_mode=True
    )
    
    print("\nResponse (JSON):")
    print(response.text)
    
    # Parse the JSON
    try:
        data = json.loads(response.text)
        print("\nParsed JSON:")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON: {e}")


async def run_streaming_example():
    """
    Run an example of streaming with advanced features.
    """
    print("\n=== Streaming Example ===")
    
    # Create an OpenAI model
    model = LLM.from_uri("openai://gpt-4")
    
    # Define a function that the model can call
    get_weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
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
    
    # Create a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can check the weather."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    print("Streaming response...")
    
    # Generate a streaming response with function calling
    function_call_detected = False
    function_call_data = {}
    
    print("\nResponse: ", end="", flush=True)
    
    async for chunk in model.generate_streaming(
        prompt=messages,
        functions=[get_weather_function],
        function_call="auto"
    ):
        if isinstance(chunk, dict) and "function_call" in chunk:
            # Function call detected
            function_call_detected = True
            
            # Update function call data
            if "name" in chunk["function_call"] and chunk["function_call"]["name"]:
                function_call_data["name"] = chunk["function_call"].get("name", "")
            
            if "arguments" in chunk["function_call"] and chunk["function_call"]["arguments"]:
                if "arguments" not in function_call_data:
                    function_call_data["arguments"] = ""
                function_call_data["arguments"] += chunk["function_call"]["arguments"]
        elif isinstance(chunk, str):
            # Text chunk
            print(chunk, end="", flush=True)
    
    print()
    
    if function_call_detected:
        print("\nFunction call detected:")
        print(f"Function: {function_call_data.get('name', '')}")
        print(f"Arguments: {function_call_data.get('arguments', '')}")


async def main():
    """Run the examples."""
    await run_function_calling_example()
    await run_vision_example()
    await run_json_mode_example()
    await run_streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
