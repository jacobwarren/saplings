"""
Example of using the function registry and parallel function calling in Saplings.

This example demonstrates how to register functions, call them in parallel,
and use them with LLMs.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

from saplings.core.function_registry import register_function
from saplings.core.parallel_function import call_functions_parallel
from saplings.core.model_adapter import LLM


# Register some example functions
@register_function(description="Get the weather for a location", group="weather")
def get_weather(location: str, unit: str = "celsius") -> Dict:
    """
    Get the weather for a location.
    
    Args:
        location: The location to get weather for
        unit: The unit to use (celsius or fahrenheit)
        
    Returns:
        Dict: Weather information
    """
    # In a real application, this would call a weather API
    # For this example, we'll just return some fake data
    print(f"Getting weather for {location} in {unit}")
    
    # Simulate API call
    time.sleep(0.5)
    
    # Return fake data
    if unit == "celsius":
        temperature = 22
    else:
        temperature = 72
    
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "condition": "sunny"
    }


@register_function(description="Get information about a city", group="location")
def get_city_info(city: str) -> Dict:
    """
    Get information about a city.
    
    Args:
        city: The city to get information for
        
    Returns:
        Dict: City information
    """
    # In a real application, this would call a database or API
    # For this example, we'll just return some fake data
    print(f"Getting information for {city}")
    
    # Simulate API call
    time.sleep(0.3)
    
    # Return fake data
    city_data = {
        "New York": {
            "country": "USA",
            "population": 8_400_000,
            "timezone": "EST",
        },
        "London": {
            "country": "UK",
            "population": 8_900_000,
            "timezone": "GMT",
        },
        "Tokyo": {
            "country": "Japan",
            "population": 13_900_000,
            "timezone": "JST",
        },
        "Paris": {
            "country": "France",
            "population": 2_100_000,
            "timezone": "CET",
        },
    }
    
    return city_data.get(city, {"error": f"No information available for {city}"})


@register_function(description="Convert currency", group="finance")
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict:
    """
    Convert currency.
    
    Args:
        amount: The amount to convert
        from_currency: The currency to convert from
        to_currency: The currency to convert to
        
    Returns:
        Dict: Conversion result
    """
    # In a real application, this would call a currency API
    # For this example, we'll just return some fake data
    print(f"Converting {amount} {from_currency} to {to_currency}")
    
    # Simulate API call
    await asyncio.sleep(0.2)
    
    # Fake exchange rates
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.75,
        "JPY": 110.0,
    }
    
    # Convert
    if from_currency in rates and to_currency in rates:
        result = amount * (rates[to_currency] / rates[from_currency])
    else:
        return {"error": f"Unsupported currency: {from_currency} or {to_currency}"}
    
    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "result": round(result, 2)
    }


async def run_parallel_functions_example():
    """Run an example of parallel function calling."""
    print("=== Parallel Function Calling Example ===")
    
    # Define function calls
    function_calls = [
        {"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}},
        {"name": "get_city_info", "arguments": {"city": "Tokyo"}},
        {"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}},
    ]
    
    print("Calling functions in parallel...")
    start_time = time.time()
    
    # Call functions in parallel
    results = await call_functions_parallel(function_calls)
    
    end_time = time.time()
    print(f"All functions completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    for name, result in results:
        print(f"\nResult from {name}:")
        print(json.dumps(result, indent=2))


async def run_llm_with_functions_example():
    """Run an example of using functions with an LLM."""
    print("\n=== LLM with Functions Example ===")
    
    # Check if we have an OpenAI API key
    try:
        # Create an OpenAI model
        model = LLM.from_uri("openai://gpt-4")
        
        # Get function definitions from the registry
        from saplings.core.function_registry import function_registry
        weather_functions = function_registry.get_group_definitions("weather")
        
        print("Sending request to model with functions...")
        
        # Create a conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can check the weather."},
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ]
        
        # Generate a response with function calling
        response = await model.generate(
            prompt=messages,
            functions=weather_functions,
            function_call="auto"
        )
        
        # Check if the model decided to call a function
        if response.function_call:
            print("\nModel decided to call a function:")
            print(f"Function: {response.function_call['name']}")
            print(f"Arguments: {response.function_call['arguments']}")
            
            # Call the function
            function_name = response.function_call["name"]
            function_args = json.loads(response.function_call["arguments"])
            
            result = function_registry.call_function(function_name, function_args)
            
            print("\nFunction result:")
            print(json.dumps(result, indent=2))
            
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
                "content": json.dumps(result)
            })
            
            # Get the final response
            final_response = await model.generate(prompt=messages)
            
            print("\nFinal response:")
            print(final_response.text)
        else:
            print("\nResponse:")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
        print("Skipping LLM example (OpenAI API key may be missing)")


async def main():
    """Run the examples."""
    await run_parallel_functions_example()
    await run_llm_with_functions_example()


if __name__ == "__main__":
    asyncio.run(main())
