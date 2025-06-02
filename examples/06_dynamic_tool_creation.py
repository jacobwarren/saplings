#!/usr/bin/env python3
"""
Dynamic Tool Creation Example

This example demonstrates how to dynamically create, register, and use tools
at runtime based on user requirements or changing conditions.
"""

import asyncio
import os
import inspect
import json
from typing import Any, Dict, List, Callable
from saplings import AgentBuilder
from saplings.api.tools import tool


class DynamicToolFactory:
    """Factory for creating tools dynamically at runtime."""
    
    def __init__(self):
        self.created_tools = {}
    
    def create_calculator_tool(self, operations: List[str]) -> Callable:
        """Create a calculator tool with specified operations."""
        
        def safe_eval(expression: str, allowed_ops: List[str]) -> float:
            """Safely evaluate mathematical expressions."""
            import ast
            import operator
            
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            # Only allow specified operations
            if "add" not in allowed_ops:
                ops.pop(ast.Add, None)
            if "sub" not in allowed_ops:
                ops.pop(ast.Sub, None)
            if "mult" not in allowed_ops:
                ops.pop(ast.Mult, None)
            if "div" not in allowed_ops:
                ops.pop(ast.Div, None)
            if "pow" not in allowed_ops:
                ops.pop(ast.Pow, None)
            
            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    if type(node.op) not in ops:
                        raise ValueError(f"Operation {type(node.op).__name__} not allowed")
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    if type(node.op) not in ops:
                        raise ValueError(f"Operation {type(node.op).__name__} not allowed")
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")
            
            return eval_expr(ast.parse(expression, mode='eval').body)
        
        @tool(
            name=f"calculator_{hash(str(operations))}",
            description=f"Calculator supporting: {', '.join(operations)}"
        )
        def dynamic_calculator(expression: str) -> str:
            """
            Calculate mathematical expressions with limited operations.
            
            Args:
                expression: Mathematical expression to evaluate
            
            Returns:
                The calculated result
            """
            try:
                result = safe_eval(expression, operations)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        tool_id = f"calculator_{hash(str(operations))}"
        self.created_tools[tool_id] = dynamic_calculator
        return dynamic_calculator
    
    def create_text_processor_tool(self, operations: List[str]) -> Callable:
        """Create a text processing tool with specified operations."""
        
        @tool(
            name=f"text_processor_{hash(str(operations))}",
            description=f"Text processor supporting: {', '.join(operations)}"
        )
        def dynamic_text_processor(text: str, operation: str = None) -> str:
            """
            Process text with various operations.
            
            Args:
                text: Text to process
                operation: Specific operation to perform (optional)
            
            Returns:
                Processed text
            """
            # If operation specified, use it; otherwise use all available
            ops_to_apply = [operation] if operation and operation in operations else operations
            
            result = text
            for op in ops_to_apply:
                if op == "uppercase":
                    result = result.upper()
                elif op == "lowercase":
                    result = result.lower()
                elif op == "reverse":
                    result = result[::-1]
                elif op == "word_count":
                    count = len(result.split())
                    result = f"Word count: {count}"
                    break  # Don't continue processing after count
                elif op == "char_count":
                    count = len(result)
                    result = f"Character count: {count}"
                    break
                elif op == "title_case":
                    result = result.title()
                elif op == "remove_spaces":
                    result = result.replace(" ", "")
            
            return result
        
        tool_id = f"text_processor_{hash(str(operations))}"
        self.created_tools[tool_id] = dynamic_text_processor
        return dynamic_text_processor
    
    def create_data_transformer_tool(self, input_format: str, output_format: str) -> Callable:
        """Create a data format transformer tool."""
        
        @tool(
            name=f"transformer_{input_format}_to_{output_format}",
            description=f"Transform data from {input_format} to {output_format}"
        )
        def dynamic_transformer(data: str) -> str:
            """
            Transform data between formats.
            
            Args:
                data: Input data in source format
            
            Returns:
                Transformed data in target format
            """
            try:
                if input_format == "json" and output_format == "csv":
                    import json
                    import csv
                    import io
                    
                    json_data = json.loads(data)
                    if isinstance(json_data, list) and json_data:
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=json_data[0].keys())
                        writer.writeheader()
                        writer.writerows(json_data)
                        return output.getvalue()
                    else:
                        return "Error: JSON must be a list of objects"
                
                elif input_format == "csv" and output_format == "json":
                    import csv
                    import json
                    import io
                    
                    input_io = io.StringIO(data)
                    reader = csv.DictReader(input_io)
                    result = list(reader)
                    return json.dumps(result, indent=2)
                
                elif input_format == "text" and output_format == "json":
                    lines = data.strip().split('\n')
                    result = {"lines": lines, "line_count": len(lines)}
                    return json.dumps(result, indent=2)
                
                else:
                    return f"Transformation from {input_format} to {output_format} not implemented"
                    
            except Exception as e:
                return f"Transformation error: {e}"
        
        tool_id = f"transformer_{input_format}_to_{output_format}"
        self.created_tools[tool_id] = dynamic_transformer
        return dynamic_transformer
    
    def create_api_client_tool(self, api_config: Dict[str, Any]) -> Callable:
        """Create an API client tool with specific configuration."""
        
        @tool(
            name=f"api_client_{api_config.get('name', 'unknown')}",
            description=f"API client for {api_config.get('description', 'external service')}"
        )
        def dynamic_api_client(endpoint: str, method: str = "GET", params: str = "{}") -> str:
            """
            Make API requests to configured service.
            
            Args:
                endpoint: API endpoint to call
                method: HTTP method (GET, POST, etc.)
                params: JSON string of parameters
            
            Returns:
                API response
            """
            try:
                # This is a mock implementation for demonstration
                # In real usage, you'd use requests or httpx
                import json
                
                base_url = api_config.get('base_url', 'https://api.example.com')
                parsed_params = json.loads(params)
                
                # Simulate API call
                mock_response = {
                    "status": "success",
                    "url": f"{base_url}/{endpoint}",
                    "method": method,
                    "params": parsed_params,
                    "data": "Mock API response data"
                }
                
                return json.dumps(mock_response, indent=2)
                
            except Exception as e:
                return f"API call error: {e}"
        
        tool_id = f"api_client_{api_config.get('name', 'unknown')}"
        self.created_tools[tool_id] = dynamic_api_client
        return dynamic_api_client
    
    def get_tool_info(self) -> List[Dict[str, str]]:
        """Get information about all created tools."""
        info = []
        for tool_id, tool_func in self.created_tools.items():
            info.append({
                "id": tool_id,
                "name": tool_func.__name__,
                "description": tool_func.__doc__.split('\n')[1].strip() if tool_func.__doc__ else "No description",
                "signature": str(inspect.signature(tool_func))
            })
        return info


async def demonstrate_basic_dynamic_tools():
    """Demonstrate basic dynamic tool creation."""
    print("=== Basic Dynamic Tool Creation ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating tool factory...")
    factory = DynamicToolFactory()
    
    print("2. Creating specialized calculator tools...")
    
    # Create different calculator tools with different capabilities
    basic_calc = factory.create_calculator_tool(["add", "sub"])
    advanced_calc = factory.create_calculator_tool(["add", "sub", "mult", "div", "pow"])
    
    # Create text processing tools
    basic_text = factory.create_text_processor_tool(["uppercase", "lowercase"])
    advanced_text = factory.create_text_processor_tool(["uppercase", "lowercase", "reverse", "word_count", "title_case"])
    
    print("3. Creating agent with dynamic tools...")
    
    agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_tools([
        basic_calc,
        advanced_calc,
        basic_text,
        advanced_text
    ]).build()
    
    print("4. Testing dynamic tools...")
    
    response = await agent.run("""
    I need to:
    1. Calculate 15 + 25 using the basic calculator
    2. Calculate 2^3 * 4 using the advanced calculator  
    3. Convert "hello world" to uppercase using text processing
    4. Count words in "The quick brown fox jumps over the lazy dog"
    
    Please use the appropriate tools for each task.
    """)
    
    print(f"Agent response: {response}")
    
    # Show tool information
    print("\n5. Tool factory information:")
    for tool_info in factory.get_tool_info():
        print(f"   - {tool_info['name']}: {tool_info['description']}")


async def demonstrate_runtime_tool_creation():
    """Demonstrate creating tools based on runtime requirements."""
    print("\n=== Runtime Tool Creation ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Setting up adaptive tool system...")
    
    factory = DynamicToolFactory()
    
    # Start with basic agent
    initial_tools = [
        factory.create_text_processor_tool(["word_count", "char_count"])
    ]
    
    agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_tools(initial_tools).build()
    
    print("2. Agent starts with basic text processing...")
    
    response1 = await agent.run("Count the words in: 'Dynamic tool creation is powerful'")
    print(f"Initial response: {response1}")
    
    print("\n3. User requests mathematical capabilities...")
    
    # Create math tools based on request
    math_tool = factory.create_calculator_tool(["add", "sub", "mult", "div"])
    
    # Create new agent with expanded capabilities
    expanded_tools = initial_tools + [math_tool]
    
    enhanced_agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_tools(expanded_tools).build()
    
    response2 = await enhanced_agent.run("""
    Now I need both text and math:
    1. Count words in "Machine learning is fascinating"
    2. Calculate (10 + 5) * 2
    """)
    
    print(f"Enhanced response: {response2}")
    
    print("\n4. User requests data transformation...")
    
    # Add data transformation capability
    transformer_tool = factory.create_data_transformer_tool("json", "csv")
    
    final_tools = expanded_tools + [transformer_tool]
    
    complete_agent = AgentBuilder.for_openai(
        "gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_tools(final_tools).build()
    
    test_json = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    
    response3 = await complete_agent.run(f"""
    Please help me with this JSON data: {test_json}
    1. Transform it to CSV format
    2. Count the number of characters in the original JSON
    3. Calculate the average age (30 + 25) / 2
    """)
    
    print(f"Complete response: {response3}")


async def demonstrate_conditional_tool_creation():
    """Demonstrate creating tools based on conditions."""
    print("\n=== Conditional Tool Creation ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Setting up conditional tool system...")
    
    factory = DynamicToolFactory()
    
    # Simulate different user profiles
    user_profiles = [
        {
            "name": "Data Scientist",
            "needs": ["math", "data_transformation"],
            "level": "advanced"
        },
        {
            "name": "Content Writer", 
            "needs": ["text_processing"],
            "level": "basic"
        },
        {
            "name": "API Developer",
            "needs": ["api_client", "data_transformation"],
            "level": "advanced"
        }
    ]
    
    for profile in user_profiles:
        print(f"\n2. Creating tools for {profile['name']}...")
        
        tools = []
        
        # Add tools based on profile needs
        if "math" in profile["needs"]:
            if profile["level"] == "advanced":
                tools.append(factory.create_calculator_tool(["add", "sub", "mult", "div", "pow"]))
            else:
                tools.append(factory.create_calculator_tool(["add", "sub"]))
        
        if "text_processing" in profile["needs"]:
            if profile["level"] == "advanced":
                tools.append(factory.create_text_processor_tool(["uppercase", "lowercase", "reverse", "word_count", "title_case"]))
            else:
                tools.append(factory.create_text_processor_tool(["uppercase", "lowercase"]))
        
        if "data_transformation" in profile["needs"]:
            tools.append(factory.create_data_transformer_tool("json", "csv"))
            tools.append(factory.create_data_transformer_tool("csv", "json"))
        
        if "api_client" in profile["needs"]:
            api_config = {
                "name": "github",
                "description": "GitHub API client",
                "base_url": "https://api.github.com"
            }
            tools.append(factory.create_api_client_tool(api_config))
        
        # Create specialized agent
        specialized_agent = AgentBuilder.for_openai(
            "gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        ).with_tools(tools).build()
        
        # Test with profile-specific task
        if profile["name"] == "Data Scientist":
            task = "Calculate the mean of these values: 15, 22, 18, 30, 25. Then convert this JSON to CSV: '[{\"value\": 15}, {\"value\": 22}]'"
        elif profile["name"] == "Content Writer":
            task = "Convert this text to title case: 'the importance of dynamic tools in ai systems'"
        else:  # API Developer
            task = "Make a mock API call to get user info from endpoint 'users/123' and convert the response to JSON format"
        
        response = await specialized_agent.run(f"""
        As a {profile['name']}, I need help with: {task}
        Please use the appropriate tools available to you.
        """)
        
        print(f"   Task: {task}")
        print(f"   Response: {response[:200]}...")


async def demonstrate_tool_composition():
    """Demonstrate composing complex tools from simpler ones."""
    print("\n=== Tool Composition ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return
    
    print("1. Creating composite tool system...")
    
    factory = DynamicToolFactory()
    
    # Create base tools
    calculator = factory.create_calculator_tool(["add", "sub", "mult", "div"])
    text_processor = factory.create_text_processor_tool(["word_count", "char_count", "uppercase"])
    json_to_csv = factory.create_data_transformer_tool("json", "csv")
    
    # Create a composite workflow tool
    @tool(name="data_analysis_workflow", description="Complete data analysis workflow")
    def data_analysis_workflow(json_data: str, analysis_type: str) -> str:
        """
        Perform complete data analysis workflow.
        
        Args:
            json_data: JSON data to analyze
            analysis_type: Type of analysis (summary, transform, etc.)
        
        Returns:
            Analysis results
        """
        try:
            import json
            
            # Parse data
            data = json.loads(json_data)
            results = []
            
            # Basic statistics
            if isinstance(data, list):
                results.append(f"Records: {len(data)}")
                
                # If numeric data, calculate statistics
                if all(isinstance(item, (int, float)) for item in data):
                    total = sum(data)
                    avg = total / len(data)
                    results.append(f"Total: {total}")
                    results.append(f"Average: {avg:.2f}")
                    results.append(f"Min: {min(data)}")
                    results.append(f"Max: {max(data)}")
                
                # If objects, analyze structure
                elif data and isinstance(data[0], dict):
                    fields = list(data[0].keys())
                    results.append(f"Fields: {', '.join(fields)}")
                    
                    # Analyze text fields
                    for field in fields:
                        if isinstance(data[0][field], str):
                            text_lens = [len(str(item[field])) for item in data if field in item]
                            avg_len = sum(text_lens) / len(text_lens) if text_lens else 0
                            results.append(f"{field} avg length: {avg_len:.1f} chars")
            
            return "Analysis Results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Analysis error: {e}"
    
    # Create agent with composed tools
    composite_agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_tools([
        calculator,
        text_processor, 
        json_to_csv,
        data_analysis_workflow
    ]).build()
    
    print("2. Testing composite workflow...")
    
    test_data = '''
    [
        {"name": "Alice Johnson", "age": 28, "department": "Engineering"},
        {"name": "Bob Smith", "age": 34, "department": "Marketing"}, 
        {"name": "Carol Davis", "age": 29, "department": "Engineering"}
    ]
    '''
    
    response = await composite_agent.run(f"""
    Please help me analyze this employee data: {test_data}
    
    I need:
    1. Complete data analysis using the workflow tool
    2. Transform the data to CSV format
    3. Calculate the average age
    4. Count total characters in all names combined
    
    Use the appropriate tools in sequence.
    """)
    
    print(f"Composite workflow response: {response}")


async def main():
    """Run all dynamic tool creation examples."""
    await demonstrate_basic_dynamic_tools()
    await demonstrate_runtime_tool_creation()
    await demonstrate_conditional_tool_creation()
    await demonstrate_tool_composition()
    
    print("\n=== Dynamic Tool Creation Examples Complete ===")
    print("\nKey Capabilities Demonstrated:")
    print("- Runtime tool generation based on requirements")
    print("- Conditional tool creation for different user profiles")
    print("- Tool composition for complex workflows")
    print("- Adaptive agent capabilities that grow with needs")
    print("- Factory pattern for systematic tool creation")
    print("\nUse Cases:")
    print("- Multi-tenant applications with user-specific tools")
    print("- Adaptive systems that evolve based on usage patterns")
    print("- Domain-specific tool creation for specialized tasks")
    print("- Plugin architectures with runtime capability extension")


if __name__ == "__main__":
    asyncio.run(main())