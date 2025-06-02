#!/usr/bin/env python3
"""
Code Repository Assistant Example

This example demonstrates creating an intelligent code repository assistant that can
analyze codebases, understand project structure, and help with development tasks.
"""

import asyncio
import os
import glob
from pathlib import Path
from saplings import AgentBuilder
from saplings.api.tools import tool


@tool(name="file_reader", description="Reads the contents of a file")
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        The contents of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool(name="directory_scanner", description="Scans a directory for files and subdirectories")
def scan_directory(directory_path: str, extensions: str = "py,js,ts,java,cpp,h") -> dict:
    """
    Scan a directory for code files.
    
    Args:
        directory_path: Path to the directory to scan
        extensions: Comma-separated list of file extensions to include
    
    Returns:
        Dictionary with file structure and statistics
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return {"error": f"Directory {directory_path} does not exist"}
        
        ext_list = [f".{ext.strip()}" for ext in extensions.split(",")]
        files = []
        
        for ext in ext_list:
            files.extend(glob.glob(f"{directory_path}/**/*{ext}", recursive=True))
        
        # Get file statistics
        total_files = len(files)
        total_lines = 0
        file_info = []
        
        for file_path in files[:50]:  # Limit to first 50 files for demo
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    file_info.append({
                        "path": file_path,
                        "lines": lines,
                        "size_kb": os.path.getsize(file_path) / 1024
                    })
            except:
                continue
        
        return {
            "directory": directory_path,
            "total_files": total_files,
            "total_lines": total_lines,
            "files": file_info,
            "extensions": ext_list
        }
    except Exception as e:
        return {"error": f"Error scanning directory: {e}"}


@tool(name="code_analyzer", description="Analyzes code for patterns, complexity, and structure")
def analyze_code(code: str, language: str = "python") -> dict:
    """
    Analyze code for basic metrics and patterns.
    
    Args:
        code: The code to analyze
        language: Programming language (python, javascript, java, etc.)
    
    Returns:
        Analysis results including complexity metrics
    """
    try:
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # Basic complexity analysis
        if language.lower() == "python":
            functions = len([line for line in lines if line.strip().startswith('def ')])
            classes = len([line for line in lines if line.strip().startswith('class ')])
            imports = len([line for line in lines if line.strip().startswith(('import ', 'from '))])
            
            # Cyclomatic complexity approximation
            complexity_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or']
            complexity = sum(line.count(keyword) for line in lines for keyword in complexity_keywords)
            
            return {
                "language": language,
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "estimated_complexity": complexity,
                "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0
            }
        
        # Generic analysis for other languages
        return {
            "language": language,
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "estimated_complexity": "unknown"
        }
        
    except Exception as e:
        return {"error": f"Error analyzing code: {e}"}


async def setup_code_assistant():
    """Set up the code repository assistant with necessary tools."""
    print("=== Setting Up Code Repository Assistant ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return None
    
    print("1. Creating code-specialized agent...")
    
    # Create agent optimized for code analysis
    assistant = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        gasa_enabled=True,  # GASA helps with understanding code relationships
        gasa_strategy="binary",
        memory_path="./code_assistant_memory"
    ).with_tools([
        read_file,
        scan_directory,
        analyze_code
    ]).build()
    
    print("2. Setting up coding knowledge base...")
    
    # Add programming best practices and patterns
    coding_knowledge = [
        {
            "content": "SOLID principles in object-oriented design: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.",
            "metadata": {"topic": "design_patterns", "category": "best_practices"}
        },
        {
            "content": "Code review best practices: Check for readability, maintainability, performance, security vulnerabilities, and proper error handling.",
            "metadata": {"topic": "code_review", "category": "process"}
        },
        {
            "content": "Python PEP 8 style guide emphasizes readable code with consistent formatting, naming conventions, and documentation.",
            "metadata": {"topic": "python_style", "category": "standards"}
        },
        {
            "content": "Git workflow best practices: Use feature branches, write clear commit messages, perform code reviews, and maintain a clean history.",
            "metadata": {"topic": "version_control", "category": "workflow"}
        },
        {
            "content": "Testing strategies: Unit tests for individual functions, integration tests for component interactions, and end-to-end tests for complete workflows.",
            "metadata": {"topic": "testing", "category": "quality_assurance"}
        }
    ]
    
    for knowledge in coding_knowledge:
        await assistant.add_document(
            content=knowledge["content"],
            metadata=knowledge["metadata"]
        )
    
    print("3. Code assistant ready!")
    return assistant


async def demonstrate_repository_analysis():
    """Demonstrate analyzing a code repository."""
    print("\n=== Repository Analysis Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Analyzing the examples directory...")
    
    # Analyze the examples directory we're creating
    response = await assistant.run("""
    Please scan the 'examples' directory and analyze its structure. Then:
    1. Provide an overview of the files found
    2. Identify any patterns in the code organization
    3. Suggest improvements for code structure or documentation
    
    Use the directory_scanner tool to get started.
    """)
    
    print(f"Repository analysis: {response}")


async def demonstrate_code_review():
    """Demonstrate code review capabilities."""
    print("\n=== Code Review Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Creating sample code for review...")
    
    # Sample code with potential issues
    sample_code = '''
def calculate_discount(price, customer_type, items):
    if customer_type == "premium":
        if items > 10:
            discount = 0.2
        else:
            discount = 0.1
    elif customer_type == "regular":
        if items > 5:
            discount = 0.15
        else:
            discount = 0.05
    else:
        discount = 0
    
    final_price = price - (price * discount)
    return final_price

def process_order(order):
    total = 0
    for item in order:
        total += item["price"] * item["quantity"]
    
    # Apply discount
    customer = order[0]["customer"]
    item_count = len(order)
    final_total = calculate_discount(total, customer, item_count)
    
    return final_total
'''
    
    print("2. Performing code review...")
    
    response = await assistant.run(f"""
    Please review this Python code and provide feedback on:
    1. Code quality and readability
    2. Potential bugs or edge cases
    3. Adherence to best practices
    4. Suggestions for improvement
    
    Code to review:
    ```python
    {sample_code}
    ```
    
    Use the code_analyzer tool to get detailed metrics first.
    """)
    
    print(f"Code review results: {response}")


async def demonstrate_architecture_guidance():
    """Demonstrate architecture and design guidance."""
    print("\n=== Architecture Guidance Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Requesting architecture guidance...")
    
    response = await assistant.run("""
    I'm building a web application that needs to:
    - Handle user authentication and authorization
    - Process file uploads and store them securely
    - Send email notifications
    - Generate reports from database data
    - Scale to handle 10,000 concurrent users
    
    Please provide:
    1. A recommended architecture pattern
    2. Technology stack suggestions
    3. Security considerations
    4. Scalability strategies
    5. Testing approach
    
    Base your recommendations on software engineering best practices.
    """)
    
    print(f"Architecture guidance: {response}")


async def demonstrate_debugging_help():
    """Demonstrate debugging assistance."""
    print("\n=== Debugging Assistance Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Analyzing a buggy code example...")
    
    buggy_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def calculate_fibonacci_sum(max_num):
    total = 0
    for i in range(max_num):
        fib_value = fibonacci(i)
        total += fib_value
    return total

# This function is very slow for large inputs
result = calculate_fibonacci_sum(35)
print(f"Sum: {result}")
'''
    
    response = await assistant.run(f"""
    This code is running very slowly. Please:
    1. Analyze the code to identify performance issues
    2. Explain why it's slow
    3. Provide an optimized version
    4. Include time complexity analysis
    
    Problematic code:
    ```python
    {buggy_code}
    ```
    
    Use the code_analyzer tool to examine the structure first.
    """)
    
    print(f"Debugging analysis: {response}")


async def demonstrate_documentation_generation():
    """Demonstrate documentation generation."""
    print("\n=== Documentation Generation Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Generating documentation for code...")
    
    undocumented_code = '''
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process(self, data):
        if data["id"] in self.cache:
            return self.cache[data["id"]]
        
        result = self._transform(data)
        result = self._validate(result)
        
        self.cache[data["id"]] = result
        return result
    
    def _transform(self, data):
        return {
            "processed_value": data["value"] * self.config["multiplier"],
            "timestamp": data.get("timestamp", "unknown"),
            "category": self._categorize(data["value"])
        }
    
    def _validate(self, data):
        if data["processed_value"] < 0:
            raise ValueError("Processed value cannot be negative")
        return data
    
    def _categorize(self, value):
        if value < 10:
            return "low"
        elif value < 100:
            return "medium"
        else:
            return "high"
'''
    
    response = await assistant.run(f"""
    Please generate comprehensive documentation for this Python class:
    1. Class-level docstring with purpose and usage examples
    2. Method docstrings with parameters, return values, and exceptions
    3. Type hints where appropriate
    4. README section explaining how to use the class
    
    Code to document:
    ```python
    {undocumented_code}
    ```
    """)
    
    print(f"Generated documentation: {response}")


async def demonstrate_refactoring_suggestions():
    """Demonstrate refactoring suggestions."""
    print("\n=== Refactoring Suggestions Demo ===\n")
    
    assistant = await setup_code_assistant()
    if not assistant:
        return
    
    print("1. Analyzing code for refactoring opportunities...")
    
    legacy_code = '''
def handle_user_request(request_type, user_data, additional_params):
    if request_type == "create_user":
        if user_data["name"] == "" or user_data["email"] == "":
            return {"error": "Name and email required"}
        
        if "@" not in user_data["email"]:
            return {"error": "Invalid email"}
        
        # Create user logic
        user_id = len(users) + 1
        users.append({
            "id": user_id,
            "name": user_data["name"],
            "email": user_data["email"],
            "created_at": datetime.now()
        })
        return {"success": True, "user_id": user_id}
    
    elif request_type == "update_user":
        user_id = additional_params["user_id"]
        for user in users:
            if user["id"] == user_id:
                if user_data.get("name"):
                    user["name"] = user_data["name"]
                if user_data.get("email"):
                    if "@" not in user_data["email"]:
                        return {"error": "Invalid email"}
                    user["email"] = user_data["email"]
                return {"success": True}
        return {"error": "User not found"}
    
    elif request_type == "delete_user":
        user_id = additional_params["user_id"]
        for i, user in enumerate(users):
            if user["id"] == user_id:
                del users[i]
                return {"success": True}
        return {"error": "User not found"}
    
    else:
        return {"error": "Unknown request type"}
'''
    
    response = await assistant.run(f"""
    Please analyze this legacy code and provide refactoring suggestions:
    1. Identify code smells and issues
    2. Suggest design patterns that could improve the structure
    3. Recommend how to separate concerns
    4. Provide a refactored version following SOLID principles
    5. Suggest how to make it more testable
    
    Legacy code to refactor:
    ```python
    {legacy_code}
    ```
    
    Use code analysis tools to examine the structure first.
    """)
    
    print(f"Refactoring suggestions: {response}")


async def main():
    """Run all code repository assistant examples."""
    await demonstrate_repository_analysis()
    await demonstrate_code_review()
    await demonstrate_architecture_guidance()
    await demonstrate_debugging_help()
    await demonstrate_documentation_generation()
    await demonstrate_refactoring_suggestions()
    
    print("\n=== Code Repository Assistant Demo Complete ===")
    print("\nCapabilities demonstrated:")
    print("- Repository structure analysis")
    print("- Automated code review with suggestions")
    print("- Architecture and design guidance")
    print("- Performance debugging assistance")
    print("- Documentation generation")
    print("- Refactoring recommendations")
    print("\nThe assistant can help with:")
    print("- Code quality improvement")
    print("- Best practices enforcement")
    print("- Technical debt reduction")
    print("- Developer productivity enhancement")


if __name__ == "__main__":
    asyncio.run(main())