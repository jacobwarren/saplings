# Tool Factory

The Tool Factory in Saplings provides dynamic tool synthesis capabilities, allowing agents to create and use custom tools at runtime.

## Overview

The Tool Factory system consists of several key components:

- **ToolFactory**: Main class for generating and managing tools
- **ToolTemplate**: Template for generating tool code
- **ToolSpecification**: Specification for a tool to be generated
- **ToolValidator**: Validates generated code for security and quality
- **CodeSigner**: Signs generated code to ensure integrity
- **Sandbox**: Executes tools in a secure environment

This system enables the dynamic creation of tools with strong security guarantees, making it safe to use LLM-generated code in production environments.

## Core Concepts

### Tool Templates

Tool templates define the structure of generated tools. Each template has:

- **ID**: Unique identifier for the template
- **Name**: Human-readable name
- **Description**: Purpose of the template
- **Template Code**: Code with placeholders for parameters
- **Required Parameters**: Parameters that must be provided

Templates use placeholders like `{{function_name}}` and `{{code_body}}` that are replaced with values from the tool specification.

### Tool Specifications

Tool specifications define the tools to be generated. Each specification has:

- **ID**: Unique identifier for the tool
- **Name**: Human-readable name
- **Description**: Purpose of the tool
- **Template ID**: ID of the template to use
- **Parameters**: Values for the template placeholders
- **Metadata**: Additional information about the tool

The Tool Factory uses these specifications to generate tool code by filling in the template placeholders.

### Code Validation

The Tool Factory validates generated code to ensure it meets security and quality standards:

- **Syntax Checking**: Ensures the code is syntactically correct
- **Security Analysis**: Checks for dangerous imports, functions, and patterns
- **Quality Assessment**: Evaluates code quality metrics

The validation level can be configured based on the security requirements of the application.

### Code Signing

The Tool Factory can sign generated code to ensure its integrity and authenticity:

- **Basic Signing**: Calculates a hash of the code
- **Advanced Signing**: Creates a cryptographic signature using a private key

Signed code can be verified before execution to prevent tampering.

### Sandboxed Execution

The Tool Factory can execute tools in a secure sandbox to prevent malicious code from affecting the host system:

- **Local Sandbox**: Executes code in a separate process with limited permissions
- **Docker Sandbox**: Executes code in a Docker container
- **E2B Sandbox**: Executes code in a cloud-based sandbox

The sandbox type can be configured based on the security requirements of the application.

## API Reference

### ToolFactory

```python
class ToolFactory:
    def __init__(
        self,
        model: Optional[LLM] = None,
        config: Optional[ToolFactoryConfig] = None,
    ):
        """Initialize the tool factory."""

    def register_template(self, template: ToolTemplate) -> None:
        """Register a tool template."""

    def get_template(self, template_id: str) -> Optional[ToolTemplate]:
        """Get a template by ID."""

    def list_templates(self) -> List[str]:
        """List all registered template IDs."""

    async def generate_tool_code(self, spec: ToolSpecification) -> str:
        """Generate tool code from a specification."""

    async def create_tool(self, spec: ToolSpecification) -> Type[ToolPlugin]:
        """Create a tool from a specification."""

    def _validate_tool_code(self, code: str) -> Tuple[bool, str]:
        """Validate the generated code."""

    def _perform_security_checks(self, code: str) -> Tuple[bool, str]:
        """Perform security checks on the code."""

    def _create_tool_class(self, spec: ToolSpecification, code: str) -> Type[ToolPlugin]:
        """Create a tool class from the specification and code."""

    def _save_tool(self, spec: ToolSpecification, code: str) -> None:
        """Save the tool to disk."""

    async def _generate_code_with_llm(self, spec: ToolSpecification) -> str:
        """Generate code using the LLM."""
```

### ToolTemplate

```python
class ToolTemplate(BaseModel):
    id: str  # Unique identifier for the template
    name: str  # Human-readable name for the template
    description: str  # Description of the template's purpose
    template_code: str  # Template code with placeholders
    required_parameters: List[str]  # List of required parameters for the template
    metadata: Dict[str, Any] = {}  # Additional metadata
```

### ToolSpecification

```python
class ToolSpecification(BaseModel):
    id: str  # Unique identifier for the tool
    name: str  # Human-readable name for the tool
    description: str  # Description of the tool's purpose
    template_id: str  # ID of the template to use
    parameters: Dict[str, Any]  # Parameters for the template
    metadata: Dict[str, Any] = {}  # Additional metadata
```

### ToolFactoryConfig

```python
class ToolFactoryConfig(BaseModel):
    output_dir: str = "tools"  # Directory for generated tools
    security_level: SecurityLevel = SecurityLevel.MEDIUM  # Security level for tool generation
    enable_code_signing: bool = False  # Whether to enable code signing (deprecated)
    signing_level: SigningLevel = SigningLevel.NONE  # Level of code signing for tools
    signing_key_path: Optional[str] = None  # Path to the signing key file
    sandbox_type: SandboxType = SandboxType.NONE  # Type of sandbox to use for tool execution
    docker_image: Optional[str] = "python:3.9-slim"  # Docker image for sandboxed execution
    e2b_api_key: Optional[str] = None  # E2B API key for cloud sandbox
    sandbox_timeout: int = 30  # Timeout in seconds for sandboxed execution
    metadata: Dict[str, Any] = {}  # Additional metadata
```

### Enums

```python
class SecurityLevel(str, Enum):
    """Security level for tool generation."""
    LOW = "low"  # Basic security checks
    MEDIUM = "medium"  # Standard security checks
    HIGH = "high"  # Strict security checks

class SandboxType(str, Enum):
    """Type of sandbox for tool execution."""
    NONE = "none"  # No sandboxing
    DOCKER = "docker"  # Docker-based sandbox
    E2B = "e2b"  # E2B-based sandbox

class SigningLevel(str, Enum):
    """Level of code signing for tools."""
    NONE = "none"  # No code signing
    BASIC = "basic"  # Basic code signing (hash verification)
    ADVANCED = "advanced"  # Advanced code signing (cryptographic signatures)
```

## Usage Examples

### Basic Usage

```python
from saplings.core.model_adapter import LLM
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, ToolTemplate, ToolSpecification
)

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a tool factory
tool_factory = ToolFactory(
    model=model,
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level="medium",
    )
)

# Register a template
template = ToolTemplate(
    id="python_tool",
    name="Python Tool",
    description="A generic Python tool",
    template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
    required_parameters=["function_name", "parameters", "description", "code_body"],
)
tool_factory.register_template(template)

# Create a tool specification
spec = ToolSpecification(
    id="data_visualizer",
    name="Data Visualizer",
    description="Creates visualizations from data",
    template_id="python_tool",
    parameters={
        "function_name": "visualize_data",
        "parameters": "data: dict, output_path: str",
        "description": "Creates a bar chart visualization from data",
        "code_body": """
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
df.plot(kind='bar')
plt.savefig(output_path)
return output_path
"""
    }
)

# Create the tool
import asyncio
tool_class = asyncio.run(tool_factory.create_tool(spec))

# Use the tool
tool = tool_class()
result = tool.execute(
    data={"A": [1, 2, 3], "B": [4, 5, 6]},
    output_path="./chart.png"
)
print(f"Chart saved to: {result}")
```

### Secure Tool Creation

```python
from saplings.core.model_adapter import LLM
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, ToolTemplate, ToolSpecification,
    SecurityLevel, SandboxType, SigningLevel
)

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a tool factory with high security
tool_factory = ToolFactory(
    model=model,
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level=SecurityLevel.HIGH,
        signing_level=SigningLevel.ADVANCED,
        signing_key_path="./keys/private_key.pem",
        sandbox_type=SandboxType.DOCKER,
        docker_image="python:3.9-slim",
        sandbox_timeout=30,
    )
)

# Register a template
template = ToolTemplate(
    id="data_analysis_tool",
    name="Data Analysis Tool",
    description="A tool for data analysis",
    template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
    required_parameters=["function_name", "parameters", "description", "code_body"],
)
tool_factory.register_template(template)

# Create a tool specification
spec = ToolSpecification(
    id="correlation_analyzer",
    name="Correlation Analyzer",
    description="Analyzes correlations between variables in a dataset",
    template_id="data_analysis_tool",
    parameters={
        "function_name": "analyze_correlations",
        "parameters": "data: dict, variables: list = None",
        "description": "Calculate correlation coefficients between variables",
        "code_body": """
import numpy as np
import pandas as pd

# Convert input to DataFrame
df = pd.DataFrame(data)

# Select variables if specified
if variables:
    df = df[variables]

# Calculate correlation matrix
corr_matrix = df.corr()

# Convert to dictionary for return
result = {
    "correlation_matrix": corr_matrix.to_dict(),
    "strongest_correlation": {
        "variables": None,
        "value": 0
    }
}

# Find strongest correlation
if len(corr_matrix) > 1:
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)

    # Apply the mask to get only the upper triangle
    upper_corr = corr_matrix.where(mask)

    # Find the maximum absolute correlation
    max_corr = upper_corr.abs().max().max()

    # Find the variables with the maximum correlation
    max_loc = upper_corr.abs().stack().idxmax()

    result["strongest_correlation"]["variables"] = list(max_loc)
    result["strongest_correlation"]["value"] = corr_matrix.loc[max_loc]

return result
"""
    }
)

# Create the tool
import asyncio
tool_class = asyncio.run(tool_factory.create_tool(spec))

# Use the tool
tool = tool_class()
result = tool.execute(
    data={
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "C": [1, 3, 5, 7, 9]
    }
)
print(f"Correlation analysis: {result}")
```

### LLM-Generated Code

```python
from saplings.core.model_adapter import LLM
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, ToolTemplate, ToolSpecification
)

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a tool factory
tool_factory = ToolFactory(
    model=model,
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level="medium",
    )
)

# Register a template
template = ToolTemplate(
    id="python_tool",
    name="Python Tool",
    description="A generic Python tool",
    template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
    required_parameters=["function_name", "parameters", "description", "code_body"],
)
tool_factory.register_template(template)

# Create a tool specification with no code_body (will be generated by LLM)
spec = ToolSpecification(
    id="text_summarizer",
    name="Text Summarizer",
    description="Summarizes text using extractive summarization",
    template_id="python_tool",
    parameters={
        "function_name": "summarize_text",
        "parameters": "text: str, num_sentences: int = 3",
        "description": "Summarize text by extracting the most important sentences",
        # No code_body provided - will be generated by LLM
    }
)

# Create the tool (code will be generated by LLM)
import asyncio
tool_class = asyncio.run(tool_factory.create_tool(spec))

# Use the tool
tool = tool_class()
result = tool.execute(
    text="Saplings is a graph-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day. It combines vector storage with graph-based memory, cascaded retrieval, budget-aware planning, and self-healing capabilities to create agents that are more efficient, grounded, and capable than traditional RAG systems. Saplings agents improve over time through a self-improvement loop that identifies errors, generates patches, and fine-tunes models using Low-Rank Adaptation (LoRA).",
    num_sentences=2
)
print(f"Summary: {result}")
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, ToolTemplate, ToolSpecification
)
from saplings.integration import (
    SecureHotLoader, SecureHotLoaderConfig, IntegrationManager
)

# Create a model
model = LLM.create("openai", "gpt-4o")

# Create a tool factory
tool_factory = ToolFactory(
    model=model,
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level="medium",
    )
)

# Register a template
template = ToolTemplate(
    id="python_tool",
    name="Python Tool",
    description="A generic Python tool",
    template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
    required_parameters=["function_name", "parameters", "description", "code_body"],
)
tool_factory.register_template(template)

# Create a tool specification
spec = ToolSpecification(
    id="data_visualizer",
    name="Data Visualizer",
    description="Creates visualizations from data",
    template_id="python_tool",
    parameters={
        "function_name": "visualize_data",
        "parameters": "data: dict, output_path: str",
        "description": "Creates a bar chart visualization from data",
        "code_body": """
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
df.plot(kind='bar')
plt.savefig(output_path)
return output_path
"""
    }
)

# Create the tool
import asyncio
tool_class = asyncio.run(tool_factory.create_tool(spec))

# Create a secure hot loader
hot_loader = SecureHotLoader(
    config=SecureHotLoaderConfig(
        watch_directories=["./tools"],
        auto_reload=True,
    )
)

# Load the tool into the hot loader
hot_loader.load_tool(tool_class)

# Create an agent with the hot loader
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=hot_loader.get_tools(),
    )
)

# Run a task that uses the tool
result = asyncio.run(agent.run(
    "Create a bar chart visualization of the following data: "
    "{'Sales': [10, 20, 30, 40], 'Expenses': [5, 15, 25, 35]}"
))
print(result)
```

## Advanced Features

### Code Signing

The Tool Factory can sign generated code to ensure its integrity and authenticity:

```python
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, SigningLevel, generate_key_pair
)

# Generate a key pair for code signing
private_key_path = "./keys/private_key.pem"
public_key_path = "./keys/public_key.pem"
generate_key_pair(private_key_path, public_key_path)

# Create a tool factory with advanced code signing
tool_factory = ToolFactory(
    config=ToolFactoryConfig(
        output_dir="./tools",
        signing_level=SigningLevel.ADVANCED,
        signing_key_path=private_key_path,
    )
)

# Now when tools are created, they will be signed with the private key
# and verified with the public key before execution
```

### Sandboxed Execution

The Tool Factory can execute tools in a secure sandbox to prevent malicious code from affecting the host system:

```python
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, SandboxType
)

# Create a tool factory with Docker sandboxing
tool_factory = ToolFactory(
    config=ToolFactoryConfig(
        output_dir="./tools",
        sandbox_type=SandboxType.DOCKER,
        docker_image="python:3.9-slim",
        sandbox_timeout=30,
    )
)

# Now when tools are executed, they will run in a Docker container
# with no network access and limited capabilities
```

### E2B Cloud Sandbox

For even stronger isolation, you can use the E2B cloud sandbox:

```python
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, SandboxType
)

# Create a tool factory with E2B sandboxing
tool_factory = ToolFactory(
    config=ToolFactoryConfig(
        output_dir="./tools",
        sandbox_type=SandboxType.E2B,
        e2b_api_key="your_e2b_api_key",
        sandbox_timeout=30,
    )
)

# Now when tools are executed, they will run in an E2B cloud sandbox
# completely isolated from your system
```

### Custom Tool Templates

You can create custom tool templates for different types of tools:

```python
from saplings.tool_factory import ToolTemplate

# Create a template for data processing tools
data_processing_template = ToolTemplate(
    id="data_processing_tool",
    name="Data Processing Tool",
    description="A tool for processing data",
    template_code="""
import pandas as pd
import numpy as np

def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    # Convert input to DataFrame
    df = pd.DataFrame({{input_parameter}})

    # Process the data
    {{code_body}}

    # Return the result
    return result
""",
    required_parameters=[
        "function_name",
        "parameters",
        "description",
        "input_parameter",
        "code_body",
    ],
)

# Create a template for API tools
api_tool_template = ToolTemplate(
    id="api_tool",
    name="API Tool",
    description="A tool for making API requests",
    template_code="""
import requests
import json

def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    # Set up the request
    url = "{{api_url}}"
    headers = {{headers}}

    # Make the request
    {{code_body}}

    # Process the response
    {{response_processing}}

    # Return the result
    return result
""",
    required_parameters=[
        "function_name",
        "parameters",
        "description",
        "api_url",
        "headers",
        "code_body",
        "response_processing",
    ],
)
```

### Tool Validation

The Tool Factory validates generated code to ensure it meets security and quality standards:

```python
from saplings.tool_factory import (
    ToolFactory, ToolFactoryConfig, SecurityLevel, ToolValidator
)

# Create a tool factory with high security
tool_factory = ToolFactory(
    config=ToolFactoryConfig(
        output_dir="./tools",
        security_level=SecurityLevel.HIGH,
    )
)

# You can also use the validator directly
validator = ToolValidator(
    config=ToolFactoryConfig(
        security_level=SecurityLevel.HIGH,
    )
)

# Validate some code
code = """
def analyze_data(data):
    import os  # This will be flagged as dangerous
    import subprocess  # This will be flagged as dangerous

    # This will be flagged as dangerous
    os.system("rm -rf /")

    return data
"""

validation_result = validator.validate(code)
print(f"Is valid: {validation_result.is_valid}")
print(f"Error: {validation_result.error_message}")
print(f"Warnings: {validation_result.warnings}")
```

## Implementation Details

### Tool Generation Process

The tool generation process works as follows:

1. **Template Selection**: Select a template based on the specification's template ID
2. **Parameter Validation**: Ensure all required parameters are provided
3. **Code Generation**: If `code_body` is missing, generate it using the LLM
4. **Template Filling**: Replace placeholders in the template with parameter values
5. **Code Validation**: Validate the generated code for security and quality
6. **Tool Creation**: Create a tool class from the generated code
7. **Code Signing**: Sign the code if enabled
8. **Tool Registration**: Register the tool for later use
9. **Tool Saving**: Save the tool to disk

### Code Validation Process

The code validation process works as follows:

1. **Syntax Checking**: Parse the code to ensure it's syntactically correct
2. **Security Analysis**:
   - Check for dangerous imports (`os`, `sys`, `subprocess`, etc.)
   - Check for dangerous functions (`eval`, `exec`, `compile`, etc.)
   - Check for network access patterns
   - Perform AST-based security checks
3. **Quality Assessment**:
   - Check for long lines
   - Check for too many nested blocks
   - Check for other quality issues

### Sandboxed Execution Process

The sandboxed execution process works as follows:

1. **Code Preparation**: Prepare the code for execution in the sandbox
2. **Sandbox Creation**: Create a sandbox environment based on the configuration
3. **Code Execution**: Execute the code in the sandbox
4. **Result Retrieval**: Retrieve the result from the sandbox
5. **Sandbox Cleanup**: Clean up the sandbox environment

#### Docker Sandbox

The Docker sandbox process works as follows:

1. **Container Creation**: Create a Docker container with the specified image
2. **Code Mounting**: Mount the code and input data into the container
3. **Code Execution**: Execute the code in the container
4. **Result Retrieval**: Retrieve the result from the container
5. **Container Cleanup**: Remove the container

#### E2B Sandbox

The E2B sandbox process works as follows:

1. **Session Creation**: Create an E2B session with the specified template
2. **Code Upload**: Upload the code to the session
3. **Code Execution**: Execute the code in the session
4. **Result Retrieval**: Retrieve the result from the session
5. **Session Cleanup**: Close the session

### Code Signing Process

The code signing process works as follows:

1. **Hash Calculation**: Calculate a hash of the code
2. **Signature Generation**:
   - For basic signing, store the hash
   - For advanced signing, sign the hash with a private key
3. **Signature Storage**: Store the signature with the code

### Signature Verification Process

The signature verification process works as follows:

1. **Hash Calculation**: Calculate a hash of the code
2. **Signature Verification**:
   - For basic signing, compare the hash with the stored hash
   - For advanced signing, verify the signature with a public key
3. **Verification Result**: Return whether the signature is valid

## Extension Points

The Tool Factory system is designed to be extensible:

### Custom Validator

You can create a custom validator by extending the `ToolValidator` class:

```python
from saplings.tool_factory import ToolValidator, ValidationResult

class CustomValidator(ToolValidator):
    def validate(self, code: str) -> ValidationResult:
        # Call the parent validator
        result = super().validate(code)

        # Add custom validation logic
        if "import tensorflow" in code:
            result.warnings.append("TensorFlow is a large dependency")

        if "while True:" in code:
            result.is_valid = False
            result.error_message = "Infinite loops are not allowed"

        return result
```

### Custom Sandbox

You can create a custom sandbox by extending the `Sandbox` class:

```python
from saplings.tool_factory import Sandbox

class CustomSandbox(Sandbox):
    async def execute(
        self,
        code: str,
        function_name: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        # Implement custom sandbox execution logic
        # ...

        return result
```

### Custom Code Signer

You can create a custom code signer by extending the `CodeSigner` class:

```python
from saplings.tool_factory import CodeSigner

class CustomCodeSigner(CodeSigner):
    def sign(self, code: str) -> Dict[str, str]:
        # Implement custom code signing logic
        # ...

        return signature_info
```

## Conclusion

The Tool Factory in Saplings provides a powerful system for dynamic tool synthesis with strong security guarantees. By combining templates, code generation, validation, signing, and sandboxed execution, it enables the safe use of LLM-generated code in production environments. This makes it possible to create and use custom tools at runtime, greatly expanding the capabilities of Saplings agents.
