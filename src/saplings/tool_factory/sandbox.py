"""
Sandbox module for Saplings tool factory.

This module provides sandbox execution environments for safely running
dynamically generated tools.
"""

import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from saplings.tool_factory.config import SandboxType, ToolFactoryConfig

logger = logging.getLogger(__name__)


class Sandbox(ABC):
    """
    Abstract base class for sandbox execution environments.
    
    This class defines the interface for sandbox execution environments.
    Concrete implementations should handle the details of setting up and
    running code in a secure environment.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the sandbox.
        
        Args:
            config: Configuration for the sandbox
        """
        self.config = config or ToolFactoryConfig()
    
    @abstractmethod
    async def execute(
        self, code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute code in the sandbox.
        
        Args:
            code: Code to execute
            function_name: Name of the function to call
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function call
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the sandbox."""
        pass


class LocalSandbox(Sandbox):
    """
    Local sandbox execution environment.
    
    This sandbox executes code in the local Python interpreter with minimal
    isolation. It should only be used for trusted code or in development.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the local sandbox.
        
        Args:
            config: Configuration for the sandbox
        """
        super().__init__(config)
        self._temp_files: List[str] = []
    
    async def execute(
        self, code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute code in the local Python interpreter.
        
        Args:
            code: Code to execute
            function_name: Name of the function to call
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function call
        """
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name
            self._temp_files.append(temp_path)
        
        try:
            # Create a namespace for the code
            namespace = {}
            
            # Execute the code in the namespace
            with open(temp_path, "r") as f:
                exec(f.read(), namespace)
            
            # Get the function
            if function_name not in namespace:
                raise ValueError(f"Function '{function_name}' not found in the code")
            
            function = namespace[function_name]
            
            # Call the function
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing code in local sandbox: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_path in self._temp_files:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file {temp_path}: {e}")
        
        self._temp_files = []


class DockerSandbox(Sandbox):
    """
    Docker-based sandbox execution environment.
    
    This sandbox executes code in a Docker container for isolation.
    It requires Docker to be installed and available on the system.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the Docker sandbox.
        
        Args:
            config: Configuration for the sandbox
        """
        super().__init__(config)
        self._temp_dir = tempfile.mkdtemp(prefix="saplings_docker_sandbox_")
        self._container_id = None
        
        # Check if Docker is available
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Docker is not available: {e}")
    
    async def execute(
        self, code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute code in a Docker container.
        
        Args:
            code: Code to execute
            function_name: Name of the function to call
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function call
        """
        import json
        
        # Create the code file
        code_path = os.path.join(self._temp_dir, "code.py")
        with open(code_path, "w") as f:
            f.write(code)
        
        # Create a wrapper script to call the function
        wrapper_path = os.path.join(self._temp_dir, "wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(f"""
import json
import sys
from code import {function_name}

# Load arguments from stdin
input_data = json.loads(sys.stdin.read())
args = input_data["args"]
kwargs = input_data["kwargs"]

# Call the function
result = {function_name}(*args, **kwargs)

# Return the result as JSON
print(json.dumps({{"result": result}}))
""")
        
        # Create the input data
        input_data = {
            "args": args,
            "kwargs": kwargs,
        }
        
        # Create the input file
        input_path = os.path.join(self._temp_dir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_data, f)
        
        # Run the Docker container
        docker_image = self.config.docker_image or "python:3.9-slim"
        timeout = self.config.sandbox_timeout or 30
        
        try:
            # Create and start the container
            container_id = subprocess.check_output(
                [
                    "docker", "run",
                    "-d",  # Detached mode
                    "--rm",  # Remove container when it exits
                    "-v", f"{self._temp_dir}:/workspace",  # Mount the workspace
                    "--workdir", "/workspace",  # Set the working directory
                    "--network", "none",  # Disable network access
                    "--cap-drop", "ALL",  # Drop all capabilities
                    "--security-opt", "no-new-privileges",  # Prevent privilege escalation
                    docker_image,
                    "sleep", str(timeout),  # Keep the container running
                ],
                stderr=subprocess.PIPE,
            ).decode().strip()
            
            self._container_id = container_id
            
            # Execute the wrapper script
            result = subprocess.check_output(
                [
                    "docker", "exec",
                    container_id,
                    "python", "wrapper.py",
                ],
                input=json.dumps(input_data).encode(),
                stderr=subprocess.PIPE,
                timeout=timeout,
            ).decode().strip()
            
            # Parse the result
            try:
                result_data = json.loads(result)
                return result_data["result"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid result from Docker sandbox: {e}")
        
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Docker sandbox execution timed out after {timeout} seconds")
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Docker sandbox execution failed: {e.stderr.decode()}")
        
        finally:
            # Stop and remove the container
            if self._container_id:
                try:
                    subprocess.run(
                        ["docker", "stop", self._container_id],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except Exception as e:
                    logger.warning(f"Error stopping Docker container: {e}")
                
                self._container_id = None
    
    def cleanup(self) -> None:
        """Clean up the Docker sandbox."""
        # Stop the container if it's still running
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                logger.warning(f"Error stopping Docker container: {e}")
            
            self._container_id = None
        
        # Clean up the temporary directory
        try:
            import shutil
            shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")


class E2BSandbox(Sandbox):
    """
    E2B-based sandbox execution environment.
    
    This sandbox executes code in an E2B cloud sandbox for maximum isolation.
    It requires an E2B API key to be configured.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the E2B sandbox.
        
        Args:
            config: Configuration for the sandbox
        """
        super().__init__(config)
        self._session = None
        
        # Check if E2B API key is configured
        if not self.config.e2b_api_key:
            raise ValueError("E2B API key is required for E2B sandbox")
        
        # Check if E2B is available
        try:
            import e2b
        except ImportError:
            raise ImportError("E2B package is not installed. Install it with: pip install e2b")
    
    async def execute(
        self, code: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute code in an E2B cloud sandbox.
        
        Args:
            code: Code to execute
            function_name: Name of the function to call
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function call
        """
        import json
        import e2b
        
        # Set the API key
        e2b.api_key = self.config.e2b_api_key
        
        # Create a new session
        self._session = await e2b.Session.create(template="python3")
        
        try:
            # Write the code to a file
            await self._session.filesystem.write(
                path="/tmp/code.py",
                content=code,
            )
            
            # Create a wrapper script to call the function
            wrapper_code = f"""
import json
import sys
from code import {function_name}

# Load arguments from stdin
input_data = json.loads(sys.stdin.read())
args = input_data["args"]
kwargs = input_data["kwargs"]

# Call the function
result = {function_name}(*args, **kwargs)

# Return the result as JSON
print(json.dumps({{"result": result}}))
"""
            
            await self._session.filesystem.write(
                path="/tmp/wrapper.py",
                content=wrapper_code,
            )
            
            # Create the input data
            input_data = {
                "args": args,
                "kwargs": kwargs,
            }
            
            # Execute the wrapper script
            process = await self._session.process.start({
                "cmd": "python /tmp/wrapper.py",
                "stdin": json.dumps(input_data),
                "timeout": self.config.sandbox_timeout * 1000,  # Convert to milliseconds
            })
            
            # Wait for the process to complete
            exit_code = await process.exit_code
            
            if exit_code != 0:
                stderr = await process.stderr.read_all()
                raise RuntimeError(f"E2B sandbox execution failed with exit code {exit_code}: {stderr}")
            
            # Get the output
            stdout = await process.stdout.read_all()
            
            # Parse the result
            try:
                result_data = json.loads(stdout)
                return result_data["result"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid result from E2B sandbox: {e}")
        
        finally:
            # Clean up the session
            if self._session:
                await self._session.close()
                self._session = None
    
    def cleanup(self) -> None:
        """Clean up the E2B sandbox."""
        # The session is automatically closed in the execute method,
        # but we'll add this method for consistency with the interface
        pass


def get_sandbox(config: Optional[ToolFactoryConfig] = None) -> Sandbox:
    """
    Get a sandbox instance based on configuration.
    
    Args:
        config: Configuration for the sandbox
        
    Returns:
        Sandbox: Sandbox instance
    """
    config = config or ToolFactoryConfig()
    
    if config.sandbox_type == SandboxType.DOCKER:
        return DockerSandbox(config)
    elif config.sandbox_type == SandboxType.E2B:
        return E2BSandbox(config)
    else:
        return LocalSandbox(config)
