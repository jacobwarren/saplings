from __future__ import annotations

"""
ToolFactory module for Saplings.

This module provides the ToolFactory class for dynamic tool synthesis.
"""


import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

from saplings.core._internal.plugin import PluginType, ToolPlugin, register_plugin
from saplings.tool_factory._internal.factory.config import (
    SandboxType,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
)

# Lazy imports to avoid circular dependencies
# These will be imported on-demand when needed
_code_signing = None
_tool_validator = None
_sandbox = None

if TYPE_CHECKING:
    from saplings.core._internal.model_adapter import LLM
    from saplings.tool_factory._internal.sandbox.sandbox import Sandbox
    from saplings.tool_factory._internal.security.code_signing import CodeSigner, SignatureVerifier
    from saplings.tool_factory._internal.security.tool_validator import ToolValidator

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Factory for dynamic tool synthesis.

    This class provides functionality for:
    - Registering tool templates
    - Generating tool code from specifications
    - Validating and securing generated code
    - Creating and registering tool plugins
    """

    # Class-level lock for thread safety during lazy initialization
    _init_lock = threading.RLock()

    # Singleton instance for global access
    _instance: Optional["ToolFactory"] = None

    @classmethod
    def get_instance(
        cls,
        model: Optional["LLM"] = None,
        executor=None,
        config: Optional[ToolFactoryConfig] = None,
    ) -> "ToolFactory":
        """
        Get or create the singleton instance of ToolFactory.

        This method provides a thread-safe way to get the global ToolFactory instance,
        creating it if it doesn't exist yet.

        Args:
        ----
            model: LLM model to use for code generation
            executor: Executor to use for code generation (alternative to model)
            config: Configuration for the tool factory

        Returns:
        -------
            ToolFactory: The singleton instance

        """
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(model=model, executor=executor, config=config)
            return cls._instance

    def __init__(
        self,
        model: Optional["LLM"] = None,
        executor=None,
        config: Optional[ToolFactoryConfig] = None,
    ) -> None:
        """
        Initialize the tool factory.

        Args:
        ----
            model: LLM model to use for code generation
            executor: Executor to use for code generation (alternative to model)
            config: Configuration for the tool factory

        """
        # Handle model/executor parameter
        if executor is not None:
            # Extract model from executor if provided
            self.model = getattr(executor, "model", model)
        else:
            self.model = model

        if self.model is None:
            msg = "Either model or executor must be provided"
            raise ValueError(msg)

        self.config = config or ToolFactoryConfig()

        # Initialize template and tool registries
        self._templates: dict[str, ToolTemplate] = {}
        self._template_dirs: list[str] = []
        self._tools: dict[str, type[ToolPlugin]] = {}
        self._templates_loaded = False

        # Create the output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize components lazily
        self._validator = None
        self._code_signer = None
        self._signature_verifier = None
        self._sandbox = None

        # Flag to track initialization status
        self._components_initialized = False

        logger.info("ToolFactory initialized with lazy component loading")

    @property
    def validator(self) -> "ToolValidator":
        """Get the tool validator, initializing it if needed."""
        self._ensure_components_initialized()
        assert self._validator is not None, "Validator not initialized"
        return self._validator

    @property
    def code_signer(self) -> "CodeSigner":
        """Get the code signer, initializing it if needed."""
        self._ensure_components_initialized()
        assert self._code_signer is not None, "Code signer not initialized"
        return self._code_signer

    @property
    def signature_verifier(self) -> "SignatureVerifier":
        """Get the signature verifier, initializing it if needed."""
        self._ensure_components_initialized()
        assert self._signature_verifier is not None, "Signature verifier not initialized"
        return self._signature_verifier

    @property
    def sandbox(self) -> Optional["Sandbox"]:
        """Get the sandbox, initializing it if needed."""
        # Sandbox is initialized on-demand in _create_tool_class
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: Optional["Sandbox"]) -> None:
        """Set the sandbox."""
        self._sandbox = value

    @property
    def templates(self) -> dict[str, ToolTemplate]:
        """Get all registered templates, loading them if needed."""
        self._ensure_templates_loaded()
        return self._templates

    @property
    def tools(self) -> dict[str, type[ToolPlugin]]:
        """Get all registered tools."""
        return self._tools

    def add_template_directory(self, directory: str) -> None:
        """
        Add a directory to search for templates.

        Templates will be loaded from this directory when templates are accessed.

        Args:
        ----
            directory: Directory path containing template files

        """
        if directory not in self._template_dirs:
            self._template_dirs.append(directory)
            # Reset the loaded flag to force reloading templates
            self._templates_loaded = False

    def _ensure_templates_loaded(self) -> None:
        """
        Ensure that templates are loaded from template directories.

        This method lazily loads templates when they are first needed.
        """
        if self._templates_loaded:
            return

        with self._init_lock:
            if self._templates_loaded:
                return

            # Load templates from directories
            for directory in self._template_dirs:
                self._load_templates_from_directory(directory)

            self._templates_loaded = True
            logger.debug(
                f"Loaded {len(self._templates)} templates from {len(self._template_dirs)} directories"
            )

    def _load_templates_from_directory(self, directory: str) -> None:
        """
        Load templates from a directory.

        Args:
        ----
            directory: Directory path containing template files

        """
        if not os.path.isdir(directory):
            logger.warning(f"Template directory does not exist: {directory}")
            return

        try:
            # Look for JSON files in the directory
            for filename in os.listdir(directory):
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath) as f:
                        template_data = json.load(f)

                    # Create a template from the data
                    template = ToolTemplate(**template_data)

                    # Register the template
                    self._templates[template.id] = template
                    logger.debug(
                        f"Loaded template from {filepath}: {template.name} (ID: {template.id})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load template from {filepath}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load templates from directory {directory}: {e}")

    def _ensure_components_initialized(self) -> None:
        """
        Ensure that all components are initialized.

        This method lazily initializes components when they are first needed,
        avoiding circular dependencies during initialization.
        """
        if self._components_initialized:
            return

        with self._init_lock:
            if self._components_initialized:
                return

            # Import components on-demand
            try:
                # Import tool validator
                from saplings.tool_factory._internal.security.tool_validator import ToolValidator

                self._validator = ToolValidator(self.config)

                # Import code signing components
                from saplings.tool_factory._internal.security.code_signing import (
                    CodeSigner,
                    SignatureVerifier,
                )

                self._code_signer = CodeSigner(self.config)
                self._signature_verifier = SignatureVerifier(self.config)

                self._components_initialized = True
                logger.debug("ToolFactory components initialized on-demand")
            except ImportError as e:
                logger.error(f"Failed to initialize ToolFactory components: {e}")
                raise

    def register_template(self, template: ToolTemplate) -> None:
        """
        Register a tool template.

        Args:
        ----
            template: Template to register

        Raises:
        ------
            ValueError: If a template with the same ID already exists

        """
        if template.id in self.templates:
            msg = f"Template with ID '{template.id}' already exists"
            raise ValueError(msg)

        self.templates[template.id] = template
        logger.info(f"Registered template: {template.name} (ID: {template.id})")

    async def generate_tool_code(self, spec: ToolSpecification) -> str:
        """
        Generate code for a tool based on a specification.

        Args:
        ----
            spec: Tool specification

        Returns:
        -------
            str: Generated code

        Raises:
        ------
            ValueError: If the template doesn't exist or parameters are missing

        """
        # Verify that the template exists
        if spec.template_id not in self.templates:
            msg = f"Template with ID '{spec.template_id}' does not exist"
            raise ValueError(msg)

        template = self.templates[spec.template_id]

        # Verify that all required parameters are provided
        for param in template.required_parameters:
            if param not in spec.parameters:
                # Check if this parameter can be generated by the LLM
                if param == "code_body":
                    # Generate the code body using the LLM
                    spec.parameters["code_body"] = await self._generate_code_with_llm(spec)
                else:
                    msg = f"Required parameter '{param}' is missing"
                    raise ValueError(msg)

        # Generate the code by replacing placeholders in the template
        code = template.template_code
        for param, value in spec.parameters.items():
            placeholder = f"{{{{{param}}}}}"
            code = code.replace(placeholder, str(value))

        return code

    async def _generate_code_with_llm(self, spec: ToolSpecification) -> str:
        """
        Generate code body using the LLM.

        Args:
        ----
            spec: Tool specification

        Returns:
        -------
            str: Generated code body

        """
        # Create the prompt
        prompt = f"""
        You are a code generator for a tool called "{spec.name}".

        Description: {spec.description}

        Generate the code body (implementation) for this tool. Do not include the function signature or docstring.
        Only provide the actual implementation code that would go inside the function body.

        For example, if the tool adds two numbers, you would generate:
        return a + b

        Generate the code body now:
        """

        # Generate the code body
        if self.model is None:
            msg = "Model is not available for code generation"
            raise ValueError(msg)

        response = await self.model.generate(prompt=prompt.strip())

        if response.text is None:
            return ""

        return response.text.strip()

    async def create_tool(self, spec: ToolSpecification) -> type[ToolPlugin]:
        """
        Create a tool from a specification.

        Args:
        ----
            spec: Tool specification

        Returns:
        -------
            Type[ToolPlugin]: Tool class

        Raises:
        ------
            ValueError: If the tool code is invalid or fails security checks

        """
        # Generate the tool code
        code = await self.generate_tool_code(spec)

        # Validate the code
        is_valid, validation_error = self._validate_tool_code(code)
        if not is_valid:
            msg = f"Invalid tool code: {validation_error}"
            raise ValueError(msg)

        # Perform security checks
        is_secure, security_error = self._perform_security_checks(code)
        if not is_secure:
            msg = f"Security check failed: {security_error}"
            raise ValueError(msg)

        # Create the tool class
        tool_class = self._create_tool_class(spec, code)

        # Register the tool
        self.tools[spec.id] = tool_class

        # Save the tool to disk
        self._save_tool(spec, code)

        logger.info(f"Created tool: {spec.name} (ID: {spec.id})")

        return tool_class

    def _validate_tool_code(self, code: str) -> tuple[bool, str]:
        """
        Validate the generated code.

        Args:
        ----
            code: Generated code

        Returns:
        -------
            Tuple[bool, str]: (is_valid, error_message)

        """
        # Use the validator to check the code
        validation_result = self.validator.validate(code)

        # Log warnings
        for warning in validation_result.warnings:
            logger.warning(f"Tool code warning: {warning}")

        return validation_result.is_valid, validation_result.error_message or ""

    def _perform_security_checks(self, code: str) -> tuple[bool, str]:
        """
        Perform security checks on the generated code.

        Args:
        ----
            code: Generated code

        Returns:
        -------
            Tuple[bool, str]: (is_secure, error_message)

        """
        # The validator already performs security checks, but we'll add additional
        # checks specific to the tool factory here if needed

        # For now, we'll just use the validator's security checks
        validation_result = self.validator._check_security(code)

        return validation_result.is_valid, validation_result.error_message or ""

    def _create_tool_class(self, spec: ToolSpecification, code: str) -> type[ToolPlugin]:
        """
        Create a tool class from the specification and code.

        Args:
        ----
            spec: Tool specification
            code: Generated code

        Returns:
        -------
            Type[ToolPlugin]: Tool class

        """
        # Create a namespace for the tool
        namespace = {}

        # Execute the code in the namespace
        exec(code, namespace)

        # Find the function in the namespace
        function_name = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                function_name = name
                break

        if not function_name:
            msg = "No function found in the generated code"
            raise ValueError(msg)

        # Get the function
        function = namespace[function_name]

        # Sign the code if enabled
        signature_info = self.code_signer.sign(code)

        # Create a dictionary to store tool factory and signature verifier references
        tool_refs: Dict[str, Any] = {
            "tool_factory": self,
            "signature_verifier": self.signature_verifier,
        }

        # Create the tool class
        class DynamicTool(ToolPlugin):
            """Dynamically generated tool."""

            @property
            def id(self) -> str:
                """ID of the tool."""
                return spec.id

            @property
            def name(self) -> str:
                """Name of the tool."""
                return spec.name

            @property
            def description(self) -> str:
                """Description of the tool."""
                return spec.description

            @property
            def version(self) -> str:
                """Version of the tool."""
                return "1.0.0"

            @property
            def plugin_type(self) -> PluginType:
                """Type of the plugin."""
                return PluginType.TOOL

            _code = code
            _signature_info = signature_info
            _sandbox_type = self.config.sandbox_type

            # Store references in a class variable
            _refs = tool_refs

            def execute(self, *args, **kwargs):
                """Execute the tool."""
                # Get the signature verifier from refs
                signature_verifier = self._refs.get("signature_verifier")
                if signature_verifier is None:
                    msg = "Signature verifier not available"
                    raise ValueError(msg)

                # Verify the signature if enabled
                if not signature_verifier.verify(self._code, self._signature_info):
                    msg = "Code signature verification failed"
                    raise ValueError(msg)

                # If sandboxing is enabled, execute in sandbox
                if self._sandbox_type != SandboxType.NONE:
                    # Get the tool factory from refs
                    tool_factory = self._refs.get("tool_factory")
                    if tool_factory is None:
                        msg = "Tool factory not available"
                        raise ValueError(msg)

                    # Lazy initialize the sandbox
                    if tool_factory._sandbox is None:
                        # Import sandbox module on-demand
                        from saplings.tool_factory._internal.sandbox.sandbox import get_sandbox

                        tool_factory._sandbox = get_sandbox(tool_factory.config)

                    # Execute in sandbox
                    return tool_factory.sandbox.execute(
                        code=self._code,
                        function_name=function_name,
                        args=list(args),  # Convert tuple to list
                        kwargs=kwargs,
                    )

                # Otherwise, execute directly
                return function(*args, **kwargs)

        # Set the class name
        DynamicTool.__name__ = f"{spec.id.capitalize()}Tool"

        # Register the tool as a plugin
        register_plugin(DynamicTool)

        return DynamicTool

    def _save_tool(self, spec: ToolSpecification, code: str) -> None:
        """
        Save the tool to disk.

        Args:
        ----
            spec: Tool specification
            code: Generated code

        """
        # Create the file path
        file_path = os.path.join(self.config.output_dir, f"{spec.id}.py")

        # Sign the code if enabled
        signature_info = self.code_signer.sign(code)

        # Convert signature info to JSON
        signature_json = json.dumps(signature_info, indent=2)

        # Add a header to the code
        header = f"""
# Generated tool: {spec.name}
# Description: {spec.description}
# Generated by: ToolFactory
# Signature: {signature_json}
"""

        # Save the code to disk
        with open(file_path, "w") as f:
            f.write(header + code)

        logger.info(f"Saved tool to: {file_path}")

        # Save signature info separately if advanced signing is enabled
        if signature_info.get("signature_type") == "advanced":
            sig_path = os.path.join(self.config.output_dir, f"{spec.id}.sig")
            with open(sig_path, "w") as f:
                f.write(signature_json)
            logger.info(f"Saved signature to: {sig_path}")

    def get_tool(self, tool_id: str) -> type[ToolPlugin]:
        """
        Get a tool by ID.

        Args:
        ----
            tool_id: ID of the tool

        Returns:
        -------
            Type[ToolPlugin]: Tool class

        Raises:
        ------
            ValueError: If the tool doesn't exist

        """
        if tool_id not in self.tools:
            msg = f"Tool with ID '{tool_id}' does not exist"
            raise ValueError(msg)

        return self.tools[tool_id]

    def list_tools(self):
        """
        List all registered tools.

        Returns
        -------
            Dict[str, Type[ToolPlugin]]: Dictionary of tool ID to tool class

        """
        return self.tools.copy()

    def list_templates(self):
        """
        List all registered templates.

        Returns
        -------
            Dict[str, ToolTemplate]: Dictionary of template ID to template

        """
        return self.templates.copy()

    def cleanup(self) -> None:
        """
        Clean up resources used by the tool factory.

        This method should be called when the tool factory is no longer needed
        to release any resources it's using.
        """
        # Clean up the sandbox if it exists
        if self.sandbox is not None:
            self.sandbox.cleanup()
            self.sandbox = None

        logger.info("Cleaned up tool factory resources")
