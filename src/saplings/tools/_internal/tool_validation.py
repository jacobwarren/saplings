from __future__ import annotations

"""
Tool validation utilities for Saplings.

This module provides functions to validate tool attributes and parameters.
"""

import ast
import builtins
import inspect

# Set of built-in names in Python
_BUILTIN_NAMES = set(vars(builtins))

# Base built-in modules that are safe to import
BASE_BUILTIN_MODULES = {
    "math",
    "random",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "uuid",
    "copy",
    "string",
    "time",
    "calendar",
    "fractions",
    "decimal",
    "bisect",
    "heapq",
    "array",
    "enum",
    "dataclasses",
}


class MethodChecker(ast.NodeVisitor):
    """
    AST visitor to check for code issues in tool methods.

    This checker validates:
    - Only defined names are used
    - Imports are from authorized packages
    - Methods are self-contained
    """

    def __init__(self, class_attributes: set[str], check_imports: bool = True) -> None:
        self.undefined_names = set()
        self.imports = {}
        self.from_imports = {}
        self.assigned_names = set()
        self.arg_names = set()
        self.class_attributes = class_attributes
        self.errors = []
        self.check_imports = check_imports
        self.typing_names = {"Any", "Optional", "List", "Dict", "Set", "Tuple", "Union", "Callable"}

    def visit_arguments(self, node):
        """Collect function arguments."""
        self.arg_names = {arg.arg for arg in node.args}
        # Check if the node has kwarg attribute (keyword arguments like **kwargs)
        if hasattr(node, "kwarg") and node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        # Check if the node has vararg attribute (variable arguments like *args)
        if hasattr(node, "vararg") and node.vararg:
            self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node):
        """Track import statements."""
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

            # Check if import is authorized
            if self.check_imports and name.name not in BASE_BUILTIN_MODULES:
                self.errors.append(f"Unauthorized import: {name.name}")

    def visit_ImportFrom(self, node):
        """Track import from statements."""
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

            # Check if import is authorized
            if self.check_imports and module not in BASE_BUILTIN_MODULES:
                self.errors.append(f"Unauthorized import from: {module}")

    def visit_Assign(self, node):
        """Track variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.visit(node.value)

    def visit_With(self, node):
        """Track aliases in 'with' statements (the 'y' in 'with X as y')."""
        for item in node.items:
            if item.optional_vars:  # This is the 'y' in 'with X as y'
                if isinstance(item.optional_vars, ast.Name):
                    self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Track exception aliases (the 'e' in 'except Exception as e')."""
        if node.name:  # This is the 'e' in 'except Exception as e'
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Track annotated assignments."""
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        """Track loop variables."""
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        """Helper method to handle generators in all types of comprehensions."""
        for generator in generators:
            if isinstance(generator.target, ast.Name):
                self.assigned_names.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node):
        """Track variables in list comprehensions."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """Track variables in dictionary comprehensions."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """Track variables in set comprehensions."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Skip self attributes but visit other attributes."""
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def visit_Name(self, node):
        """Check for undefined names."""
        if isinstance(node.ctx, ast.Load) and not (
            node.id in _BUILTIN_NAMES
            or node.id in BASE_BUILTIN_MODULES
            or node.id in self.arg_names
            or node.id == "self"
            or node.id in self.class_attributes
            or node.id in self.imports
            or node.id in self.from_imports
            or node.id in self.assigned_names
            or node.id in self.typing_names
        ):
            self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        """Check for undefined function calls."""
        if isinstance(node.func, ast.Name) and not (
            node.func.id in _BUILTIN_NAMES
            or node.func.id in BASE_BUILTIN_MODULES
            or node.func.id in self.arg_names
            or node.func.id == "self"
            or node.func.id in self.class_attributes
            or node.func.id in self.imports
            or node.func.id in self.from_imports
            or node.func.id in self.assigned_names
        ):
            self.errors.append(f"Function '{node.func.id}' is undefined.")
        self.generic_visit(node)


class ClassLevelChecker(ast.NodeVisitor):
    """
    AST visitor to check class-level attributes and initialization.

    This checker validates:
    - Class attributes are properly defined
    - __init__ parameters have default values
    - Complex attributes are defined in __init__, not as class attributes
    """

    def __init__(self) -> None:
        self.imported_names = set()
        self.complex_attributes = set()
        self.class_attributes = set()
        self.non_defaults = set()
        self.non_literal_defaults = set()
        self.in_method = False
        self.invalid_attributes = []

    def visit_FunctionDef(self, node):
        """Check function definitions, with special handling for __init__."""
        if node.name == "__init__":
            self._check_init_function_parameters(node)

        # Track whether we're in a method or at class level
        old_context = self.in_method
        self.in_method = True
        self.generic_visit(node)
        self.in_method = old_context

    def visit_Assign(self, node):
        """Check class attribute assignments."""
        if self.in_method:
            return

        # Track class attributes
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.class_attributes.add(target.id)

        # Check if the assignment is more complex than simple literals
        is_complex = False
        for subnode in ast.walk(node.value):
            if isinstance(subnode, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Compare)):
                is_complex = True
                break
            if not isinstance(subnode, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Name)):
                is_complex = True
                break

        if is_complex:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.complex_attributes.add(target.id)

        # Check specific class attributes
        if len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
            attr_name = node.targets[0].id

            # Check 'name' attribute
            if attr_name == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(
                        "Class attribute 'name' must be a string constant"
                    )
                elif hasattr(node.value, "value") and not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{type(node.value.value).__name__}'"
                    )
                elif hasattr(node.value, "value") and not node.value.value.isidentifier():
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier, found '{node.value.value}'"
                    )

    def _check_init_function_parameters(self, node):
        """Check that __init__ parameters have default values."""
        # Get positional args and defaults
        if hasattr(node, "args") and hasattr(node.args, "args"):
            args = node.args.args
            defaults = []
            if hasattr(node.args, "defaults"):
                defaults = node.args.defaults

            # Check if each parameter has a default value
            if len(args) > 1:  # Skip 'self'
                for i, arg in enumerate(args[1:], 1):  # Start at 1 to skip 'self'
                    # Calculate the default index
                    default_index = i - (len(args) - len(defaults))

                    # Threshold for default index
                    DEFAULT_INDEX_THRESHOLD = 0

                    if default_index < DEFAULT_INDEX_THRESHOLD:
                        # This parameter doesn't have a default
                        self.non_defaults.add(arg.arg)
                    elif default_index >= 0:
                        # Check if default is a literal
                        default = defaults[default_index]
                        if not isinstance(default, (ast.Constant, ast.Dict, ast.List, ast.Set)):
                            self.non_literal_defaults.add(arg.arg)


def is_valid_identifier(name: str) -> bool:
    """
    Check if a string is a valid Python identifier.

    Args:
    ----
        name: The string to check

    Returns:
    -------
        True if the string is a valid Python identifier, False otherwise

    """
    if not name:
        return False

    # Check if the name is a valid Python identifier
    if not name.isidentifier():
        return False

    # Check if the name is a Python keyword
    import keyword

    return not keyword.iskeyword(name)


def validate_tool_attributes(tool_class: type, check_imports: bool = True) -> None:
    """
    Validate that a tool class has all required attributes and follows best practices.

    This function performs comprehensive validation of a tool class, including:
    - Required attributes (name, description, inputs, output_type)
    - Class attribute definitions
    - Method implementations
    - Import usage
    - Parameter validation

    Args:
    ----
        tool_class: The tool class to validate
        check_imports: Whether to check for unauthorized imports

    Raises:
    ------
        ValueError: If the tool class has validation errors

    """
    required_attributes = {
        "name": str,
        "description": str,
        "parameters": dict,
        "output_type": str,
    }

    errors = []

    # Check required attributes
    for attr_name, attr_type in required_attributes.items():
        if not hasattr(tool_class, attr_name):
            errors.append(f"Missing required attribute: {attr_name}")
            continue

        attr_value = getattr(tool_class, attr_name)
        if not isinstance(attr_value, attr_type):
            errors.append(
                f"Attribute {attr_name} should be of type {attr_type.__name__}, "
                f"got {type(attr_value).__name__} instead"
            )

    # Check name is valid
    if hasattr(tool_class, "name"):
        name = tool_class.name
        if not name or not isinstance(name, str) or not is_valid_identifier(name):
            errors.append(f"Invalid tool name: {name}. Must be a valid Python identifier.")

    # Check parameters format
    if hasattr(tool_class, "parameters") and isinstance(tool_class.parameters, dict):
        parameters = tool_class.parameters
        for param_name, param_spec in parameters.items():
            if not isinstance(param_spec, dict):
                errors.append(f"Parameter specification for {param_name} should be a dictionary")
                continue

            if "type" not in param_spec:
                errors.append(f"Parameter {param_name} is missing required 'type' field")

            if "description" not in param_spec:
                errors.append(f"Parameter {param_name} is missing required 'description' field")

            # Check parameter type is valid
            if "type" in param_spec:
                param_type = param_spec["type"]
                valid_types = [
                    "string",
                    "boolean",
                    "integer",
                    "number",
                    "image",
                    "audio",
                    "array",
                    "object",
                    "any",
                ]
                if param_type not in valid_types:
                    errors.append(
                        f"Invalid parameter type for {param_name}: {param_type}. Must be one of {valid_types}"
                    )

    # Check output_type is valid
    if hasattr(tool_class, "output_type"):
        output_type = tool_class.output_type
        valid_types = [
            "string",
            "boolean",
            "integer",
            "number",
            "image",
            "audio",
            "array",
            "object",
            "any",
        ]
        if output_type not in valid_types:
            errors.append(f"Invalid output_type: {output_type}. Must be one of {valid_types}")

    # Check forward method exists and has correct signature
    if not hasattr(tool_class, "forward"):
        errors.append("Missing required method: forward")
    else:
        forward = tool_class.forward
        if not callable(forward):
            errors.append("'forward' attribute must be a callable method")
        # Check forward method signature matches parameters
        elif hasattr(tool_class, "parameters") and not getattr(
            tool_class, "skip_forward_signature_validation", False
        ):
            try:
                signature = inspect.signature(forward)
                actual_params = {p for p in signature.parameters if p != "self"}
                expected_params = set(getattr(tool_class, "parameters", {}).keys())

                if actual_params != expected_params:
                    errors.append(
                        f"Forward method parameters {actual_params} don't match parameters {expected_params}"
                    )
            except (ValueError, TypeError):
                # Can't get signature, skip this check
                pass

    # Perform AST-based validation if source code is available
    try:
        source = inspect.getsource(tool_class)
        tree = ast.parse(source)

        # Find the class definition
        class_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == tool_class.__name__:
                class_node = node
                break

        if class_node:
            # Check class-level attributes
            class_checker = ClassLevelChecker()
            class_checker.visit(class_node)

            # Add errors from class checker
            if class_checker.invalid_attributes:
                errors.extend(class_checker.invalid_attributes)

            if class_checker.complex_attributes:
                errors.append(
                    f"Complex attributes should be defined in __init__, not as class attributes: "
                    f"{', '.join(class_checker.complex_attributes)}"
                )

            if class_checker.non_defaults:
                errors.append(
                    f"Parameters in __init__ must have default values, found required parameters: "
                    f"{', '.join(class_checker.non_defaults)}"
                )

            if class_checker.non_literal_defaults:
                errors.append(
                    f"Parameters in __init__ must have literal default values, found non-literal defaults: "
                    f"{', '.join(class_checker.non_literal_defaults)}"
                )

            # Check methods
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    method_checker = MethodChecker(
                        class_checker.class_attributes, check_imports=check_imports
                    )
                    method_checker.visit(node)
                    errors.extend(
                        [f"In method '{node.name}': {error}" for error in method_checker.errors]
                    )

    except (OSError, TypeError):
        # Can't get source code, skip AST-based checks
        if check_imports and hasattr(tool_class, "forward"):
            try:
                # Still try to check the forward method
                source = inspect.getsource(tool_class.forward)
                tree = ast.parse(source)
                checker = MethodChecker(set(), check_imports=check_imports)
                checker.visit(tree)
                errors.extend([f"In method 'forward': {error}" for error in checker.errors])
            except (OSError, TypeError):
                # Can't get source code for forward method either, skip this check
                pass

    if errors:
        raise ValueError(f"Tool validation failed for {tool_class.__name__}:\n" + "\n".join(errors))


def validate_tool_parameters(tool, *args, **kwargs) -> None:
    """
    Validate that the parameters passed to a tool match its expected inputs.

    Args:
    ----
        tool: The tool instance
        *args: Positional arguments
        **kwargs: Keyword arguments

    Raises:
    ------
        ValueError: If the parameters don't match the tool's expected inputs

    """
    if not hasattr(tool, "inputs"):
        return

    inputs = tool.inputs
    errors = []

    # Check if args can be mapped to inputs
    if args:
        if len(args) > len(inputs):
            errors.append(f"Too many positional arguments: expected {len(inputs)}, got {len(args)}")

        # Map positional args to input names
        input_names = list(inputs.keys())
        for i, arg in enumerate(args):
            if i < len(input_names):
                input_name = input_names[i]
                input_spec = inputs[input_name]

                # Check type if specified
                if "type" in input_spec:
                    expected_type = input_spec["type"]
                    # Perform basic type checking
                    if expected_type == "string" and not isinstance(arg, str):
                        errors.append(f"Argument {i} ({input_name}) should be a string")
                    elif expected_type == "integer" and not isinstance(arg, int):
                        errors.append(f"Argument {i} ({input_name}) should be an integer")
                    elif expected_type == "number" and not isinstance(arg, (int, float)):
                        errors.append(f"Argument {i} ({input_name}) should be a number")
                    elif expected_type == "boolean" and not isinstance(arg, bool):
                        errors.append(f"Argument {i} ({input_name}) should be a boolean")

    # Check kwargs
    for kwarg_name, kwarg_value in kwargs.items():
        if kwarg_name not in inputs:
            errors.append(f"Unexpected keyword argument: {kwarg_name}")
            continue

        input_spec = inputs[kwarg_name]

        # Check type if specified
        if "type" in input_spec:
            expected_type = input_spec["type"]
            # Perform basic type checking
            if expected_type == "string" and not isinstance(kwarg_value, str):
                errors.append(f"Argument {kwarg_name} should be a string")
            elif expected_type == "integer" and not isinstance(kwarg_value, int):
                errors.append(f"Argument {kwarg_name} should be an integer")
            elif expected_type == "number" and not isinstance(kwarg_value, (int, float)):
                errors.append(f"Argument {kwarg_name} should be a number")
            elif expected_type == "boolean" and not isinstance(kwarg_value, bool):
                errors.append(f"Argument {kwarg_name} should be a boolean")

    # Check for missing required arguments
    provided_args = set(kwargs.keys()) | set(list(inputs.keys())[: len(args)])
    for input_name, input_spec in inputs.items():
        if input_name not in provided_args and input_spec.get("required", True):
            errors.append(f"Missing required argument: {input_name}")

    if errors:
        raise ValueError("Tool parameter validation failed:\n" + "\n".join(errors))


def validate_tool(tool):
    """
    Validate a tool instance.

    Args:
    ----
        tool: The tool instance to validate

    Raises:
    ------
        ValueError: If the tool is invalid

    """
    # Check required attributes
    required_attrs = ["name", "description", "parameters"]
    for attr in required_attrs:
        if not hasattr(tool, attr):
            msg = f"Tool is missing required attribute: {attr}"
            raise ValueError(msg)

    # Check that the tool is callable
    if not callable(tool):
        msg = "Tool must be callable"
        raise ValueError(msg)

    # Check parameters
    if not isinstance(tool.parameters, dict):
        msg = "Tool parameters must be a dictionary"
        raise ValueError(msg)

    for param_name, param_info in tool.parameters.items():
        if not isinstance(param_info, dict):
            msg = f"Parameter info for {param_name} must be a dictionary"
            raise ValueError(msg)

        # Check required parameter fields
        required_fields = ["type", "description"]
        for field in required_fields:
            if field not in param_info:
                msg = f"Parameter {param_name} is missing required field: {field}"
                raise ValueError(msg)


__all__ = [
    "validate_tool",
    "validate_tool_attributes",
    "validate_tool_parameters",
]
