"""
Patch generator module for Saplings.

This module provides the PatchGenerator class for auto-fixing errors in code.
"""

import ast
import logging
import os
import re
import sys
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check if pylint is available
try:
    import pylint
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False
    logger.info("Pylint not available. Static analysis will be limited.")

# Check if pyflakes is available
try:
    import pyflakes
    PYFLAKES_AVAILABLE = True
except ImportError:
    PYFLAKES_AVAILABLE = False
    logger.info("Pyflakes not available. Static analysis will be limited.")


class PatchStatus(str, Enum):
    """Status of a patch."""

    GENERATED = "generated"  # Patch has been generated
    APPLIED = "applied"      # Patch has been applied
    VALIDATED = "validated"  # Patch has been validated
    FAILED = "failed"        # Patch generation or application failed


class PatchResult:
    """Result of applying a patch."""

    def __init__(
        self,
        success: bool,
        patched_code: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        Initialize the patch result.

        Args:
            success: Whether the patch was successfully applied
            patched_code: The patched code (if successful)
            error: Error message (if unsuccessful)
        """
        self.success = success
        self.patched_code = patched_code
        self.error = error


class Patch:
    """A code patch."""

    def __init__(
        self,
        original_code: str,
        patched_code: str,
        error: str,
        error_info: Dict[str, Any],
        status: PatchStatus = PatchStatus.GENERATED,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the patch.

        Args:
            original_code: Original code with error
            patched_code: Patched code
            error: Error message
            error_info: Information about the error
            status: Status of the patch
            metadata: Additional metadata
        """
        self.original_code = original_code
        self.patched_code = patched_code
        self.error = error
        self.error_info = error_info
        self.status = status
        self.metadata = metadata or {}
        self.timestamp = self.metadata.get("timestamp", None)


class PatchGenerator:
    """
    Generator for code patches.

    This class analyzes errors in code and generates patches to fix them.
    """

    def __init__(
        self,
        max_retries: int = 3,
        success_pair_collector: Optional[Any] = None,
    ):
        """
        Initialize the patch generator.

        Args:
            max_retries: Maximum number of retry attempts
            success_pair_collector: Collector for successful error-fix pairs
        """
        self.max_retries = max_retries
        self.retry_count = 0
        self.patches: List[Patch] = []
        self.success_pair_collector = success_pair_collector

    def analyze_code_with_static_tools(self, code: str) -> Dict[str, Any]:
        """
        Analyze code using static analysis tools to identify issues.

        Args:
            code: Code to analyze

        Returns:
            Dict with analysis results
        """
        analysis_results = {
            "pylint": [],
            "pyflakes": [],
            "ast": {"valid": False, "errors": []}
        }

        # Try to parse with AST first
        try:
            ast.parse(code)
            analysis_results["ast"]["valid"] = True
        except SyntaxError as e:
            analysis_results["ast"]["valid"] = False
            analysis_results["ast"]["errors"].append({
                "line": e.lineno,
                "column": e.offset,
                "message": str(e)
            })

        # Use pylint for static analysis if available
        if PYLINT_AVAILABLE:
            try:
                from pylint.lint import Run
                from pylint.reporters.text import TextReporter

                class CustomReporter(TextReporter):
                    def __init__(self):
                        super().__init__()
                        self.messages = []

                    def handle_message(self, msg):
                        self.messages.append(msg)

                reporter = CustomReporter()
                # Write code to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                    temp_file.write(code.encode('utf-8'))
                    temp_path = temp_file.name

                # Run pylint with custom reporter
                Run([temp_path], reporter=reporter, exit=False)
                os.unlink(temp_path)

                # Process messages
                for msg in reporter.messages:
                    analysis_results["pylint"].append({
                        'line': msg.line,
                        'column': msg.column,
                        'message': msg.msg,
                        'symbol': msg.symbol,
                        'msg_id': msg.msg_id
                    })
            except Exception as e:
                logger.warning(f"Error running pylint: {e}")

        # Use pyflakes for additional error detection if available
        if PYFLAKES_AVAILABLE:
            try:
                from pyflakes.api import check
                from pyflakes.reporter import Reporter

                class CustomFlakesReporter(Reporter):
                    def __init__(self):
                        self.messages = []

                    def unexpectedError(self, filename, msg):
                        self.messages.append({"type": "unexpected", "message": str(msg)})

                    def syntaxError(self, filename, msg, lineno, offset, text):
                        self.messages.append({
                            "type": "syntax",
                            "message": str(msg),
                            "line": lineno,
                            "offset": offset,
                            "text": text
                        })

                    def flake(self, message):
                        self.messages.append({
                            "type": "flake",
                            "message": str(message),
                            "line": message.lineno,
                            "col": message.col
                        })

                reporter = CustomFlakesReporter()
                # Write code to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                    temp_file.write(code.encode('utf-8'))
                    temp_path = temp_file.name

                # Run pyflakes
                check(temp_path, reporter)
                os.unlink(temp_path)

                analysis_results['pyflakes'] = reporter.messages
            except Exception as e:
                logger.warning(f"Error running pyflakes: {e}")

        return analysis_results

    def analyze_error(self, code: str, error: str) -> Dict[str, Any]:
        """
        Analyze an error message to identify the type and patterns.

        Args:
            code: Code with error
            error: Error message

        Returns:
            Dict[str, Any]: Information about the error
        """
        error_info = {
            "type": "Unknown",
            "message": error,
            "patterns": [],
            "line_number": None,
        }

        # Extract error type
        error_type_match = re.match(r"([A-Za-z]+Error):", error)
        if error_type_match:
            error_info["type"] = error_type_match.group(1)

        # Extract error message
        error_message_match = re.match(r"[A-Za-z]+Error: (.*)", error)
        if error_message_match:
            error_info["message"] = error_message_match.group(1)

        # Extract line number if present
        line_number_match = re.search(r"line (\d+)", error)
        if line_number_match:
            error_info["line_number"] = int(line_number_match.group(1))

        # Identify common error patterns
        if error_info["type"] == "SyntaxError":
            if "unexpected EOF" in error or "unexpected end of file" in error:
                error_info["patterns"].append("missing_parenthesis")
            elif "invalid syntax" in error:
                error_info["patterns"].append("invalid_syntax")
            elif "expected an indented block" in error:
                error_info["patterns"].append("indentation_error")

        elif error_info["type"] == "NameError":
            if "name" in error and "is not defined" in error:
                error_info["patterns"].append("undefined_variable")
                # Extract the variable name
                var_match = re.search(r"name '([^']+)' is not defined", error)
                if var_match:
                    error_info["variable"] = var_match.group(1)

        elif error_info["type"] == "TypeError":
            if "takes" in error and "arguments" in error:
                error_info["patterns"].append("argument_error")
            elif "object is not" in error:
                error_info["patterns"].append("type_mismatch")

        elif error_info["type"] == "ImportError" or error_info["type"] == "ModuleNotFoundError":
            if "No module named" in error:
                error_info["patterns"].append("missing_module")
                # Extract the module name
                module_match = re.search(r"No module named '([^']+)'", error)
                if module_match:
                    error_info["module"] = module_match.group(1)

        # Try to analyze the code to get more context
        try:
            ast.parse(code)
            error_info["ast_available"] = True
        except SyntaxError:
            error_info["ast_available"] = False

        # Run static analysis tools if available
        error_info["static_analysis"] = self.analyze_code_with_static_tools(code)

        return error_info

    def generate_patch(self, code: str, error: str) -> Patch:
        """
        Generate a patch for an error.

        Args:
            code: Code with error
            error: Error message

        Returns:
            Patch: Generated patch
        """
        # Analyze the error
        error_info = self.analyze_error(code, error)

        # Initialize patched code with original code
        patched_code = code
        used_static_analysis = False

        # First, try to use static analysis results to guide the fix
        static_analysis = error_info.get("static_analysis", {})

        # Check if we have useful static analysis results
        if self._can_fix_with_static_analysis(static_analysis, error_info):
            static_patched_code = self._fix_with_static_analysis(code, static_analysis, error_info)
            # Only use the static analysis result if it actually changed the code
            if static_patched_code != code:
                patched_code = static_patched_code
                used_static_analysis = True

        # If static analysis didn't produce a fix or didn't change the code, fall back to pattern-based fixes
        if patched_code == code:
            if "missing_parenthesis" in error_info["patterns"]:
                patched_code = self._fix_missing_parenthesis(code)

            elif "undefined_variable" in error_info["patterns"] and "variable" in error_info:
                patched_code = self._fix_undefined_variable(code, error_info["variable"])

            elif "indentation_error" in error_info["patterns"] or "expected an indented block" in error:
                patched_code = self._fix_indentation(code)

            elif "invalid_syntax" in error_info["patterns"]:
                patched_code = self._fix_invalid_syntax(code)

            elif "argument_error" in error_info["patterns"]:
                patched_code = self._fix_argument_error(code, error)

            elif "type_mismatch" in error_info["patterns"]:
                patched_code = self._fix_type_mismatch(code, error)

            elif "missing_module" in error_info["patterns"] and "module" in error_info:
                patched_code = self._fix_missing_module(code, error_info["module"])

        # Create and return the patch
        patch = Patch(
            original_code=code,
            patched_code=patched_code,
            error=error,
            error_info=error_info,
            status=PatchStatus.GENERATED,
            metadata={
                "retry_count": self.retry_count,
                "used_static_analysis": used_static_analysis
            },
        )

        return patch

    def _can_fix_with_static_analysis(self, static_analysis: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """
        Determine if we can fix the error using static analysis results.

        Args:
            static_analysis: Static analysis results
            error_info: Error information (not used in current implementation but kept for future use)

        Returns:
            bool: True if we can fix with static analysis, False otherwise
        """
        # Check if AST parsing failed (syntax error)
        if not static_analysis.get("ast", {}).get("valid", True):
            return True

        # Check if we have pylint results
        if static_analysis.get("pylint") and len(static_analysis["pylint"]) > 0:
            return True

        # Check if we have pyflakes results
        if static_analysis.get("pyflakes") and len(static_analysis["pyflakes"]) > 0:
            return True

        return False

    def _fix_with_static_analysis(self, code: str, static_analysis: Dict[str, Any], error_info: Dict[str, Any]) -> str:
        """
        Fix code using static analysis results.

        Args:
            code: Code to fix
            static_analysis: Static analysis results
            error_info: Error information (not used in current implementation but kept for future use)

        Returns:
            str: Fixed code
        """
        # Start with the original code
        fixed_code = code

        # First, check for syntax errors from AST
        if not static_analysis.get("ast", {}).get("valid", True):
            ast_errors = static_analysis.get("ast", {}).get("errors", [])
            if ast_errors:
                # Use the first error to guide the fix
                error = ast_errors[0]
                line_num = error.get("line")

                if line_num is not None:
                    # Try to fix the syntax error
                    if "unexpected EOF" in error.get("message", "") or "unexpected end of file" in error.get("message", ""):
                        fixed_code = self._fix_missing_parenthesis(code)
                    elif "invalid syntax" in error.get("message", ""):
                        fixed_code = self._fix_invalid_syntax(code)
                    elif "expected an indented block" in error.get("message", ""):
                        fixed_code = self._fix_indentation(code)

        # If we couldn't fix with AST errors, try pylint
        if fixed_code == code and static_analysis.get("pylint"):
            pylint_msgs = static_analysis["pylint"]

            # Group messages by line number
            line_to_msgs = {}
            for msg in pylint_msgs:
                line = msg.get("line")
                if line not in line_to_msgs:
                    line_to_msgs[line] = []
                line_to_msgs[line].append(msg)

            # Process messages by line
            for line, msgs in sorted(line_to_msgs.items()):
                for msg in msgs:
                    symbol = msg.get("symbol")
                    message = msg.get("message", "")

                    # Handle different pylint error types
                    if symbol == "undefined-variable":
                        # Extract variable name from message
                        var_match = re.search(r"Undefined variable '([^']+)'", message)
                        if var_match:
                            variable = var_match.group(1)
                            fixed_code = self._fix_undefined_variable(fixed_code, variable)

                    elif symbol == "missing-module-docstring":
                        # Add a module docstring
                        if not re.search(r'^""".*"""', fixed_code, re.DOTALL):
                            fixed_code = f'"""\nModule docstring.\n"""\n\n{fixed_code}'

                    elif symbol == "invalid-name":
                        # We don't fix naming conventions automatically
                        pass

                    elif symbol == "unused-import":
                        # Extract the import name
                        import_match = re.search(r"Unused import ([^']+)", message)
                        if import_match:
                            import_name = import_match.group(1)
                            # Remove the import
                            fixed_code = self._remove_import(fixed_code, import_name)

                    elif symbol == "too-many-arguments":
                        # We don't fix this automatically
                        pass

                    elif symbol in ["missing-function-docstring", "missing-class-docstring"]:
                        # Add a docstring to the function or class
                        fixed_code = self._add_docstring(fixed_code, line)

        # If we couldn't fix with pylint, try pyflakes
        if fixed_code == code and static_analysis.get("pyflakes"):
            pyflakes_msgs = static_analysis["pyflakes"]

            for msg in pyflakes_msgs:
                msg_type = msg.get("type")
                message = msg.get("message", "")

                if msg_type == "syntax":
                    # Already handled by AST
                    pass

                elif "undefined name" in message:
                    # Extract variable name
                    var_match = re.search(r"undefined name '([^']+)'", message)
                    if var_match:
                        variable = var_match.group(1)
                        fixed_code = self._fix_undefined_variable(fixed_code, variable)

                elif "imported but unused" in message:
                    # Extract import name
                    import_match = re.search(r"'([^']+)' imported but unused", message)
                    if import_match:
                        import_name = import_match.group(1)
                        fixed_code = self._remove_import(fixed_code, import_name)

        return fixed_code

    def _remove_import(self, code: str, import_name: str) -> str:
        """
        Remove an unused import from the code.

        Args:
            code: Code to modify
            import_name: Name of the import to remove

        Returns:
            str: Modified code
        """
        lines = code.split("\n")
        modified_lines = []

        for line in lines:
            # Skip lines that import only this module
            if re.match(rf"^\s*import\s+{import_name}\s*$", line):
                continue

            # Handle "from x import y" style imports
            if "from" in line and "import" in line:
                # Check if this is the only import
                if re.match(rf"^\s*from\s+.*\s+import\s+{import_name}\s*$", line):
                    continue

                # Check if it's part of multiple imports
                if re.match(rf"^\s*from\s+.*\s+import\s+.*{import_name}.*", line):
                    # Remove just this import
                    parts = line.split("import")
                    imports = parts[1].split(",")
                    filtered_imports = [imp for imp in imports if import_name not in imp.strip()]

                    if filtered_imports:
                        line = parts[0] + "import " + ", ".join(filtered_imports)
                    else:
                        # Skip the line if no imports remain
                        continue

            modified_lines.append(line)

        return "\n".join(modified_lines)

    def _add_docstring(self, code: str, line_num: int) -> str:
        """
        Add a docstring to a function or class.

        Args:
            code: Code to modify
            line_num: Line number of the function or class definition

        Returns:
            str: Modified code
        """
        lines = code.split("\n")

        # Make sure line_num is valid
        if line_num < 1 or line_num > len(lines):
            return code

        # Find the indentation level
        match = re.match(r"^(\s*)(def|class)\s+([a-zA-Z0-9_]+)", lines[line_num - 1])
        if not match:
            return code

        indent = match.group(1)
        entity_type = match.group(2)
        # entity_name is captured but not used in current implementation
        # may be useful for customizing docstrings in the future

        # Find where to insert the docstring
        for i in range(line_num, len(lines)):
            if ":" in lines[i]:
                # Insert after the colon
                docstring_indent = indent + "    "
                if entity_type == "def":
                    docstring = f'{docstring_indent}"""\n{docstring_indent}Function docstring.\n\n{docstring_indent}Returns:\n{docstring_indent}    None\n{docstring_indent}"""'
                else:  # class
                    docstring = f'{docstring_indent}"""\n{docstring_indent}Class docstring.\n{docstring_indent}"""'

                lines.insert(i + 1, docstring)
                break

        return "\n".join(lines)

    def apply_patch(self, patch: Patch) -> PatchResult:
        """
        Apply a patch and update the retry count.

        Args:
            patch: Patch to apply

        Returns:
            PatchResult: Result of applying the patch
        """
        # Check if we've reached the retry limit
        if self.retry_count >= self.max_retries:
            return PatchResult(
                success=False,
                error="Maximum retry limit reached",
            )

        # Increment retry count
        self.retry_count += 1

        # Add the patch to the list of patches
        self.patches.append(patch)

        # Update the patch status
        patch.status = PatchStatus.APPLIED

        # Return success
        return PatchResult(
            success=True,
            patched_code=patch.patched_code,
        )

    def validate_patch(self, patched_code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a patched code by executing it.

        Args:
            patched_code: Patched code to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Execute the patched code
        success, error = self._execute_code(patched_code)

        return success, error

    def after_success(self, patch: Patch) -> None:
        """
        Process a successful patch for collection and learning.

        Args:
            patch: Successful patch
        """
        # Update the patch status
        patch.status = PatchStatus.VALIDATED

        # If we have a success pair collector, collect the pair
        if self.success_pair_collector is not None:
            self.success_pair_collector.collect(patch)

        logger.info(f"Successfully applied patch after {self.retry_count} retries")

    def reset(self) -> None:
        """Reset the patch generator state."""
        self.retry_count = 0
        self.patches = []

    def _execute_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Execute code in a safe environment to check for errors.

        Args:
            code: Code to execute

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            # First, try to parse the code to catch syntax errors
            ast.parse(code)

            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_file.write(code.encode('utf-8'))
                temp_path = temp_file.name

            try:
                # Execute the code in the sandbox with a timeout

                # Use subprocess to execute with timeout and isolation
                import subprocess
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5 second timeout
                )

                # Check for errors
                if result.returncode != 0:
                    return False, result.stderr

                return True, None
            except subprocess.TimeoutExpired:
                return False, "Code execution timed out (exceeded 5 seconds)"
            except Exception as e:
                return False, str(e)
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        except Exception as e:
            return False, str(e)

    def _fix_missing_parenthesis(self, code: str) -> str:
        """
        Fix missing parenthesis in code.

        Args:
            code: Code with missing parenthesis

        Returns:
            str: Fixed code
        """
        # Simple heuristic: count opening and closing parentheses
        open_count = code.count("(")
        close_count = code.count(")")

        if open_count > close_count:
            # Missing closing parenthesis
            return code + ")" * (open_count - close_count)
        elif close_count > open_count:
            # Missing opening parenthesis (less common)
            return "(" * (close_count - open_count) + code

        # If counts are equal, try to fix other common issues
        # Check for missing parenthesis in function calls
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "print" in line and "(" not in line:
                lines[i] = line.replace("print", "print(") + ")"

        return "\n".join(lines)

    def _fix_undefined_variable(self, code: str, variable: str) -> str:
        """
        Fix undefined variable in code by analyzing context and adding appropriate definition.

        Args:
            code: Code with undefined variable
            variable: Name of the undefined variable

        Returns:
            str: Fixed code with properly defined variable
        """
        lines = code.split("\n")

        # Try to analyze the code to understand the context
        try:
            tree = ast.parse(code)

            # Find where the variable is used and in what context
            variable_uses = []
            variable_type = None

            class VariableVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    if node.id == variable and isinstance(node.ctx, ast.Load):
                        variable_uses.append(node)
                    self.generic_visit(node)

                def visit_Call(self, node):
                    # Check if the variable is used in a function call
                    if isinstance(node.func, ast.Name) and node.func.id == variable:
                        nonlocal variable_type
                        variable_type = "function"
                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    # Check if the variable is used as an object with attributes
                    if isinstance(node.value, ast.Name) and node.value.id == variable:
                        nonlocal variable_type
                        variable_type = "object"
                    self.generic_visit(node)

                def visit_Subscript(self, node):
                    # Check if the variable is used as a container (list, dict, etc.)
                    if isinstance(node.value, ast.Name) and node.value.id == variable:
                        nonlocal variable_type
                        variable_type = "container"
                    self.generic_visit(node)

            visitor = VariableVisitor()
            visitor.visit(tree)

            # Determine the scope where the variable should be defined
            function_scopes = []
            class_scopes = []

            class ScopeVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    # Check if any variable uses are within this function
                    for var_node in variable_uses:
                        if (hasattr(var_node, 'lineno') and
                            node.lineno <= var_node.lineno <=
                            (node.end_lineno if hasattr(node, 'end_lineno') else float('inf'))):
                            function_scopes.append(node)
                            break
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    # Check if any variable uses are within this class
                    for var_node in variable_uses:
                        if (hasattr(var_node, 'lineno') and
                            node.lineno <= var_node.lineno <=
                            (node.end_lineno if hasattr(node, 'end_lineno') else float('inf'))):
                            class_scopes.append(node)
                            break
                    self.generic_visit(node)

            scope_visitor = ScopeVisitor()
            scope_visitor.visit(tree)

            # Determine the appropriate definition based on usage context
            if variable_type == "function":
                definition = f"{variable} = lambda *args, **kwargs: None  # TODO: Implement this function"
            elif variable_type == "object":
                definition = f"{variable} = type('DummyObject', (), {{}})()  # TODO: Replace with appropriate object"
            elif variable_type == "container":
                # Check if it's used like a dict or list
                is_dict = False
                for node in variable_uses:
                    parent = next((p for p in ast.walk(tree) if hasattr(p, 'value') and p.value == node), None)
                    if parent and isinstance(parent, ast.Subscript) and isinstance(parent.slice, ast.Constant):
                        if isinstance(parent.slice.value, str):
                            is_dict = True
                            break

                if is_dict:
                    definition = f"{variable} = {{}}  # TODO: Populate this dictionary"
                else:
                    definition = f"{variable} = []  # TODO: Populate this list"
            else:
                # Default to None for unknown types
                definition = f"{variable} = None  # TODO: Replace with appropriate value"

            # Insert the definition in the appropriate scope
            if function_scopes:
                # Variable is used in a function, insert at the beginning of the function
                func_node = function_scopes[0]
                func_line = func_node.lineno

                # Find the first line after the function definition
                for i in range(func_line, len(lines)):
                    if ":" in lines[i]:
                        indent = re.match(r"^(\s*)", lines[i]).group(1) + "    "
                        lines.insert(i + 1, f"{indent}{definition}")
                        return "\n".join(lines)

            elif class_scopes:
                # Variable is used in a class method, might be a missing self attribute
                class_node = class_scopes[0]

                # Check if it might be a self attribute
                is_self_attr = False
                for node in variable_uses:
                    parent = next((p for p in ast.walk(tree) if hasattr(p, 'body') and node in ast.walk(p)), None)
                    if parent and isinstance(parent, ast.FunctionDef):
                        if parent.args.args and parent.args.args[0].arg == 'self':
                            is_self_attr = True
                            break

                if is_self_attr:
                    # Find the __init__ method or create one
                    init_line = -1
                    init_indent = ""
                    for i, line in enumerate(lines):
                        if re.match(r"^\s*def\s+__init__\s*\(", line):
                            init_line = i
                            init_indent = re.match(r"^(\s*)", line).group(1) + "    "
                            break

                    if init_line >= 0:
                        # Find where to insert in __init__
                        for i in range(init_line, len(lines)):
                            if ":" in lines[i]:
                                lines.insert(i + 1, f"{init_indent}self.{variable} = None  # TODO: Initialize properly")
                                return "\n".join(lines)
                    else:
                        # Create __init__ method
                        class_line = class_node.lineno
                        class_indent = re.match(r"^(\s*)", lines[class_line-1]).group(1)
                        method_indent = class_indent + "    "
                        body_indent = method_indent + "    "

                        init_method = [
                            f"{method_indent}def __init__(self):",
                            f"{body_indent}super().__init__()",
                            f"{body_indent}self.{variable} = None  # TODO: Initialize properly"
                        ]

                        # Find where to insert the __init__ method
                        for i in range(class_line, len(lines)):
                            if ":" in lines[i]:
                                for j, line in enumerate(init_method):
                                    lines.insert(i + j + 1, line)
                                return "\n".join(lines)
                else:
                    # Regular class variable
                    class_line = class_node.lineno
                    class_indent = re.match(r"^(\s*)", lines[class_line-1]).group(1) + "    "

                    # Insert after class definition
                    for i in range(class_line, len(lines)):
                        if ":" in lines[i]:
                            lines.insert(i + 1, f"{class_indent}{definition}")
                            return "\n".join(lines)

            else:
                # Module-level variable, insert at an appropriate location
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(("import ", "from ")) or line.strip() == "":
                        import_end = i
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                # Insert after imports
                lines.insert(import_end + 1, definition)

        except (SyntaxError, Exception) as e:
            # Fallback for syntax errors or other issues
            logger.warning(f"Error analyzing code for undefined variable: {e}")

            # Simple heuristic: add a variable definition at the beginning
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(("#", "import", "from")):
                    lines.insert(i, f"{variable} = None  # TODO: Replace with appropriate value")
                    break
            else:
                lines.insert(0, f"{variable} = None  # TODO: Replace with appropriate value")

        return "\n".join(lines)

    def _fix_indentation(self, code: str) -> str:
        """
        Fix indentation errors in code.

        Args:
            code: Code with indentation errors

        Returns:
            str: Fixed code
        """
        lines = code.split("\n")
        fixed_lines = []

        # Track expected indentation level
        expected_indent = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue

            # Check if this line should increase indentation for the next line
            if stripped.endswith(":"):
                fixed_lines.append(line)
                expected_indent += 4

                # Special case for "expected an indented block" error
                # If the next line exists and is not indented, fix it
                if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(" "):
                    # Don't process this line now, it will be handled in the next iteration
                    continue

            # Get current indentation
            current_indent = len(line) - len(line.lstrip())

            # If indentation is less than expected and this is not a dedent line
            if current_indent < expected_indent and not (
                stripped.startswith("else:")
                or stripped.startswith("elif ")
                or stripped.startswith("except:")
                or stripped.startswith("except ")
                or stripped.startswith("finally:")
            ):
                # Fix indentation
                fixed_lines.append(" " * expected_indent + stripped)
            else:
                fixed_lines.append(line)

                # Update expected indentation if this is a dedent line
                if (
                    stripped.startswith("else:")
                    or stripped.startswith("elif ")
                    or stripped.startswith("except:")
                    or stripped.startswith("except ")
                    or stripped.startswith("finally:")
                ):
                    expected_indent = current_indent

        return "\n".join(fixed_lines)

    def _fix_invalid_syntax(self, code: str) -> str:
        """
        Fix invalid syntax in code by analyzing common syntax errors.

        Args:
            code: Code with invalid syntax

        Returns:
            str: Fixed code with proper syntax
        """
        lines = code.split("\n")
        fixed_lines = []

        # Track indentation and bracket levels
        expected_indent = 0
        bracket_stack = []
        paren_stack = []
        brace_stack = []

        # Track string literals
        in_string = False
        string_char = None

        # Track line continuations
        continued_line = False

        for i, line in enumerate(lines):
            original_line = line
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue

            # Handle string continuations
            if in_string:
                # We're in a multi-line string, just add the line as is
                fixed_lines.append(line)

                # Check if the string ends on this line
                for j, char in enumerate(line):
                    if char == string_char and (j == 0 or line[j-1] != '\\'):
                        in_string = False
                        string_char = None
                        break

                continue

            # Check for string start
            for j, char in enumerate(line):
                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
                    # Check if it's a triple quote
                    if j + 2 < len(line) and line[j:j+3] in ['"""', "'''"]:
                        if not in_string:
                            in_string = True
                            string_char = line[j]
                        elif string_char == line[j] and line[j:j+3] in ['"""', "'''"]:
                            in_string = False
                            string_char = None
                    # Single quote
                    elif not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

            # If we're in a string, continue to next line
            if in_string:
                fixed_lines.append(line)
                continue

            # Fix missing colons in if/for/while/def/class statements
            if re.search(r"^\s*(if|for|while|def|class)\s+.*[^:]\s*$", line):
                line = line + ":"

            # Fix common comparison operator typos
            line = re.sub(r'\s+=\s+=\s+', ' == ', line)
            line = re.sub(r'\s+!\s+=\s+', ' != ', line)
            line = re.sub(r'\s+<\s+=\s+', ' <= ', line)
            line = re.sub(r'\s+>\s+=\s+', ' >= ', line)

            # Fix missing parentheses in print statements (Python 3)
            if re.search(r'^\s*print\s+[^(]', line):
                # Don't match if it's already a valid print function call
                if not re.search(r'^\s*print\s*\(', line):
                    content = line.split('print', 1)[1].strip()
                    line = line.split('print', 1)[0] + 'print(' + content + ')'

            # Fix missing commas in lists, tuples, and dicts
            if '[' in line or '(' in line or '{' in line:
                # Track brackets for this line
                line_brackets = []
                in_container = False
                container_type = None

                for j, char in enumerate(line):
                    if char in '([{':
                        line_brackets.append(char)
                        if not in_container:
                            in_container = True
                            container_type = char
                    elif char in ')]}':
                        if line_brackets and ((char == ')' and line_brackets[-1] == '(') or
                                             (char == ']' and line_brackets[-1] == '[') or
                                             (char == '}' and line_brackets[-1] == '{')):
                            line_brackets.pop()
                            if not line_brackets:
                                in_container = False
                                container_type = None

                    # Check for missing commas between items
                    if in_container and j > 0 and j < len(line) - 1:
                        prev_char = line[j-1]
                        next_char = line[j+1]

                        # Look for patterns like "item1" "item2" or 1 2 or True False
                        if ((prev_char.isalnum() or prev_char in '"\']})') and
                            (char.isspace() or char == '#') and
                            (next_char.isalnum() or next_char in '"\'[({') and
                            char != ',' and char != ':'):

                            # Don't add comma if it's a dict key-value pair
                            if not (container_type == '{' and ':' in line[j:]):
                                line = line[:j] + ',' + line[j:]

            # Fix unclosed brackets/parentheses/braces
            for char in line:
                if char == '(':
                    paren_stack.append(char)
                elif char == ')':
                    if paren_stack:
                        paren_stack.pop()
                elif char == '[':
                    bracket_stack.append(char)
                elif char == ']':
                    if bracket_stack:
                        bracket_stack.pop()
                elif char == '{':
                    brace_stack.append(char)
                elif char == '}':
                    if brace_stack:
                        brace_stack.pop()

            # Check if this line ends with a continuation character
            if stripped.endswith('\\'):
                continued_line = True
            else:
                continued_line = False

            # Fix indentation issues
            current_indent = len(line) - len(line.lstrip())
            if current_indent != expected_indent and not continued_line:
                # Only fix indentation if it's clearly wrong
                if abs(current_indent - expected_indent) % 4 == 0:
                    line = ' ' * expected_indent + line.lstrip()

            # Update expected indentation for next line
            if stripped.endswith(':'):
                expected_indent = current_indent + 4

            # Add the fixed line
            fixed_lines.append(line)

            # Check if the line is significantly different and add a comment
            if line != original_line:
                # Add a comment explaining the fix
                indent = re.match(r'^(\s*)', line).group(1)
                fixed_lines.append(f"{indent}# Fixed syntax: {original_line.strip()}")

        # Fix unclosed brackets at the end of the file
        if paren_stack or bracket_stack or brace_stack:
            # Add closing brackets
            closing_line = ""
            for _ in range(len(paren_stack)):
                closing_line += ")"
            for _ in range(len(bracket_stack)):
                closing_line += "]"
            for _ in range(len(brace_stack)):
                closing_line += "}"

            if closing_line:
                fixed_lines.append(closing_line + "  # Added missing closing brackets")

        return "\n".join(fixed_lines)

    def _fix_argument_error(self, code: str, error: str) -> str:
        """
        Fix argument errors in function calls by analyzing the error and function signature.

        Args:
            code: Code with argument errors
            error: Error message

        Returns:
            str: Fixed code with proper function arguments
        """
        lines = code.split("\n")

        # Common error patterns
        too_many_args_pattern = re.search(r"([a-zA-Z0-9_]+)\(\) takes (\d+) .* but (\d+) .* given", error)
        missing_required_pattern = re.search(r"([a-zA-Z0-9_]+)\(\) missing (\d+) required positional argument", error)
        unexpected_keyword_pattern = re.search(r"([a-zA-Z0-9_]+)\(\) got an unexpected keyword argument '([^']+)'", error)
        multiple_values_pattern = re.search(r"([a-zA-Z0-9_]+)\(\) got multiple values for argument '([^']+)'", error)

        try:
            # Parse the code to analyze it
            tree = ast.parse(code)

            # Find the problematic line number if available
            line_num_match = re.search(r"line (\d+)", error)
            line_idx = None
            if line_num_match:
                try:
                    line_num = int(line_num_match.group(1))
                    # Adjust for 0-based indexing
                    if 1 <= line_num <= len(lines):
                        line_idx = line_num - 1
                except ValueError:
                    pass

            # Handle too many/few arguments error
            if too_many_args_pattern:
                func_name = too_many_args_pattern.group(1)
                expected_args = int(too_many_args_pattern.group(2))
                given_args = int(too_many_args_pattern.group(3))

                # Try to find the function definition to get parameter names
                func_def = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        func_def = node
                        break

                # Find the function call
                func_calls = []
                for i, line in enumerate(lines):
                    if func_name in line and "(" in line and ")" in line:
                        # Extract the function call
                        call_match = re.search(rf"{func_name}\s*\((.*?)\)", line)
                        if call_match:
                            func_calls.append((i, call_match))

                # If we have a specific line number, filter to that call
                if line_idx is not None:
                    func_calls = [(i, match) for i, match in func_calls if i == line_idx]

                for i, call_match in func_calls:
                    # Parse the arguments
                    args_str = call_match.group(1)

                    # Handle complex argument parsing with potential nested parentheses
                    args = []
                    if args_str.strip():
                        # Simple case: no nested parentheses
                        if "(" not in args_str or ")" not in args_str:
                            args = [arg.strip() for arg in args_str.split(",")]
                        else:
                            # Complex case: handle nested parentheses
                            current_arg = ""
                            paren_level = 0

                            for char in args_str:
                                if char == "," and paren_level == 0:
                                    args.append(current_arg.strip())
                                    current_arg = ""
                                else:
                                    if char == "(":
                                        paren_level += 1
                                    elif char == ")":
                                        paren_level -= 1
                                    current_arg += char

                            if current_arg:
                                args.append(current_arg.strip())

                    # Separate positional and keyword arguments
                    pos_args = []
                    kw_args = []

                    for arg in args:
                        if "=" in arg:
                            kw_args.append(arg)
                        else:
                            pos_args.append(arg)

                    if given_args > expected_args:
                        # Too many arguments, need to fix

                        if func_def:
                            # We have the function definition, so we can be smarter about which args to keep
                            param_names = [arg.arg for arg in func_def.args.args]

                            # Keep required positional arguments
                            new_pos_args = pos_args[:len(param_names)]

                            # Keep valid keyword arguments
                            new_kw_args = []
                            for kw_arg in kw_args:
                                name = kw_arg.split("=")[0].strip()
                                if name in param_names:
                                    new_kw_args.append(kw_arg)

                            # Combine arguments
                            new_args = new_pos_args + new_kw_args
                        else:
                            # We don't have the function definition, so just keep the first expected_args
                            new_args = args[:expected_args]

                        # Replace the arguments in the line
                        new_args_str = ", ".join(new_args)
                        new_line = line.replace(args_str, new_args_str)
                        lines[i] = new_line

                    elif given_args < expected_args:
                        # Too few arguments, need to add some

                        if func_def:
                            # We have the function definition, so we can add the missing parameters with appropriate names
                            param_names = [arg.arg for arg in func_def.args.args]

                            # Find which parameters are missing
                            provided_pos_count = len(pos_args)
                            provided_kw_names = [kw_arg.split("=")[0].strip() for kw_arg in kw_args]

                            # Add missing positional parameters
                            for j in range(provided_pos_count, len(param_names)):
                                param_name = param_names[j]
                                if param_name not in provided_kw_names:
                                    # Add as a keyword argument
                                    kw_args.append(f"{param_name}=None  # TODO: Provide appropriate value")

                            # Combine arguments
                            new_args = pos_args + kw_args
                        else:
                            # We don't have the function definition, so just add None for missing args
                            for _ in range(expected_args - given_args):
                                args.append("None  # TODO: Provide appropriate value")
                            new_args = args

                        # Replace the arguments in the line
                        new_args_str = ", ".join(new_args)
                        new_line = line.replace(args_str, new_args_str)
                        lines[i] = new_line

                return "\n".join(lines)

            # Handle missing required argument error
            elif missing_required_pattern:
                func_name = missing_required_pattern.group(1)
                missing_count = int(missing_required_pattern.group(2))

                # Extract the missing argument name if available
                missing_arg_match = re.search(r"'([^']+)'", error)
                missing_arg = missing_arg_match.group(1) if missing_arg_match else None

                # Find the function call
                func_calls = []
                for i, line in enumerate(lines):
                    if func_name in line and "(" in line and ")" in line:
                        call_match = re.search(rf"{func_name}\s*\((.*?)\)", line)
                        if call_match:
                            func_calls.append((i, call_match))

                # If we have a specific line number, filter to that call
                if line_idx is not None:
                    func_calls = [(i, match) for i, match in func_calls if i == line_idx]

                for i, call_match in func_calls:
                    args_str = call_match.group(1)

                    # Add the missing argument
                    if missing_arg:
                        if args_str.strip():
                            new_args_str = f"{args_str}, {missing_arg}=None  # TODO: Provide appropriate value"
                        else:
                            new_args_str = f"{missing_arg}=None  # TODO: Provide appropriate value"
                    else:
                        # We don't know the argument name, so just add placeholders
                        if args_str.strip():
                            new_args_str = args_str
                            for j in range(missing_count):
                                new_args_str += f", arg{j+1}=None  # TODO: Provide appropriate value"
                        else:
                            new_args_str = ", ".join([f"arg{j+1}=None  # TODO: Provide appropriate value" for j in range(missing_count)])

                    # Replace the arguments in the line
                    new_line = line.replace(args_str, new_args_str)
                    lines[i] = new_line

                return "\n".join(lines)

            # Handle unexpected keyword argument error
            elif unexpected_keyword_pattern:
                func_name = unexpected_keyword_pattern.group(1)
                bad_keyword = unexpected_keyword_pattern.group(2)

                # Find the function call
                func_calls = []
                for i, line in enumerate(lines):
                    if func_name in line and "(" in line and ")" in line and bad_keyword in line:
                        call_match = re.search(rf"{func_name}\s*\((.*?)\)", line)
                        if call_match:
                            func_calls.append((i, call_match))

                # If we have a specific line number, filter to that call
                if line_idx is not None:
                    func_calls = [(i, match) for i, match in func_calls if i == line_idx]

                for i, call_match in func_calls:
                    args_str = call_match.group(1)

                    # Parse the arguments
                    args = [arg.strip() for arg in args_str.split(",")]

                    # Remove the bad keyword argument
                    new_args = []
                    for arg in args:
                        if not arg.startswith(f"{bad_keyword}="):
                            new_args.append(arg)

                    # Replace the arguments in the line
                    new_args_str = ", ".join(new_args)
                    new_line = line.replace(args_str, new_args_str)
                    lines[i] = new_line

                    # Add a comment explaining the removal
                    indent = re.match(r"^(\s*)", line).group(1)
                    lines.insert(i, f"{indent}# Removed invalid keyword argument: {bad_keyword}")

                return "\n".join(lines)

            # Handle multiple values for argument error
            elif multiple_values_pattern:
                func_name = multiple_values_pattern.group(1)
                duplicate_arg = multiple_values_pattern.group(2)

                # Find the function call
                func_calls = []
                for i, line in enumerate(lines):
                    if func_name in line and "(" in line and ")" in line and duplicate_arg in line:
                        call_match = re.search(rf"{func_name}\s*\((.*?)\)", line)
                        if call_match:
                            func_calls.append((i, call_match))

                # If we have a specific line number, filter to that call
                if line_idx is not None:
                    func_calls = [(i, match) for i, match in func_calls if i == line_idx]

                for i, call_match in func_calls:
                    args_str = call_match.group(1)

                    # Parse the arguments
                    args = [arg.strip() for arg in args_str.split(",")]

                    # Find positional and keyword occurrences of the argument
                    pos_indices = []
                    kw_index = -1

                    # Try to find the function definition to get parameter positions
                    param_positions = {}
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == func_name:
                            for j, arg in enumerate(node.args.args):
                                param_positions[arg.arg] = j
                            break

                    # If we know the position of the duplicate argument, we can be smarter
                    if duplicate_arg in param_positions:
                        pos_index = param_positions[duplicate_arg]

                        # Check if we have a positional argument at that position
                        if pos_index < len(args) and "=" not in args[pos_index]:
                            pos_indices.append(pos_index)

                    # Check for keyword argument
                    for j, arg in enumerate(args):
                        if arg.startswith(f"{duplicate_arg}="):
                            kw_index = j
                            break

                    # Keep the keyword argument if it exists, otherwise keep the last positional
                    new_args = list(args)
                    if kw_index >= 0 and pos_indices:
                        # Remove the positional arguments
                        for idx in sorted(pos_indices, reverse=True):
                            new_args.pop(idx)
                    elif len(pos_indices) > 1:
                        # Keep only the last positional argument
                        for idx in sorted(pos_indices[:-1], reverse=True):
                            new_args.pop(idx)

                    # Replace the arguments in the line
                    new_args_str = ", ".join(new_args)
                    new_line = line.replace(args_str, new_args_str)
                    lines[i] = new_line

                    # Add a comment explaining the change
                    indent = re.match(r"^(\s*)", line).group(1)
                    lines.insert(i, f"{indent}# Fixed multiple values for argument: {duplicate_arg}")

                return "\n".join(lines)

            # Generic argument error handling
            else:
                # Extract function name from error if possible
                func_name_match = re.search(r"([a-zA-Z0-9_]+)\(", error)
                if func_name_match:
                    func_name = func_name_match.group(1)

                    # Find the function call
                    for i, line in enumerate(lines):
                        if func_name in line and "(" in line and ")" in line:
                            # If we have a line number and it doesn't match, skip
                            if line_idx is not None and i != line_idx:
                                continue

                            # Add a try-except block around the function call
                            indent = re.match(r"^(\s*)", line).group(1)
                            try_block = [
                                f"{indent}try:",
                                f"{indent}    {line}",
                                f"{indent}except TypeError as e:",
                                f"{indent}    print(f\"Argument error: {{e}}\")",
                                f"{indent}    # TODO: Fix the function arguments"
                            ]
                            lines[i:i+1] = try_block
                            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Error analyzing code for argument error: {e}")

        # Fallback to basic argument count fixing
        match = re.search(r"([a-zA-Z0-9_]+)\(\) takes (\d+) .* but (\d+) .* given", error)
        if match:
            func_name = match.group(1)
            expected_args = int(match.group(2))
            given_args = int(match.group(3))

            # Find the function call
            for i, line in enumerate(lines):
                if func_name in line and "(" in line and ")" in line:
                    # Extract the function call
                    call_match = re.search(rf"{func_name}\((.*?)\)", line)
                    if call_match:
                        args_str = call_match.group(1)
                        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

                        if given_args > expected_args:
                            # Too many arguments, remove some
                            args = args[:expected_args]
                            new_args_str = ", ".join(args)
                            lines[i] = line.replace(call_match.group(1), new_args_str)
                        elif given_args < expected_args:
                            # Too few arguments, add some
                            for _ in range(expected_args - given_args):
                                args.append("None  # TODO: Provide appropriate value")
                            new_args_str = ", ".join(args)
                            lines[i] = line.replace(call_match.group(1), new_args_str)

        return "\n".join(lines)

    def _fix_type_mismatch(self, code: str, error: str) -> str:
        """
        Fix type mismatch errors by analyzing the error and code context.

        Args:
            code: Code with type mismatch errors
            error: Error message

        Returns:
            str: Fixed code with proper type conversions
        """
        lines = code.split("\n")

        try:
            # Extract information from the error message
            type_info = {}

            # Common error patterns
            str_int_pattern = re.search(r"(str|int|float|list|dict|tuple|bool|set)\s+(?:and|,)\s+(str|int|float|list|dict|tuple|bool|set)", error)
            cannot_convert_pattern = re.search(r"cannot convert (.*?) to (.*?)( implicitly)?", error)
            expected_got_pattern = re.search(r"expected (.*?), got (.*)", error)

            if str_int_pattern:
                type_info["types"] = [str_int_pattern.group(1), str_int_pattern.group(2)]
            elif cannot_convert_pattern:
                type_info["from_type"] = cannot_convert_pattern.group(1)
                type_info["to_type"] = cannot_convert_pattern.group(2)
            elif expected_got_pattern:
                type_info["expected"] = expected_got_pattern.group(1)
                type_info["got"] = expected_got_pattern.group(2)

            # Find the problematic line number if available
            line_num_match = re.search(r"line (\d+)", error)
            if line_num_match:
                try:
                    line_num = int(line_num_match.group(1))
                    # Adjust for 0-based indexing
                    if 1 <= line_num <= len(lines):
                        type_info["line_num"] = line_num - 1
                except ValueError:
                    pass

            # Define common type conversion functions for reference
            # Note: This dictionary is not currently used but kept for future implementation
            # that may need to dynamically select conversion functions

            # Handle specific type mismatch scenarios
            if "str" in error and ("int" in error or "float" in error):
                # String and number type mismatch

                # Case 1: String concatenation with numbers
                for i, line in enumerate(lines):
                    if "+" in line and not re.search(r'"\s*\+\s*"', line) and not re.search(r"'\s*\+\s*'", line):
                        # Check for string concatenation with numbers
                        parts = []
                        current_part = ""
                        in_string = False
                        string_char = None

                        # Parse the line more carefully to handle strings with + in them
                        for char in line:
                            if char in ['"', "'"]:
                                if not in_string:
                                    in_string = True
                                    string_char = char
                                    current_part += char
                                elif char == string_char:
                                    in_string = False
                                    string_char = None
                                    current_part += char
                                else:
                                    current_part += char
                            elif char == "+" and not in_string:
                                parts.append(current_part)
                                current_part = ""
                            else:
                                current_part += char

                        if current_part:
                            parts.append(current_part)

                        # Check each part for numbers that need conversion
                        for j, part in enumerate(parts):
                            part = part.strip()
                            # Check if it's a numeric literal
                            if re.match(r'^-?\d+(\.\d+)?$', part):
                                parts[j] = f"str({part})"
                            # Check if it's a variable that might need conversion
                            elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                                # This is a variable name, might need conversion
                                # Look for variable definitions to determine type
                                var_name = part
                                var_type = self._infer_variable_type(code, var_name)
                                if var_type in ["int", "float"]:
                                    parts[j] = f"str({part})"

                        # Reconstruct the line
                        lines[i] = "+".join(parts)

                # Case 2: Function expecting string but got number or vice versa
                if "line_num" in type_info:
                    line_idx = type_info["line_num"]
                    line = lines[line_idx]

                    # Look for function calls
                    func_call_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', line)
                    if func_call_match:
                        func_name = func_call_match.group(1)
                        args = func_call_match.group(2)

                        # Common functions that expect specific types
                        if func_name == "int" and ("str" in type_info.get("from_type", "")):
                            # Trying to convert non-numeric string to int
                            # Add error handling
                            indent = re.match(r'^(\s*)', line).group(1)
                            try_block = [
                                f"{indent}try:",
                                f"{indent}    {line}",
                                f"{indent}except ValueError:",
                                f"{indent}    print(f\"Error: Could not convert to integer. Using default value.\")",
                                f"{indent}    # TODO: Handle the error appropriately"
                            ]
                            lines[line_idx:line_idx+1] = try_block
                            return "\n".join(lines)

                        elif func_name in ["open", "read", "write"] and "int" in type_info.get("got", ""):
                            # File operations expecting string but got number
                            new_args = []
                            for arg in args.split(","):
                                arg = arg.strip()
                                if re.match(r'^-?\d+(\.\d+)?$', arg):
                                    new_args.append(f"str({arg})")
                                else:
                                    new_args.append(arg)
                            new_line = line.replace(args, ", ".join(new_args))
                            lines[line_idx] = new_line
                            return "\n".join(lines)

            elif "list" in error and "str" in error:
                # List and string type mismatch
                for i, line in enumerate(lines):
                    # Check for attempts to index into a string as if it were a list
                    list_index_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]', line)
                    if list_index_match:
                        var_name = list_index_match.group(1)
                        var_type = self._infer_variable_type(code, var_name)
                        if var_type == "str":
                            # Convert string indexing to list indexing
                            lines[i] = line.replace(f"{var_name}[", f"list({var_name})[")
                            return "\n".join(lines)

            elif "NoneType" in error:
                # None type errors
                for i, line in enumerate(lines):
                    # Check for operations on potentially None variables
                    for var_match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)(\.|\[|\()', line):
                        var_name = var_match.group(1)
                        operation = var_match.group(2)

                        # Add None check
                        indent = re.match(r'^(\s*)', line).group(1)
                        if operation == ".":
                            # Attribute access on potentially None
                            try_block = [
                                f"{indent}if {var_name} is not None:",
                                f"{indent}    {line}",
                                f"{indent}else:",
                                f"{indent}    print(f\"Error: {var_name} is None. Cannot perform operation.\")",
                                f"{indent}    # TODO: Initialize {var_name} properly"
                            ]
                            lines[i:i+1] = try_block
                            return "\n".join(lines)

            # If we couldn't apply a specific fix, try a generic approach
            if "line_num" in type_info:
                line_idx = type_info["line_num"]
                line = lines[line_idx]

                # Add a generic try-except block
                indent = re.match(r'^(\s*)', line).group(1)
                try_block = [
                    f"{indent}try:",
                    f"{indent}    {line}",
                    f"{indent}except TypeError as e:",
                    f"{indent}    print(f\"Type error: {{e}}. Check variable types.\")",
                    f"{indent}    # TODO: Fix the type mismatch"
                ]
                lines[line_idx:line_idx+1] = try_block
                return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Error analyzing code for type mismatch: {e}")

        # Fallback to basic string/int conversion if we couldn't apply a more specific fix
        if "str" in error and "int" in error:
            for i, line in enumerate(lines):
                if "+" in line:
                    # Check for string concatenation with numbers
                    parts = line.split("+")
                    for j, part in enumerate(parts):
                        if part.strip().isdigit():
                            parts[j] = f"str({part.strip()})"
                    lines[i] = "+".join(parts)

        return "\n".join(lines)

    def _infer_variable_type(self, code: str, variable: str) -> str:
        """
        Infer the type of a variable from the code.

        Args:
            code: The code to analyze
            variable: The variable name

        Returns:
            str: The inferred type ("int", "str", etc.) or "unknown"
        """
        try:
            tree = ast.parse(code)

            # Look for assignments to the variable
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == variable:
                            # Found an assignment to the variable
                            if isinstance(node.value, ast.Constant):
                                # Direct assignment of a constant
                                if isinstance(node.value.value, int):
                                    return "int"
                                elif isinstance(node.value.value, float):
                                    return "float"
                                elif isinstance(node.value.value, str):
                                    return "str"
                                elif isinstance(node.value.value, bool):
                                    return "bool"
                                elif node.value.value is None:
                                    return "None"
                            elif isinstance(node.value, ast.List):
                                return "list"
                            elif isinstance(node.value, ast.Dict):
                                return "dict"
                            elif isinstance(node.value, ast.Tuple):
                                return "tuple"
                            elif isinstance(node.value, ast.Set):
                                return "set"
                            elif isinstance(node.value, ast.Call):
                                if isinstance(node.value.func, ast.Name):
                                    # Function call like int(), str(), etc.
                                    return node.value.func.id

            # If we couldn't determine from assignments, look at how it's used
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    # Check binary operations for clues
                    if (isinstance(node.left, ast.Name) and node.left.id == variable) or \
                       (isinstance(node.right, ast.Name) and node.right.id == variable):
                        # Variable used in an operation
                        if isinstance(node.op, ast.Add):
                            # Addition could be string concatenation or numeric addition
                            other_side = node.right if isinstance(node.left, ast.Name) and node.left.id == variable else node.left
                            if isinstance(other_side, ast.Constant):
                                if isinstance(other_side.value, str):
                                    return "str"
                                elif isinstance(other_side.value, (int, float)):
                                    return "numeric"
                        elif isinstance(node.op, (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                            # These operations suggest numeric types
                            return "numeric"

                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == variable:
                    # Variable used as a container (list, dict, etc.)
                    return "container"

                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == variable:
                    # Variable used as a function
                    return "function"

        except Exception:
            pass

        return "unknown"

    def _fix_missing_module(self, code: str, module: str) -> str:
        """
        Fix missing module errors by adding proper import handling.

        Args:
            code: Code with missing module error
            module: Name of the missing module

        Returns:
            str: Fixed code with proper module handling
        """
        lines = code.split("\n")

        # Common module mappings for standard library and popular packages
        std_lib_modules = {
            "math", "os", "sys", "datetime", "time", "random", "json",
            "re", "collections", "itertools", "functools", "pathlib"
        }

        popular_modules = {
            "numpy": "np",
            "pandas": "pd",
            "matplotlib.pyplot": "plt",
            "tensorflow": "tf",
            "torch": "torch",
            "sklearn": "sklearn",
            "requests": "requests",
            "bs4": "BeautifulSoup",
            "django": "django",
            "flask": "Flask",
            "sqlalchemy": "sqlalchemy",
            "pytest": "pytest"
        }

        # Check if the module is a submodule
        module_parts = module.split('.')
        base_module = module_parts[0]

        # Find the import statement or where it should be inserted
        import_line = -1
        for i, line in enumerate(lines):
            if f"import {module}" in line or f"from {module}" in line:
                import_line = i
                break
            elif line.startswith("import ") or line.startswith("from "):
                import_line = i

        # If no import statements found, find the first non-comment, non-docstring line
        if import_line == -1:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                    import_line = i
                    break

        # If still not found, insert at the beginning
        if import_line == -1:
            import_line = 0

        # Determine the appropriate import statement
        if base_module in std_lib_modules:
            # Standard library module
            import_statement = f"import {module}"
        elif base_module in popular_modules:
            # Popular module with common alias
            if '.' in module:
                # Submodule of a popular package
                import_statement = f"from {base_module} import {'.'.join(module_parts[1:])}"
            else:
                # Main module
                alias = popular_modules[base_module]
                import_statement = f"import {module} as {alias}"
        else:
            # Unknown module, use standard import
            import_statement = f"import {module}"

        # Add try-except block for graceful handling
        try_except_block = [
            f"try:",
            f"    {import_statement}",
            f"except ImportError:",
            f"    print(f\"Error: The '{module}' module is required but not installed.\")",
            f"    print(f\"Please install it using: pip install {base_module}\")",
            f"    import sys",
            f"    sys.exit(1)"
        ]

        # Insert the try-except block
        for i, line in enumerate(try_except_block):
            lines.insert(import_line + i, line)

        return "\n".join(lines)
