"""
Tool validator module for Saplings tool factory.

This module provides validation capabilities for dynamically generated tools.
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from saplings.tool_factory.config import SecurityLevel, ToolFactoryConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a tool validation."""
    
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.warnings is None:
            self.warnings = []


class ToolValidator:
    """
    Validator for dynamically generated tools.
    
    This class provides functionality for validating tool code to ensure it
    meets security and quality standards.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the tool validator.
        
        Args:
            config: Configuration for the validator
        """
        self.config = config or ToolFactoryConfig()
        
        # Define patterns to check based on security level
        self.dangerous_imports = [
            "os",
            "sys",
            "subprocess",
            "shutil",
            "pathlib",
        ]
        
        self.dangerous_functions = [
            "eval",
            "exec",
            "compile",
            "open",
            "__import__",
        ]
        
        self.network_patterns = [
            "socket",
            "urllib",
            "requests",
            "http",
            "ftp",
        ]
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate tool code.
        
        Args:
            code: Code to validate
            
        Returns:
            ValidationResult: Result of the validation
        """
        # Check for syntax errors
        syntax_result = self._check_syntax(code)
        if not syntax_result.is_valid:
            return syntax_result
        
        # Check for security issues
        security_result = self._check_security(code)
        if not security_result.is_valid:
            return security_result
        
        # Check for quality issues
        quality_result = self._check_quality(code)
        if not quality_result.is_valid:
            return quality_result
        
        # Combine warnings
        warnings = []
        warnings.extend(syntax_result.warnings)
        warnings.extend(security_result.warnings)
        warnings.extend(quality_result.warnings)
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def _check_syntax(self, code: str) -> ValidationResult:
        """
        Check for syntax errors in the code.
        
        Args:
            code: Code to check
            
        Returns:
            ValidationResult: Result of the syntax check
        """
        try:
            ast.parse(code)
            return ValidationResult(is_valid=True)
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Syntax error: {str(e)}",
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Parsing error: {str(e)}",
            )
    
    def _check_security(self, code: str) -> ValidationResult:
        """
        Check for security issues in the code.
        
        Args:
            code: Code to check
            
        Returns:
            ValidationResult: Result of the security check
        """
        warnings = []
        
        # Check for dangerous imports
        for imp in self.dangerous_imports:
            if re.search(rf"import\s+{imp}|from\s+{imp}\s+import", code):
                if self.config.security_level == SecurityLevel.HIGH:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Dangerous import: {imp}",
                    )
                elif self.config.security_level == SecurityLevel.MEDIUM:
                    warnings.append(f"Potentially dangerous import: {imp}")
        
        # Check for dangerous functions
        for func in self.dangerous_functions:
            if re.search(rf"\b{func}\s*\(", code):
                if self.config.security_level in [SecurityLevel.MEDIUM, SecurityLevel.HIGH]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Dangerous function call: {func}",
                    )
                elif self.config.security_level == SecurityLevel.LOW:
                    warnings.append(f"Potentially dangerous function call: {func}")
        
        # Additional checks for HIGH security level
        if self.config.security_level == SecurityLevel.HIGH:
            # Check for network access
            for pattern in self.network_patterns:
                if re.search(rf"import\s+{pattern}|from\s+{pattern}\s+import", code):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Network access attempt: {pattern}",
                    )
        
        # Perform AST-based security checks
        try:
            tree = ast.parse(code)
            visitor = SecurityVisitor(self.config)
            visitor.visit(tree)
            
            if visitor.errors:
                return ValidationResult(
                    is_valid=False,
                    error_message=visitor.errors[0],
                )
            
            warnings.extend(visitor.warnings)
        except Exception as e:
            logger.warning(f"Error during AST-based security check: {e}")
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def _check_quality(self, code: str) -> ValidationResult:
        """
        Check for code quality issues.
        
        Args:
            code: Code to check
            
        Returns:
            ValidationResult: Result of the quality check
        """
        warnings = []
        
        # Check for long lines
        for i, line in enumerate(code.splitlines()):
            if len(line) > 100:
                warnings.append(f"Line {i+1} is too long ({len(line)} characters)")
        
        # Check for too many nested blocks
        try:
            tree = ast.parse(code)
            visitor = QualityVisitor()
            visitor.visit(tree)
            warnings.extend(visitor.warnings)
        except Exception as e:
            logger.warning(f"Error during AST-based quality check: {e}")
        
        return ValidationResult(is_valid=True, warnings=warnings)


class SecurityVisitor(ast.NodeVisitor):
    """AST visitor for security checks."""
    
    def __init__(self, config: ToolFactoryConfig):
        """
        Initialize the security visitor.
        
        Args:
            config: Configuration for the validator
        """
        self.config = config
        self.errors = []
        self.warnings = []
        self.imported_modules = set()
    
    def visit_Import(self, node):
        """Visit an Import node."""
        for name in node.names:
            self.imported_modules.add(name.name.split(".")[0])
            
            # Check for dangerous imports
            if name.name in ["os", "sys", "subprocess", "shutil"]:
                if self.config.security_level == SecurityLevel.HIGH:
                    self.errors.append(f"Dangerous import: {name.name}")
                elif self.config.security_level == SecurityLevel.MEDIUM:
                    self.warnings.append(f"Potentially dangerous import: {name.name}")
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit an ImportFrom node."""
        if node.module:
            self.imported_modules.add(node.module.split(".")[0])
            
            # Check for dangerous imports
            if node.module in ["os", "sys", "subprocess", "shutil"]:
                if self.config.security_level == SecurityLevel.HIGH:
                    self.errors.append(f"Dangerous import: {node.module}")
                elif self.config.security_level == SecurityLevel.MEDIUM:
                    self.warnings.append(f"Potentially dangerous import: {node.module}")
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit a Call node."""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in ["eval", "exec", "compile", "__import__"]:
                if self.config.security_level in [SecurityLevel.MEDIUM, SecurityLevel.HIGH]:
                    self.errors.append(f"Dangerous function call: {node.func.id}")
                elif self.config.security_level == SecurityLevel.LOW:
                    self.warnings.append(f"Potentially dangerous function call: {node.func.id}")
        
        # Check for file operations
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ["open", "write", "read", "unlink", "remove"]:
                if self.config.security_level == SecurityLevel.HIGH:
                    self.errors.append(f"File operation: {node.func.attr}")
                elif self.config.security_level == SecurityLevel.MEDIUM:
                    self.warnings.append(f"File operation: {node.func.attr}")
        
        self.generic_visit(node)


class QualityVisitor(ast.NodeVisitor):
    """AST visitor for quality checks."""
    
    def __init__(self):
        """Initialize the quality visitor."""
        self.warnings = []
        self.nesting_level = 0
        self.max_nesting_level = 0
        self.function_complexity = {}
    
    def visit_FunctionDef(self, node):
        """Visit a FunctionDef node."""
        # Reset nesting level for each function
        old_nesting_level = self.nesting_level
        self.nesting_level = 0
        
        # Visit the function body
        self.generic_visit(node)
        
        # Store the function complexity
        self.function_complexity[node.name] = self.max_nesting_level
        
        # Check for excessive complexity
        if self.max_nesting_level > 5:
            self.warnings.append(
                f"Function '{node.name}' has excessive nesting (level {self.max_nesting_level})"
            )
        
        # Restore the nesting level
        self.nesting_level = old_nesting_level
    
    def visit_If(self, node):
        """Visit an If node."""
        self.nesting_level += 1
        self.max_nesting_level = max(self.max_nesting_level, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_For(self, node):
        """Visit a For node."""
        self.nesting_level += 1
        self.max_nesting_level = max(self.max_nesting_level, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_While(self, node):
        """Visit a While node."""
        self.nesting_level += 1
        self.max_nesting_level = max(self.max_nesting_level, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_Try(self, node):
        """Visit a Try node."""
        self.nesting_level += 1
        self.max_nesting_level = max(self.max_nesting_level, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_With(self, node):
        """Visit a With node."""
        self.nesting_level += 1
        self.max_nesting_level = max(self.max_nesting_level, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
