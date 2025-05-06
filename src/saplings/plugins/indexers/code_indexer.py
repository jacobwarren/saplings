from __future__ import annotations

"""
CodeIndexer plugin for Saplings.

This module provides an indexer specialized for code repositories,
extracting entities and relationships from code files.
"""


import ast
import logging
import os
import re
from typing import TYPE_CHECKING, Any

from saplings.core.plugin import IndexerPlugin, PluginType
from saplings.memory.indexer import Entity, Indexer, Relationship

if TYPE_CHECKING:
    from saplings.memory.config import MemoryConfig
    from saplings.memory.document import Document

logger = logging.getLogger(__name__)


def get_source_from_metadata(metadata: Any) -> str | None:
    """
    Extract source from document metadata.

    Args:
    ----
        metadata: Document metadata (DocumentMetadata, dict, or None)

    Returns:
    -------
        str | None: Source path or None if not available

    """
    if metadata is None:
        return None

    if hasattr(metadata, "source"):
        return metadata.source
    if isinstance(metadata, dict) and "source" in metadata:
        return metadata["source"]

    return None


class CodeIndexer(IndexerPlugin, Indexer):
    """
    Indexer specialized for code repositories.

    This indexer extracts entities and relationships from code files,
    such as classes, functions, imports, and dependencies.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the code indexer.

        Args:
        ----
            config: Memory configuration

        """
        super().__init__(config)
        self.supported_extensions = {
            ".py": self._index_python_file,
            ".js": self._index_javascript_file,
            ".ts": self._index_typescript_file,
            ".java": self._index_java_file,
            ".cpp": self._index_cpp_file,
            ".c": self._index_c_file,
            ".h": self._index_header_file,
        }

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "code_indexer"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "Indexer specialized for code repositories"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.INDEXER

    def extract_entities(self, document: Document) -> list[Entity]:
        """
        Extract entities from a document.

        Args:
        ----
            document: Document to extract entities from

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        # Check if this is a code file
        file_path = get_source_from_metadata(document.metadata)
        if not file_path or not isinstance(file_path, str):
            return []

        # Get file extension
        _, ext = os.path.splitext(file_path)
        if ext not in self.supported_extensions:
            return []

        # Call the appropriate indexing function
        indexer_func = self.supported_extensions[ext]
        return indexer_func(document)

    def extract_relationships(
        self, document: Document, entities: list[Entity]
    ) -> list[Relationship]:
        """
        Extract relationships between entities in a document.

        Args:
        ----
            document: Document to extract relationships from
            entities: Entities extracted from the document

        Returns:
        -------
            List[Relationship]: Extracted relationships

        """
        # Check if this is a code file
        file_path = get_source_from_metadata(document.metadata)
        if not file_path or not isinstance(file_path, str):
            return []

        # Get file extension
        _, ext = os.path.splitext(file_path)
        if ext not in self.supported_extensions:
            return []

        # Create a mapping of entity names to entity IDs
        entity_map = {
            entity.name: f"entity:{entity.entity_type}:{entity.name}" for entity in entities
        }

        relationships = []

        # Add relationships based on entity types
        for entity in entities:
            if entity.entity_type == "class":
                # Check for inheritance relationships
                parent_classes = entity.metadata.get("parent_classes", [])
                for parent in parent_classes:
                    if parent in entity_map:
                        relationship = Relationship(
                            source_id=f"entity:class:{entity.name}",
                            target_id=f"entity:class:{parent}",
                            relationship_type="inherits_from",
                            metadata={"confidence": 1.0},
                        )
                        relationships.append(relationship)

            elif entity.entity_type == "function":
                # Check for function calls
                called_functions = entity.metadata.get("calls", [])
                for called_func in called_functions:
                    if called_func in entity_map:
                        relationship = Relationship(
                            source_id=f"entity:function:{entity.name}",
                            target_id=f"entity:function:{called_func}",
                            relationship_type="calls",
                            metadata={"confidence": 1.0},
                        )
                        relationships.append(relationship)

            elif entity.entity_type == "module":
                # Check for imports
                imported_modules = entity.metadata.get("imports", [])
                for imported_module in imported_modules:
                    if imported_module in entity_map:
                        relationship = Relationship(
                            source_id=f"entity:module:{entity.name}",
                            target_id=f"entity:module:{imported_module}",
                            relationship_type="imports",
                            metadata={"confidence": 1.0},
                        )
                        relationships.append(relationship)

        return relationships

    def _index_python_file(self, document: Document) -> list[Entity]:
        """
        Index a Python file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        try:
            # Parse the Python code
            tree = ast.parse(document.content)

            # Extract module name from file path
            file_path = get_source_from_metadata(document.metadata)
            if file_path and isinstance(file_path, str):
                module_name = os.path.basename(file_path).replace(".py", "")

                # Create module entity
                module_entity = Entity(
                    name=module_name,
                    entity_type="module",
                    metadata={
                        "source_document": document.id,
                        "file_path": file_path,
                        "imports": [],
                    },
                )
                entities.append(module_entity)

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)

            if module_entity:
                module_entity.metadata["imports"] = imports

            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Get parent classes
                    parent_classes = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            parent_classes.append(base.id)

                    # Create class entity
                    class_entity = Entity(
                        name=node.name,
                        entity_type="class",
                        metadata={
                            "source_document": document.id,
                            "parent_classes": parent_classes,
                            "module": module_name,
                        },
                    )
                    entities.append(class_entity)

            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this is a method
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            for child in parent.body:
                                if isinstance(child, ast.FunctionDef) and child.name == node.name:
                                    is_method = True
                                    break

                    # Create function entity
                    function_entity = Entity(
                        name=node.name,
                        entity_type="function" if not is_method else "method",
                        metadata={
                            "source_document": document.id,
                            "module": module_name,
                            "is_method": is_method,
                        },
                    )
                    entities.append(function_entity)

        except Exception as e:
            logger.warning(f"Error indexing Python file: {e}")

        return entities

    def _index_javascript_file(self, document: Document) -> list[Entity]:
        """
        Index a JavaScript file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Extract file name as module name
        file_path = get_source_from_metadata(document.metadata)
        if file_path and isinstance(file_path, str):
            module_name = os.path.basename(file_path).replace(".js", "")

            # Create module entity
            module_entity = Entity(
                name=module_name,
                entity_type="module",
                metadata={
                    "source_document": document.id,
                    "file_path": file_path,
                    "language": "javascript",
                },
            )
            entities.append(module_entity)

        # Use regex to extract classes and functions
        # Extract classes (ES6 class syntax)
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, document.content):
            class_name = match.group(1)
            parent_class = match.group(2)

            class_entity = Entity(
                name=class_name,
                entity_type="class",
                metadata={
                    "source_document": document.id,
                    "parent_classes": [parent_class] if parent_class else [],
                    "module": module_name,
                    "language": "javascript",
                },
            )
            entities.append(class_entity)

        # Extract functions
        function_pattern = r"function\s+(\w+)\s*\("
        for match in re.finditer(function_pattern, document.content):
            function_name = match.group(1)

            function_entity = Entity(
                name=function_name,
                entity_type="function",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "javascript",
                },
            )
            entities.append(function_entity)

        # Extract arrow functions assigned to variables
        arrow_function_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=]*)\s*=>"
        for match in re.finditer(arrow_function_pattern, document.content):
            function_name = match.group(1)

            function_entity = Entity(
                name=function_name,
                entity_type="function",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "javascript",
                    "is_arrow_function": True,
                },
            )
            entities.append(function_entity)

        return entities

    def _index_typescript_file(self, document: Document) -> list[Entity]:
        """
        Index a TypeScript file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        # TypeScript indexing is similar to JavaScript but with type annotations
        entities = self._index_javascript_file(document)

        # Update language metadata
        for entity in entities:
            if "language" in entity.metadata:
                entity.metadata["language"] = "typescript"

        # Extract interfaces
        interface_pattern = r"interface\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(interface_pattern, document.content):
            interface_name = match.group(1)
            parent_interface = match.group(2)

            interface_entity = Entity(
                name=interface_name,
                entity_type="interface",
                metadata={
                    "source_document": document.id,
                    "parent_interfaces": [parent_interface] if parent_interface else [],
                    "module": entities[0].name if entities else None,
                    "language": "typescript",
                },
            )
            entities.append(interface_entity)

        return entities

    def _index_java_file(self, document: Document) -> list[Entity]:
        """
        Index a Java file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Extract package and class name
        package_pattern = r"package\s+([\w.]+);"
        package_match = re.search(package_pattern, document.content)
        package_name = package_match.group(1) if package_match else "default"

        # Extract class name from file path
        file_path = get_source_from_metadata(document.metadata)
        if file_path and isinstance(file_path, str):
            class_name = os.path.basename(file_path).replace(".java", "")

            # Create package entity
            package_entity = Entity(
                name=package_name,
                entity_type="package",
                metadata={
                    "source_document": document.id,
                    "file_path": file_path,
                    "language": "java",
                },
            )
            entities.append(package_entity)

            # Extract class definition
            class_pattern = r"(?:public|private|protected)?\s+(?:abstract|final)?\s+class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?"
            class_match = re.search(class_pattern, document.content)

            if class_match:
                parent_class = class_match.group(2)
                implemented_interfaces = class_match.group(3)

                interfaces = []
                if implemented_interfaces:
                    interfaces = [i.strip() for i in implemented_interfaces.split(",")]

                class_entity = Entity(
                    name=class_name,
                    entity_type="class",
                    metadata={
                        "source_document": document.id,
                        "parent_classes": [parent_class] if parent_class else [],
                        "implements": interfaces,
                        "package": package_name,
                        "language": "java",
                    },
                )
                entities.append(class_entity)

            # Extract methods
            method_pattern = r"(?:public|private|protected)?\s+(?:static|final|abstract)?\s+(?:[\w<>[\],\s]+)\s+(\w+)\s*\([^)]*\)"
            for match in re.finditer(method_pattern, document.content):
                method_name = match.group(1)

                # Skip constructor (same name as class)
                if method_name == class_name:
                    continue

                method_entity = Entity(
                    name=method_name,
                    entity_type="method",
                    metadata={
                        "source_document": document.id,
                        "class": class_name,
                        "package": package_name,
                        "language": "java",
                    },
                )
                entities.append(method_entity)

        return entities

    def _index_cpp_file(self, document: Document) -> list[Entity]:
        """
        Index a C++ file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Extract file name as module name
        file_path = get_source_from_metadata(document.metadata)
        if file_path and isinstance(file_path, str):
            module_name = os.path.basename(file_path).replace(".cpp", "")

            # Create module entity
            module_entity = Entity(
                name=module_name,
                entity_type="module",
                metadata={
                    "source_document": document.id,
                    "file_path": file_path,
                    "language": "cpp",
                },
            )
            entities.append(module_entity)

        # Extract classes
        class_pattern = r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)?\s*(\w+))?"
        for match in re.finditer(class_pattern, document.content):
            class_name = match.group(1)
            parent_class = match.group(2)

            class_entity = Entity(
                name=class_name,
                entity_type="class",
                metadata={
                    "source_document": document.id,
                    "parent_classes": [parent_class] if parent_class else [],
                    "module": module_name,
                    "language": "cpp",
                },
            )
            entities.append(class_entity)

        # Extract functions
        function_pattern = r"(?:[\w:*&<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(function_pattern, document.content):
            function_name = match.group(1)

            # Skip constructors and destructors
            if function_name.startswith("~") or any(
                function_name == class_entity.name
                for class_entity in entities
                if class_entity.entity_type == "class"
            ):
                continue

            function_entity = Entity(
                name=function_name,
                entity_type="function",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "cpp",
                },
            )
            entities.append(function_entity)

        return entities

    def _index_c_file(self, document: Document) -> list[Entity]:
        """
        Index a C file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Extract file name as module name
        file_path = get_source_from_metadata(document.metadata)
        if file_path and isinstance(file_path, str):
            module_name = os.path.basename(file_path).replace(".c", "")

            # Create module entity
            module_entity = Entity(
                name=module_name,
                entity_type="module",
                metadata={
                    "source_document": document.id,
                    "file_path": file_path,
                    "language": "c",
                },
            )
            entities.append(module_entity)

        # Extract functions
        function_pattern = r"(?:[\w*&\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*\{"
        for match in re.finditer(function_pattern, document.content):
            function_name = match.group(1)

            function_entity = Entity(
                name=function_name,
                entity_type="function",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "c",
                },
            )
            entities.append(function_entity)

        # Extract structs
        struct_pattern = r"struct\s+(\w+)\s*\{"
        for match in re.finditer(struct_pattern, document.content):
            struct_name = match.group(1)

            struct_entity = Entity(
                name=struct_name,
                entity_type="struct",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "c",
                },
            )
            entities.append(struct_entity)

        return entities

    def _index_header_file(self, document: Document) -> list[Entity]:
        """
        Index a C/C++ header file.

        Args:
        ----
            document: Document to index

        Returns:
        -------
            List[Entity]: Extracted entities

        """
        entities = []

        # Extract file name as module name
        file_path = get_source_from_metadata(document.metadata)
        if file_path and isinstance(file_path, str):
            module_name = os.path.basename(file_path).replace(".h", "")

            # Create module entity
            module_entity = Entity(
                name=module_name,
                entity_type="header",
                metadata={
                    "source_document": document.id,
                    "file_path": file_path,
                    "language": "c/cpp",
                },
            )
            entities.append(module_entity)

        # Extract function declarations
        function_pattern = r"(?:[\w*&<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*;"
        for match in re.finditer(function_pattern, document.content):
            function_name = match.group(1)

            function_entity = Entity(
                name=function_name,
                entity_type="function_declaration",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "c/cpp",
                },
            )
            entities.append(function_entity)

        # Extract classes (C++)
        class_pattern = r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)?\s*(\w+))?"
        for match in re.finditer(class_pattern, document.content):
            class_name = match.group(1)
            parent_class = match.group(2)

            class_entity = Entity(
                name=class_name,
                entity_type="class",
                metadata={
                    "source_document": document.id,
                    "parent_classes": [parent_class] if parent_class else [],
                    "module": module_name,
                    "language": "cpp",
                },
            )
            entities.append(class_entity)

        # Extract structs (C)
        struct_pattern = r"struct\s+(\w+)\s*\{"
        for match in re.finditer(struct_pattern, document.content):
            struct_name = match.group(1)

            struct_entity = Entity(
                name=struct_name,
                entity_type="struct",
                metadata={
                    "source_document": document.id,
                    "module": module_name,
                    "language": "c",
                },
            )
            entities.append(struct_entity)

        return entities
