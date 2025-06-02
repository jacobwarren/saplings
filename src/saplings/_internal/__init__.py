from __future__ import annotations

"""
Internal module for Saplings.

This module contains internal implementation details that are not part of the public API.
These components should not be used directly by application code.

Import Guidelines for Internal Modules:
---------------------------------------

1. Public API Imports:
   - Internal modules should import from the public API when accessing other components
   - Example: `from saplings.api.memory import MemoryStore`

2. Internal Implementation Imports:
   - When a component needs access to internal details of its own component, use relative imports
   - Example: `from ._internal_module import InternalClass`

3. Cross-Component Internal Imports:
   - Avoid importing from other components' internal modules when possible
   - If necessary, add a comment explaining why the internal import is required
   - Example: `# Import from internal module because X is not exposed in the public API`

4. Type Checking Imports:
   - Use TYPE_CHECKING guard for imports only needed for type checking
   - Example:
     ```python
     if TYPE_CHECKING:
         from saplings.api.memory import MemoryStore
     ```

5. Circular Dependency Prevention:
   - If an import would create a circular dependency, use the public API
   - If that's not possible, use TYPE_CHECKING or import inside functions

Following these guidelines helps prevent circular dependencies and maintains
a clear separation between public and internal APIs.
"""

# This module intentionally does not export any symbols
__all__ = []
