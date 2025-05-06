from __future__ import annotations

"""
Saplings Runtimes Module.

This package contains runtime components for Saplings, such as CLI entry points,
web server integrations, FastAPI endpoints, and demo applications. These components
represent the highest level in the Saplings architecture and are allowed to depend
on any other Saplings module.

Runtimes are appropriate for:
- CLI tools that use Saplings functionality
- Web server integrations (FastAPI, Flask, etc.)
- Example applications that demonstrate full system capabilities
- UI integrations and visualization tools
- Streamlit or Gradio applications

Placement in this module indicates that the component:
1. Is an entry point or main executable
2. May have dependencies on all other Saplings modules
3. Is not intended to be imported by other Saplings components
"""


__all__ = []  # No exports by default
