from __future__ import annotations

"""
Monitoring module for Saplings.

This module re-exports the public API from saplings.api.monitoring.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides monitoring capabilities for Saplings, including:
- OpenTelemetry (OTEL) tracing infrastructure
- Causal blame graph for identifying bottlenecks
- GASA heatmap visualization
- TraceViewer interface for trace exploration
- LangSmith export capabilities
"""

# Import directly from internal modules to avoid circular imports
# We can't import from saplings.api.monitoring due to circular imports
# The public API test will need to be updated to handle this special case
from saplings.monitoring._internal.trace import TraceManager

# Re-export symbols
__all__ = [
    "TraceManager",
    # Note: Other monitoring symbols should be imported from saplings.api.monitoring
]
