from __future__ import annotations

"""
LangSmith integration API module for Saplings.

This module provides the public API for LangSmith integration.
"""

from saplings.api.stability import beta

# Import LangSmith exporter if available
try:
    from saplings.monitoring._internal.langsmith import (
        LangSmithExporter as _InternalLangSmithExporter,
    )

    LANGSMITH_AVAILABLE = True

    @beta
    class LangSmithExporter(_InternalLangSmithExporter):
        """
        Exporter for LangSmith.

        This class provides functionality for exporting traces and spans to
        LangSmith for visualization and analysis.
        """


except ImportError:
    LANGSMITH_AVAILABLE = False

    @beta
    class LangSmithExporter:
        """
        Placeholder for LangSmith exporter when LangSmith is not installed.

        This class provides a placeholder that raises an error when instantiated,
        indicating that LangSmith is not installed.
        """

        def __init__(self, **_):
            raise ImportError(
                "LangSmith not installed. Install LangSmith with: pip install saplings[langsmith]"
            )


__all__ = [
    "LangSmithExporter",
    "LANGSMITH_AVAILABLE",
]
