"""
Optional dependency management for Saplings.

This module provides a centralized way to handle optional dependencies,
with lazy loading and helpful error messages.
"""

from __future__ import annotations

import logging
from typing import Dict

from saplings._internal.lazy_imports import LazyImporter as _LazyImporter

# Import the centralized lazy import system
from saplings._internal.lazy_imports import OptionalDependency as _OptionalDependency
from saplings._internal.lazy_imports import lazy_import

# Re-export for backward compatibility
OptionalDependency = _OptionalDependency
LazyImporter = _LazyImporter

logger = logging.getLogger(__name__)


# Classes are imported from lazy_imports.py above


# Define all optional dependencies using the centralized system
OPTIONAL_DEPENDENCIES: Dict[str, OptionalDependency] = {
    # ML/AI Dependencies
    "torch": OptionalDependency(
        import_name="torch", install_cmd="pip install saplings[gasa]", min_version="2.0.0"
    ),
    "transformers": OptionalDependency(
        import_name="transformers", install_cmd="pip install saplings[gasa]", min_version="4.30.0"
    ),
    "vllm": OptionalDependency(
        import_name="vllm", install_cmd="pip install saplings[vllm]", min_version="0.8.5"
    ),
    # Vector Search
    "faiss": OptionalDependency(
        import_name="faiss",
        install_cmd="pip install saplings[vector]",
        min_version="1.7.0",
        package_name="faiss-cpu",
    ),
    # Monitoring
    "langsmith": OptionalDependency(
        import_name="langsmith",
        install_cmd="pip install saplings[monitoring]",
        min_version="0.0.60",
    ),
    # Browser Tools
    "selenium": OptionalDependency(
        import_name="selenium", install_cmd="pip install saplings[browser]", min_version="4.10.0"
    ),
    "pillow": OptionalDependency(
        import_name="PIL",
        install_cmd="pip install saplings[browser]",
        min_version="9.0.0",
        package_name="pillow",
    ),
    # MCP
    "mcpadapt": OptionalDependency(
        import_name="mcpadapt", install_cmd="pip install saplings[mcp]", min_version="0.1.0"
    ),
    # Visualization
    "matplotlib": OptionalDependency(
        import_name="matplotlib", install_cmd="pip install saplings[viz]", min_version="3.5.0"
    ),
    "plotly": OptionalDependency(
        import_name="plotly", install_cmd="pip install saplings[viz]", min_version="5.5.0"
    ),
}


def get_optional_dependency(name: str) -> OptionalDependency:
    """
    Get an optional dependency by name.

    Args:
    ----
        name: The dependency name

    Returns:
    -------
        The OptionalDependency instance

    Raises:
    ------
        KeyError: If the dependency is not defined

    """
    if name not in OPTIONAL_DEPENDENCIES:
        raise KeyError(f"Unknown optional dependency: {name}")
    return OPTIONAL_DEPENDENCIES[name]


def check_feature_availability() -> Dict[str, bool]:
    """
    Check availability of all optional features.

    Returns
    -------
        Dictionary mapping feature names to availability

    """
    features = {
        "gasa": OPTIONAL_DEPENDENCIES["torch"].available
        and OPTIONAL_DEPENDENCIES["transformers"].available,
        "monitoring": OPTIONAL_DEPENDENCIES["langsmith"].available,
        "browser": OPTIONAL_DEPENDENCIES["selenium"].available,
        "mcp": OPTIONAL_DEPENDENCIES["mcpadapt"].available,
        "vector": OPTIONAL_DEPENDENCIES["faiss"].available,
        "viz": OPTIONAL_DEPENDENCIES["matplotlib"].available
        or OPTIONAL_DEPENDENCIES["plotly"].available,
        "vllm": OPTIONAL_DEPENDENCIES["vllm"].available,
    }
    return features


def require_feature(feature_name: str) -> None:
    """
    Require a feature to be available, raising ImportError if not.

    Args:
    ----
        feature_name: The feature name (e.g., 'gasa', 'monitoring')

    Raises:
    ------
        ImportError: If the feature is not available

    """
    feature_deps = {
        "gasa": ["torch", "transformers"],
        "monitoring": ["langsmith"],
        "browser": ["selenium"],
        "mcp": ["mcpadapt"],
        "vector": ["faiss"],
        "viz": ["matplotlib"],  # Just check one for viz
        "vllm": ["vllm"],
    }

    if feature_name not in feature_deps:
        raise ValueError(f"Unknown feature: {feature_name}")

    missing_deps = []
    for dep_name in feature_deps[feature_name]:
        dep = OPTIONAL_DEPENDENCIES[dep_name]
        if not dep.available:
            missing_deps.append(dep_name)

    if missing_deps:
        # Get install command from first missing dependency
        first_dep = OPTIONAL_DEPENDENCIES[missing_deps[0]]
        raise ImportError(
            f"{feature_name} features require additional dependencies: {', '.join(missing_deps)}\n"
            f"Install with: {first_dep.install_cmd}"
        )


# Create lazy importers for heavy dependencies using the centralized system
torch = lazy_import(
    "torch",
    "PyTorch is required for GASA features. Install with: pip install saplings[gasa]",
    OPTIONAL_DEPENDENCIES["torch"],
)

transformers = lazy_import(
    "transformers",
    "Transformers is required for GASA features. Install with: pip install saplings[gasa]",
    OPTIONAL_DEPENDENCIES["transformers"],
)

vllm = lazy_import(
    "vllm",
    "vLLM is required for vLLM adapter. Install with: pip install saplings[vllm]",
    OPTIONAL_DEPENDENCIES["vllm"],
)

faiss = lazy_import(
    "faiss",
    "FAISS is required for vector search. Install with: pip install saplings[vector]",
    OPTIONAL_DEPENDENCIES["faiss"],
)

langsmith = lazy_import(
    "langsmith",
    "LangSmith is required for monitoring. Install with: pip install saplings[monitoring]",
    OPTIONAL_DEPENDENCIES["langsmith"],
)

selenium = lazy_import(
    "selenium",
    "Selenium is required for browser tools. Install with: pip install saplings[browser]",
    OPTIONAL_DEPENDENCIES["selenium"],
)

mcpadapt = lazy_import(
    "mcpadapt",
    "MCP Adapt is required for MCP tools. Install with: pip install saplings[mcp]",
    OPTIONAL_DEPENDENCIES["mcpadapt"],
)
