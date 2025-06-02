from __future__ import annotations

"""
Configuration service for Saplings.

This module provides a central configuration service that abstracts environment
variable access and provides a more reproducible approach to configuration.
"""

from saplings.core._internal.config_service import config_service

__all__ = ["config_service"]
