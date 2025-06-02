"""
Type stubs for LangSmith library.
This is a minimal stub file to satisfy type checkers when LangSmith is not installed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class Run:
    """LangSmith Run object."""

    name: str
    id: str


class Client:
    """LangSmith Client."""

    def __init__(self, api_key: str) -> None: ...
    def list_projects(self) -> List[Any]: ...
    def create_project(self, name: str) -> Any: ...
    def create_run(
        self,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        child_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> str: ...
