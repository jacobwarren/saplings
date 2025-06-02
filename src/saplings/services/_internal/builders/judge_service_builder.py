from __future__ import annotations

"""
Builder for JudgeService.

This module provides a builder for the JudgeService to simplify initialization
and configuration. The builder pattern helps separate configuration from initialization
and provides a fluent interface for service creation.
"""

import logging
import os
from typing import Any, Dict

from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.model_adapter import LLM
from saplings.services._internal.providers.judge_service import JudgeService

logger = logging.getLogger(__name__)


class JudgeServiceBuilder(ServiceBuilder[JudgeService]):
    """
    Builder for JudgeService.

    This class provides a fluent interface for building JudgeService instances with
    proper configuration and dependency injection. It separates configuration from
    initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for JudgeService
    builder = JudgeServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_trace_manager(trace_manager) \
                    .with_initialize_judge(True) \
                    .with_rubric_path("path/to/rubrics") \
                    .with_lazy_loading(True) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the JudgeService builder."""
        super().__init__(JudgeService)
        self.require_dependency("model")
        self._initialize_judge = False
        self._rubric_path = None
        self._lazy_load_rubrics = True
        self._rubric_templates = {}
        self._scoring_models = {}

    def with_model(self, model: LLM) -> "JudgeServiceBuilder":
        """
        Set the model for the JudgeService.

        Args:
        ----
            model: The LLM model to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model", model)
        return self

    def with_trace_manager(self, trace_manager: Any) -> "JudgeServiceBuilder":
        """
        Set the trace manager for the JudgeService.

        Args:
        ----
            trace_manager: The trace manager to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_initialize_judge(self, initialize_judge: bool) -> "JudgeServiceBuilder":
        """
        Set whether to initialize the judge during build.

        Args:
        ----
            initialize_judge: Whether to initialize the judge during build

        Returns:
        -------
            The builder instance for method chaining

        """
        self._initialize_judge = initialize_judge
        return self

    def with_rubric_path(self, rubric_path: str) -> "JudgeServiceBuilder":
        """
        Set the path to rubric templates.

        Args:
        ----
            rubric_path: Path to rubric templates directory or file

        Returns:
        -------
            The builder instance for method chaining

        """
        if rubric_path and os.path.exists(rubric_path):
            self._rubric_path = rubric_path
            self.with_dependency("rubric_path", rubric_path)
        else:
            logger.warning(f"Rubric path does not exist: {rubric_path}")
        return self

    def with_lazy_loading(self, lazy_load: bool) -> "JudgeServiceBuilder":
        """
        Set whether to lazy load rubrics and scoring models.

        Args:
        ----
            lazy_load: Whether to lazy load rubrics and scoring models

        Returns:
        -------
            The builder instance for method chaining

        """
        self._lazy_load_rubrics = lazy_load
        self.with_dependency("lazy_load_rubrics", lazy_load)
        return self

    def with_rubric_template(self, name: str, template: Dict[str, Any]) -> "JudgeServiceBuilder":
        """
        Add a rubric template.

        Args:
        ----
            name: Name of the template
            template: Template definition

        Returns:
        -------
            The builder instance for method chaining

        """
        self._rubric_templates[name] = template
        return self

    def with_scoring_model(self, name: str, model: Any) -> "JudgeServiceBuilder":
        """
        Add a scoring model.

        Args:
        ----
            name: Name of the scoring model
            model: Scoring model instance

        Returns:
        -------
            The builder instance for method chaining

        """
        self._scoring_models[name] = model
        return self

    async def build_async(self) -> JudgeService:
        """
        Build the JudgeService instance with the configured dependencies and initialize the judge.

        This async version of build allows for initializing the judge during construction.

        Returns
        -------
            The initialized JudgeService instance

        Raises
        ------
            InitializationError: If service initialization fails

        """
        # Add rubric templates and scoring models to dependencies
        if self._rubric_templates:
            self.with_dependency("rubric_templates", self._rubric_templates)

        if self._scoring_models:
            self.with_dependency("scoring_models", self._scoring_models)

        # Build the service
        service = self.build()

        # Initialize the judge if requested
        if self._initialize_judge:
            await service.initialize_judge()

        return service
