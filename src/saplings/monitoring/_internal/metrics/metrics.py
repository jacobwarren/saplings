from __future__ import annotations

"""
Metrics module for monitoring components.

This module provides metrics functionality for the Saplings framework.
"""

import logging
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Type of metric."""

    COUNTER = "counter"
    """Counter metric that only increases."""

    GAUGE = "gauge"
    """Gauge metric that can increase or decrease."""

    HISTOGRAM = "histogram"
    """Histogram metric that tracks distribution of values."""


class Metric:
    """Base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Dict[str, str] = None,
    ) -> None:
        """
        Initialize a metric.

        Args:
        ----
            name: Name of the metric
            description: Description of the metric
            metric_type: Type of the metric
            labels: Labels for the metric

        """
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.labels = labels or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metric to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the metric

        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.metric_type,
            "labels": self.labels,
        }


class Counter(Metric):
    """Counter metric that only increases."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Dict[str, str] = None,
        initial_value: float = 0.0,
    ) -> None:
        """
        Initialize a counter metric.

        Args:
        ----
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            initial_value: Initial value of the counter

        """
        super().__init__(name, description, MetricType.COUNTER, labels)
        self.value = initial_value

    def increment(self, value: float = 1.0) -> None:
        """
        Increment the counter.

        Args:
        ----
            value: Value to increment by

        """
        if value < 0:
            logger.warning(f"Counter {self.name} cannot be decremented. Ignoring value {value}.")
            return

        self.value += value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the counter to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the counter

        """
        result = super().to_dict()
        result["value"] = self.value
        return result


class Gauge(Metric):
    """Gauge metric that can increase or decrease."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Dict[str, str] = None,
        initial_value: float = 0.0,
    ) -> None:
        """
        Initialize a gauge metric.

        Args:
        ----
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            initial_value: Initial value of the gauge

        """
        super().__init__(name, description, MetricType.GAUGE, labels)
        self.value = initial_value

    def set(self, value: float) -> None:
        """
        Set the gauge value.

        Args:
        ----
            value: Value to set

        """
        self.value = value

    def increment(self, value: float = 1.0) -> None:
        """
        Increment the gauge.

        Args:
        ----
            value: Value to increment by

        """
        self.value += value

    def decrement(self, value: float = 1.0) -> None:
        """
        Decrement the gauge.

        Args:
        ----
            value: Value to decrement by

        """
        self.value -= value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the gauge to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the gauge

        """
        result = super().to_dict()
        result["value"] = self.value
        return result


class Histogram(Metric):
    """Histogram metric that tracks distribution of values."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Dict[str, str] = None,
        buckets: List[float] = None,
    ) -> None:
        """
        Initialize a histogram metric.

        Args:
        ----
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            buckets: Histogram buckets

        """
        super().__init__(name, description, MetricType.HISTOGRAM, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.values: List[float] = []
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        """
        Observe a value.

        Args:
        ----
            value: Value to observe

        """
        self.values.append(value)
        self.sum += value
        self.count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the histogram to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the histogram

        """
        result = super().to_dict()
        result["buckets"] = self.buckets
        result["values"] = self.values
        result["sum"] = self.sum
        result["count"] = self.count
        return result
