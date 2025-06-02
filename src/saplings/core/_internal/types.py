from __future__ import annotations

"""
Core type definitions for Saplings.

This module provides standard data classes for complex method signatures
to ensure consistency across the codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Forward references
Document = Any  # From saplings.memory
LLM = Any  # From saplings.core.model_adapter
LLMResponse = Any  # From saplings.core.model_adapter


@dataclass
class ExecutionContext:
    """Standard context for execution operations."""

    prompt: str
    documents: Optional[List[Document]] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    trace_id: Optional[str] = None
    timeout: Optional[float] = None
    verification_strategy: Optional[str] = None


@dataclass
class ValidationContext:
    """Standard context for validation operations."""

    input_data: Dict[str, Any]
    output_data: Any
    validation_type: str = "general"
    trace_id: Optional[str] = None
    timeout: Optional[float] = None


@dataclass
class ValidationResult:
    """Standard result for validation operations."""

    is_valid: bool
    score: float
    feedback: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ModelContext:
    """Standard context for model operations."""

    provider: str
    model_name: str
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    retry_config: Optional[Dict[str, Any]] = None
    circuit_breaker_config: Optional[Dict[str, Any]] = None
    cache_enabled: bool = True
    cache_namespace: str = "model"
    cache_ttl: Optional[int] = 3600
    cache_provider: str = "memory"
    cache_strategy: Optional[str] = None


@dataclass
class GenerationContext:
    """Standard context for generation operations."""

    prompt: Union[str, List[Dict[str, Any]]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    use_cache: bool = False
    timeout: Optional[float] = None
    additional_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of an execution."""

    text: str
    provider: str
    model_name: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]
    verified: bool = False
    verification_score: Optional[float] = None
    verification_feedback: Optional[str] = None
    draft_latency_ms: Optional[int] = None
    final_latency_ms: Optional[int] = None
    total_latency_ms: Optional[int] = None
    refinement_attempts: int = 0
    response: Optional[LLMResponse] = None
