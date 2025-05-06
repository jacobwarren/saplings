"""
Saplings Service Layer.
======================

This package contains cohesive, injectable sub-services that encapsulate a
single concern of the agent.  Using these services allows the high-level
:class:`saplings.agent.Agent` (or future *AgentFacade*) to be composed rather
than bloated, dramatically improving testability and separation of concerns.

Sub-services should expose a minimal, well-typed public API and avoid any
cross-cutting logic that does not belong to their domain.
"""
