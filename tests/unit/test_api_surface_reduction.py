"""
Test API surface reduction for publication readiness.

This module tests Task 7.1: Analyze and reduce public API surface.
"""

from __future__ import annotations

from typing import List

import pytest


class TestAPISurfaceReduction:
    """Test API surface reduction and categorization."""

    def test_analyze_current_api_surface(self):
        """Test that we can analyze the current API surface."""
        import saplings

        # Get all exported items
        all_items = saplings.__all__

        # Verify we have the expected number of items
        assert len(all_items) == 242, f"Expected 242 items, got {len(all_items)}"

        # Categorize items according to standardization document
        core_items = self._get_core_api_items()
        advanced_items = self._get_advanced_api_items()
        internal_items = self._get_internal_api_items()

        # Verify categorization covers all items
        categorized_items = set(core_items) | set(advanced_items) | set(internal_items)
        all_items_set = set(all_items)

        uncategorized = all_items_set - categorized_items
        if uncategorized:
            pytest.fail(f"Uncategorized items found: {sorted(uncategorized)}")

        # Verify no overlap between categories
        core_set = set(core_items)
        advanced_set = set(advanced_items)
        internal_set = set(internal_items)

        core_advanced_overlap = core_set & advanced_set
        core_internal_overlap = core_set & internal_set
        advanced_internal_overlap = advanced_set & internal_set

        assert not core_advanced_overlap, f"Core/Advanced overlap: {core_advanced_overlap}"
        assert not core_internal_overlap, f"Core/Internal overlap: {core_internal_overlap}"
        assert (
            not advanced_internal_overlap
        ), f"Advanced/Internal overlap: {advanced_internal_overlap}"

    def test_core_api_size_is_reasonable(self):
        """Test that core API is reasonably sized (30-50 items)."""
        core_items = self._get_core_api_items()

        assert 30 <= len(core_items) <= 50, (
            f"Core API should have 30-50 items, got {len(core_items)}. "
            f"Items: {sorted(core_items)}"
        )

    def test_core_api_contains_essentials(self):
        """Test that core API contains all essential components."""
        core_items = self._get_core_api_items()

        # Essential classes that must be in core API
        essential_items = {
            "Agent",
            "AgentBuilder",
            "AgentConfig",
            "Tool",
            "PythonInterpreterTool",
            "FinalAnswerTool",
            "MemoryStore",
            "Document",
            "DocumentMetadata",
            "LLM",
            "LLMBuilder",
            "LLMResponse",
            "ExecutionService",
            "MemoryManager",
            "ToolService",
            "SaplingsError",
            "ModelError",
            "ProviderError",
            "count_tokens",
            "run_sync",
            "async_run_sync",
        }

        missing_essentials = essential_items - set(core_items)
        assert (
            not missing_essentials
        ), f"Missing essential items from core API: {missing_essentials}"

    def test_advanced_api_contains_specialized_features(self):
        """Test that advanced API contains specialized features."""
        advanced_items = self._get_advanced_api_items()

        # Specialized features that should be in advanced API
        specialized_features = {
            "GASAService",
            "SelfHealingService",
            "OrchestrationService",
            "TraceManager",
            "BlameGraph",
            "MaskVisualizer",
            "Sanitizer",
            "RedactingFilter",
            "SecureHotLoader",
            "ToolFactory",
            "DockerSandbox",
            "E2BSandbox",
        }

        missing_specialized = specialized_features - set(advanced_items)
        assert (
            not missing_specialized
        ), f"Missing specialized features from advanced API: {missing_specialized}"

    def test_internal_items_not_in_public_api(self):
        """Test that internal items should not be in public API."""
        internal_items = self._get_internal_api_items()

        # These items should be internal-only
        for item in internal_items:
            # Check if item starts with I (interface) or has internal-like naming
            if item.startswith("I") and item[1:2].isupper():
                # This is likely a service interface - should be internal
                continue
            if "Config" in item and item not in self._get_core_api_items():
                # Complex configuration items should be internal
                continue

    def _get_core_api_items(self) -> List[str]:
        """Get items that should be in core API."""
        return [
            # Essential classes
            "Agent",
            "AgentBuilder",
            "AgentConfig",
            # Basic tools
            "Tool",
            "PythonInterpreterTool",
            "FinalAnswerTool",
            # Memory basics
            "MemoryStore",
            "Document",
            "DocumentMetadata",
            # Model basics
            "LLM",
            "LLMBuilder",
            "LLMResponse",
            # Basic services
            "ExecutionService",
            "MemoryManager",
            "ToolService",
            # Core exceptions
            "SaplingsError",
            "ModelError",
            "ProviderError",
            # Utilities
            "count_tokens",
            "run_sync",
            "async_run_sync",
            # Basic tool functions
            "register_tool",
            "tool",
            "validate_tool",
            # Basic memory
            "MemoryStoreBuilder",
            "MemoryConfig",
            # Basic indexing
            "Indexer",
            "SimpleIndexer",
            "get_indexer",
            # Basic vector storage
            "VectorStore",
            "InMemoryVectorStore",
            "get_vector_store",
            # Basic model adapters
            "OpenAIAdapter",
            "AnthropicAdapter",
            # Basic service builders
            "ExecutionServiceBuilder",
            "MemoryManagerBuilder",
            "ToolServiceBuilder",
            # Version
            "__version__",
            # Basic configuration
            "Config",
            "ConfigValue",
            "ConfigurationError",
            # Basic tool collection
            "ToolCollection",
            "ToolRegistry",
            # Basic search tools
            "GoogleSearchTool",
            "DuckDuckGoSearchTool",
            "WikipediaSearchTool",
            # Basic model metadata
            "ModelMetadata",
            "ModelCapability",
            "ModelRole",
        ]

    def _get_advanced_api_items(self) -> List[str]:
        """Get items that should be in advanced API."""
        return [
            # Specialized services
            "GASAService",
            "GASAServiceBuilder",
            "SelfHealingService",
            "SelfHealingServiceBuilder",
            "OrchestrationService",
            "OrchestrationServiceBuilder",
            "JudgeService",
            "JudgeServiceBuilder",
            "PlannerService",
            "PlannerServiceBuilder",
            "RetrievalService",
            "RetrievalServiceBuilder",
            "ValidatorService",
            "ValidatorServiceBuilder",
            "ModalityService",
            "ModalityServiceBuilder",
            # Advanced tools
            "ClickTool",
            "ClosePopupsTool",
            "GetPageTextTool",
            "GoBackTool",
            "GoToTool",
            "ScrollTool",
            "SearchTextTool",
            "WaitTool",
            "close_browser",
            "get_browser_tools",
            "initialize_browser",
            "save_screenshot",
            "MCPClient",
            "MCPTool",
            "SpeechToTextTool",
            "UserInputTool",
            "VisitWebpageTool",
            "is_browser_tools_available",
            "is_mcp_available",
            # Advanced memory
            "DependencyGraph",
            "DependencyGraphBuilder",
            "DocumentNode",
            "IndexerRegistry",
            "FaissVectorStore",
            "GraphExpander",
            # Monitoring
            "TraceManager",
            "TraceViewer",
            "BlameGraph",
            "BlameNode",
            "BlameEdge",
            "MonitoringConfig",
            # Security
            "Sanitizer",
            "RedactingFilter",
            "install_global_filter",
            "install_import_hook",
            "redact",
            "sanitize",
            # Tool factory
            "ToolFactory",
            "ToolFactoryConfig",
            "SecureHotLoader",
            "SecureHotLoaderConfig",
            "create_secure_hot_loader",
            "CodeSigner",
            "SignatureVerifier",
            "DockerSandbox",
            "E2BSandbox",
            "Sandbox",
            "SandboxType",
            "SecurityLevel",
            "SigningLevel",
            "ToolSpecification",
            "ToolTemplate",
            "ToolValidator",
            # Advanced configuration
            "GASAConfig",
            "GASAConfigBuilder",
            "ModelServiceBuilder",
            # GASA (Graph-Aware Sparse Attention)
            "BlockDiagonalPacker",
            "FallbackStrategy",
            "GraphDistanceCalculator",
            "MaskFormat",
            "MaskStrategy",
            "MaskType",
            "MaskVisualizer",
            "StandardMaskBuilder",
            "TokenMapper",
            "block_pack",
            # Self-Healing
            "Adapter",
            "AdapterManager",
            "AdapterMetadata",
            "AdapterPriority",
            "LoRaConfig",
            "LoRaTrainer",
            "Patch",
            "PatchGenerator",
            "PatchResult",
            "PatchStatus",
            "RetryStrategy",
            "SelfHealingConfig",
            "SuccessPairCollector",
            "TrainingMetrics",
            # Validators
            "ExecutionValidator",
            "KeywordValidator",
            "LengthValidator",
            "RuntimeValidator",
            "StaticValidator",
            "ValidationResult",
            "ValidationStatus",
            "ValidationStrategy",
            "Validator",
            "ValidatorConfig",
            "ValidatorRegistry",
            "ValidatorType",
            "get_validator_registry",
            # Retrieval
            "CascadeRetriever",
            "EmbeddingRetriever",
            "EntropyCalculator",
            "RetrievalConfig",
            "TFIDFRetriever",
            # Judge
            "CritiqueFormat",
            "JudgeAgent",
            "JudgeConfig",
            "JudgeResult",
            "Rubric",
            "RubricItem",
            "ScoringDimension",
            # Modality
            "AudioHandler",
            "ImageHandler",
            "ModalityHandler",
            "ModalityConfig",
            "ModalityType",
            "TextHandler",
            "VideoHandler",
            "get_handler_for_modality",
            # Orchestration
            "AgentNode",
            "CommunicationChannel",
            "GraphRunner",
            "GraphRunnerConfig",
            "NegotiationStrategy",
            "OrchestrationConfig",
            # Tokenizers
            "SimpleTokenizer",
            "TokenizerFactory",
            "SHADOW_MODEL_AVAILABLE",
            # Registry and Service Locator
            "PluginRegistry",
            "PluginType",
            "RegistryContext",
            "ServiceLocator",
            # Advanced model adapters
            "HuggingFaceAdapter",
            "VLLMAdapter",
            # Advanced utilities
            "get_model_sync",
            "get_tokens_remaining",
            "split_text_by_tokens",
            "truncate_text_tokens",
            # Advanced tool functions
            "get_all_default_tools",
            "get_default_tool",
            "get_registered_tools",
            "validate_tool_attributes",
            "validate_tool_parameters",
            # Resource management
            "ResourceExhaustedError",
        ]

    def _get_internal_api_items(self) -> List[str]:
        """Get items that should be internal-only."""
        return [
            # Service interfaces (for typing only)
            "IExecutionService",
            "IGasaService",
            "IJudgeService",
            "IMemoryManager",
            "IModalityService",
            "IModelCachingService",
            "IModelInitializationService",
            "IMonitoringService",
            "IOrchestrationService",
            "IPlannerService",
            "IRetrievalService",
            "ISelfHealingService",
            "IToolService",
            "IValidatorService",
            # Complex configuration objects
            "GasaConfig",
            "ModelCachingConfig",
            "ModelInitializationConfig",
            "PlannerConfig",
            "ToolConfig",
            "ValidationConfig",
            # Context objects
            "ExecutionContext",
            "ExecutionResult",
            "ModelContext",
            "GenerationContext",
            "MonitoringEvent",
            "OrchestrationResult",
            "PlanningResult",
            "RetrievalResult",
            "SelfHealingResult",
            "ToolResult",
            "ValidationContext",
            # Container and DI (should use simplified interface)
            "Container",
            "container",
            "reset_container",
            "configure_container",
            "reset_container_config",
            # AgentFacade (beta components)
            "AgentFacade",
            "AgentFacadeBuilder",
        ]
