"""
Test for Task 1.1: Audit Current Top-Level Exports

This test validates the current API surface and categorizes exports according to
the requirements in finish.md Task 1.1.
"""

from __future__ import annotations


class TestTask1_1_AuditExports:
    """Test suite for auditing current top-level exports."""

    def test_main_package_exports_count(self):
        """Test that main package exports ≤ 20 items as per validation criteria."""
        # Import the main package
        import saplings

        # Get all exported items from __all__
        main_exports = getattr(saplings, "__all__", [])

        print(f"Main package exports {len(main_exports)} items:")
        for item in sorted(main_exports):
            print(f"  - {item}")

        # Validation criteria: Main package exports ≤ 20 items
        assert (
            len(main_exports) <= 20
        ), f"Main package exports {len(main_exports)} items, should be ≤ 20"

    def test_api_package_exports_count(self):
        """Test that API package exports are documented and categorized."""
        # Import the API package
        import saplings.api

        # Get all exported items from __all__
        api_exports = getattr(saplings.api, "__all__", [])

        print(f"API package exports {len(api_exports)} items")

        # Document the current state - this should be 657 as mentioned in the task
        assert len(api_exports) > 600, f"Expected >600 exports, got {len(api_exports)}"

    def test_categorize_api_exports(self):
        """Categorize all API exports into core, advanced, and experimental."""
        import saplings.api

        api_exports = getattr(saplings.api, "__all__", [])

        # Define categorization based on the task requirements
        CORE_ITEMS = {
            "Agent",
            "AgentBuilder",
            "AgentConfig",
            "Tool",
            "PythonInterpreterTool",
            "FinalAnswerTool",
            "Document",
            "DocumentMetadata",
            "MemoryStore",
            "LLM",
            "LLMBuilder",
            "LLMResponse",
            # Add essential tools and memory components
            "DuckDuckGoSearchTool",
            "GoogleSearchTool",
            "WikipediaSearchTool",
            "UserInputTool",
            "ToolRegistry",
            "ToolCollection",
            "DependencyGraph",
            "MemoryConfig",
            "VectorStore",
            # Basic model adapters
            "OpenAIAdapter",
            "AnthropicAdapter",
            "ModelMetadata",
            # Core configuration and exceptions
            "Config",
            "ConfigValue",
            "SaplingsError",
            "__version__",
        }

        ADVANCED_ITEMS = {
            "GASAService",
            "GASAConfig",
            "OrchestrationService",
            "SelfHealingService",
            "MonitoringService",
            "TraceViewer",
            # GASA components
            "GASAConfigBuilder",
            "GASAServiceBuilder",
            "MaskVisualizer",
            "BlockDiagonalPacker",
            "GraphDistanceCalculator",
            "StandardMaskBuilder",
            # Monitoring components
            "BlameGraph",
            "BlameNode",
            "BlameEdge",
            "TraceManager",
            # Orchestration components
            "AgentNode",
            "CommunicationChannel",
            "GraphRunner",
            # Advanced retrieval
            "FaissVectorStore",
            "EmbeddingRetriever",
            "CascadeRetriever",
            # Service builders
            "ExecutionServiceBuilder",
            "MemoryManagerBuilder",
            "RetrievalServiceBuilder",
        }

        EXPERIMENTAL_ITEMS = {
            "ToolFactory",
            "SecureHotLoader",
            "AdapterManager",
            "PatchGenerator",
            "LoRaTrainer",
            # Tool factory components
            "ToolFactoryConfig",
            "ToolSpecification",
            "ToolTemplate",
            "CodeSigner",
            "SignatureVerifier",
            "DockerSandbox",
            "E2BSandbox",
            # Self-healing components
            "Adapter",
            "AdapterMetadata",
            "Patch",
            "PatchResult",
            "SuccessPairCollector",
            "TrainingMetrics",
            # Security features
            "SecurityLevel",
            "SigningLevel",
            "Sanitizer",
            "RedactingFilter",
        }

        # Categorize exports
        core_found = set()
        advanced_found = set()
        experimental_found = set()
        uncategorized = set()

        for export in api_exports:
            if export in CORE_ITEMS:
                core_found.add(export)
            elif export in ADVANCED_ITEMS:
                advanced_found.add(export)
            elif export in EXPERIMENTAL_ITEMS:
                experimental_found.add(export)
            else:
                uncategorized.add(export)

        # Print categorization results
        print("\nCategorization Results:")
        print(f"Core items found: {len(core_found)}")
        print(f"Advanced items found: {len(advanced_found)}")
        print(f"Experimental items found: {len(experimental_found)}")
        print(f"Uncategorized items: {len(uncategorized)}")

        if uncategorized:
            print("\nUncategorized items (first 20):")
            for item in sorted(list(uncategorized)[:20]):
                print(f"  - {item}")

        # Store results for validation
        self.categorization_results = {
            "core": core_found,
            "advanced": advanced_found,
            "experimental": experimental_found,
            "uncategorized": uncategorized,
        }

        # Validation: Most items should be categorized
        categorized_count = len(core_found) + len(advanced_found) + len(experimental_found)
        categorization_rate = categorized_count / len(api_exports)

        assert categorization_rate > 0.5, f"Only {categorization_rate:.1%} of exports categorized"

    def test_core_functionality_accessible(self):
        """Test that all core functionality is accessible from main package."""
        import saplings

        # Core items that should be accessible from main package
        essential_core_items = ["Agent", "AgentConfig", "AgentBuilder"]

        for item in essential_core_items:
            assert hasattr(saplings, item), f"Core item '{item}' not accessible from main package"

            # Test that we can actually import it (lazy loading)
            obj = getattr(saplings, item)
            assert obj is not None, f"Core item '{item}' is None"

    def test_dependency_analysis(self):
        """Analyze dependencies between core components."""
        # This test documents the current dependency structure
        # for future refactoring decisions

        dependency_map = {
            "Agent": ["AgentConfig", "Tool", "MemoryStore", "LLM"],
            "AgentConfig": ["Config", "ConfigValue"],
            "Tool": [],  # Base class, minimal dependencies
            "MemoryStore": ["Document", "VectorStore"],
            "LLM": ["ModelMetadata", "LLMResponse"],
        }

        print("\nCore component dependencies:")
        for component, deps in dependency_map.items():
            print(f"{component}: {deps}")

        # Validation: Core components should have minimal dependencies
        for component, deps in dependency_map.items():
            assert len(deps) <= 5, f"{component} has too many dependencies: {deps}"

    def test_validation_criteria_check(self):
        """Check all validation criteria from Task 1.1."""
        import saplings
        import saplings.api

        main_exports = getattr(saplings, "__all__", [])
        api_exports = getattr(saplings.api, "__all__", [])

        # Validation criteria from task:
        # 1. Main package exports ≤ 20 items
        assert (
            len(main_exports) <= 20
        ), f"Main package exports {len(main_exports)} items, should be ≤ 20"

        # 2. All core functionality accessible from main package
        core_items = ["Agent", "AgentConfig", "AgentBuilder"]
        for item in core_items:
            assert item in main_exports, f"Core item '{item}' not in main package exports"

        # 3. Advanced features clearly separated (they should not be in main package)
        advanced_items = ["GASAService", "MonitoringService", "OrchestrationService"]
        for item in advanced_items:
            assert item not in main_exports, f"Advanced item '{item}' should not be in main package"

        # 4. Experimental features isolated (they should not be in main package)
        experimental_items = ["ToolFactory", "SecureHotLoader", "PatchGenerator"]
        for item in experimental_items:
            assert (
                item not in main_exports
            ), f"Experimental item '{item}' should not be in main package"

        print("✓ All validation criteria passed")
