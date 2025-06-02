#!/usr/bin/env python3
"""
API Export Audit Script for Task 1.1

This script audits the current API surface and categorizes exports according to
the requirements in finish.md Task 1.1.
"""

from __future__ import annotations

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def audit_main_package():
    """Audit the main package exports."""
    print("=== Main Package Audit ===")

    import saplings

    main_exports = getattr(saplings, "__all__", [])

    print(f"Main package exports: {len(main_exports)} items")
    print("Items:")
    for item in sorted(main_exports):
        print(f"  - {item}")

    return main_exports


def audit_api_package():
    """Audit the API package exports."""
    print("\n=== API Package Audit ===")

    import saplings.api

    api_exports = getattr(saplings.api, "__all__", [])

    print(f"API package exports: {len(api_exports)} items")

    return api_exports


def categorize_exports(api_exports):
    """Categorize API exports into core, advanced, and experimental."""
    print("\n=== Export Categorization ===")

    # Define categorization based on the task requirements
    CORE_ITEMS = {
        # Core Agent functionality
        "Agent",
        "AgentBuilder",
        "AgentConfig",
        # Essential tools
        "Tool",
        "PythonInterpreterTool",
        "FinalAnswerTool",
        "DuckDuckGoSearchTool",
        "GoogleSearchTool",
        "WikipediaSearchTool",
        "UserInputTool",
        "ToolRegistry",
        "ToolCollection",
        # Core memory and documents
        "Document",
        "DocumentMetadata",
        "MemoryStore",
        "DependencyGraph",
        "MemoryConfig",
        "VectorStore",
        "InMemoryVectorStore",
        # Core LLM functionality
        "LLM",
        "LLMBuilder",
        "LLMResponse",
        "ModelMetadata",
        "OpenAIAdapter",
        "AnthropicAdapter",
        # Core configuration and exceptions
        "Config",
        "ConfigValue",
        "SaplingsError",
        "ConfigurationError",
        "ModelError",
        "ProviderError",
        "__version__",
        # Core utilities
        "count_tokens",
        "split_text_by_tokens",
        "truncate_text_tokens",
    }

    ADVANCED_ITEMS = {
        # GASA components
        "GASAService",
        "GASAConfig",
        "GASAConfigBuilder",
        "GASAServiceBuilder",
        "MaskVisualizer",
        "BlockDiagonalPacker",
        "GraphDistanceCalculator",
        "StandardMaskBuilder",
        "TokenMapper",
        "MaskFormat",
        "MaskStrategy",
        # Monitoring components
        "MonitoringService",
        "TraceViewer",
        "BlameGraph",
        "BlameNode",
        "BlameEdge",
        "TraceManager",
        "MonitoringConfig",
        # Orchestration components
        "OrchestrationService",
        "AgentNode",
        "CommunicationChannel",
        "GraphRunner",
        "GraphRunnerConfig",
        "NegotiationStrategy",
        # Advanced retrieval
        "FaissVectorStore",
        "EmbeddingRetriever",
        "CascadeRetriever",
        "TFIDFRetriever",
        "GraphExpander",
        "EntropyCalculator",
        # Service builders and advanced services
        "ExecutionServiceBuilder",
        "MemoryManagerBuilder",
        "RetrievalServiceBuilder",
        "ModalityServiceBuilder",
        "ValidatorServiceBuilder",
        "ToolServiceBuilder",
        "ExecutionService",
        "MemoryManager",
        "RetrievalService",
        "ModalityService",
        # Advanced model adapters
        "HuggingFaceAdapter",
        "VLLMAdapter",
        # Judge system
        "JudgeAgent",
        "JudgeService",
        "JudgeConfig",
        "JudgeResult",
        "Rubric",
        "RubricItem",
        "CritiqueFormat",
        "ScoringDimension",
    }

    EXPERIMENTAL_ITEMS = {
        # Tool factory components
        "ToolFactory",
        "ToolFactoryConfig",
        "ToolSpecification",
        "ToolTemplate",
        "SecureHotLoader",
        "SecureHotLoaderConfig",
        "create_secure_hot_loader",
        # Security and sandboxing
        "CodeSigner",
        "SignatureVerifier",
        "DockerSandbox",
        "E2BSandbox",
        "Sandbox",
        "SandboxType",
        "SecurityLevel",
        "SigningLevel",
        "Sanitizer",
        "RedactingFilter",
        # Self-healing components
        "AdapterManager",
        "PatchGenerator",
        "LoRaTrainer",
        "Adapter",
        "AdapterMetadata",
        "AdapterPriority",
        "Patch",
        "PatchResult",
        "SuccessPairCollector",
        "TrainingMetrics",
        "LoRaConfig",
        "SelfHealingService",
        "SelfHealingConfig",
        "RetryStrategy",
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
    print(f"Core items found: {len(core_found)}")
    print(f"Advanced items found: {len(advanced_found)}")
    print(f"Experimental items found: {len(experimental_found)}")
    print(f"Uncategorized items: {len(uncategorized)}")

    total_categorized = len(core_found) + len(advanced_found) + len(experimental_found)
    categorization_rate = total_categorized / len(api_exports)
    print(f"Categorization rate: {categorization_rate:.1%}")

    if uncategorized:
        print("\nUncategorized items:")
        for item in sorted(uncategorized):
            print(f"  - {item}")

    return {
        "core": core_found,
        "advanced": advanced_found,
        "experimental": experimental_found,
        "uncategorized": uncategorized,
    }


def validate_criteria(main_exports, api_exports, categorization):
    """Validate all criteria from Task 1.1."""
    print("\n=== Validation Criteria Check ===")

    results = {}

    # 1. Main package exports ≤ 20 items
    results["main_exports_limit"] = len(main_exports) <= 20
    print(f"1. Main package exports ≤ 20: {results['main_exports_limit']} ({len(main_exports)}/20)")

    # 2. All core functionality accessible from main package
    essential_core = {"Agent", "AgentConfig", "AgentBuilder"}
    results["core_accessible"] = essential_core.issubset(set(main_exports))
    print(f"2. Core functionality accessible: {results['core_accessible']}")

    # 3. Advanced features clearly separated
    advanced_sample = {"GASAService", "MonitoringService", "OrchestrationService"}
    results["advanced_separated"] = not advanced_sample.intersection(set(main_exports))
    print(f"3. Advanced features separated: {results['advanced_separated']}")

    # 4. Experimental features isolated
    experimental_sample = {"ToolFactory", "SecureHotLoader", "PatchGenerator"}
    results["experimental_isolated"] = not experimental_sample.intersection(set(main_exports))
    print(f"4. Experimental features isolated: {results['experimental_isolated']}")

    # 5. Backward compatibility maintained (check that main exports work)
    try:
        import saplings

        agent_class = saplings.Agent
        config_class = saplings.AgentConfig
        results["backward_compatibility"] = agent_class is not None and config_class is not None
    except Exception as e:
        results["backward_compatibility"] = False
        print(f"   Error: {e}")
    print(f"5. Backward compatibility: {results['backward_compatibility']}")

    all_passed = all(results.values())
    print(f"\nOverall validation: {'✓ PASS' if all_passed else '✗ FAIL'}")

    return results


def main():
    """Main audit function."""
    print("API Export Audit for Task 1.1")
    print("=" * 50)

    try:
        # Audit main package
        main_exports = audit_main_package()

        # Audit API package
        api_exports = audit_api_package()

        # Categorize exports
        categorization = categorize_exports(api_exports)

        # Validate criteria
        validation_results = validate_criteria(main_exports, api_exports, categorization)

        print("\n=== Summary ===")
        print(f"Main package: {len(main_exports)} exports")
        print(f"API package: {len(api_exports)} exports")
        print(f"Core items: {len(categorization['core'])}")
        print(f"Advanced items: {len(categorization['advanced'])}")
        print(f"Experimental items: {len(categorization['experimental'])}")
        print(f"Uncategorized: {len(categorization['uncategorized'])}")

        if all(validation_results.values()):
            print("\n✓ Task 1.1 audit completed successfully!")
            return 0
        else:
            print("\n✗ Some validation criteria failed")
            return 1

    except Exception as e:
        print(f"Error during audit: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
