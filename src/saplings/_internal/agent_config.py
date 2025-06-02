from __future__ import annotations

"""
Agent configuration module for Saplings.

This module provides the AgentConfig class which centralizes all configuration
options for the Agent and AgentFacade classes, making it easy to customize
behavior while providing sensible defaults.
"""


import os
from typing import Any, ClassVar, Dict, List, Optional


class AgentConfig:
    """
    Configuration for the Agent and AgentFacade classes.

    This class centralizes all configuration options,
    making it easy to customize behavior while providing sensible defaults.

    For common use cases, use the preset factory methods:
    - `minimal()`: Minimal configuration with only essential features
    - `standard()`: Standard configuration with balanced features
    - `full_featured()`: Full-featured configuration with all features enabled
    - `for_openai()`: Configuration optimized for OpenAI models
    - `for_anthropic()`: Configuration optimized for Anthropic models
    - `for_vllm()`: Configuration optimized for vLLM models
    """

    # Configuration presets as class variables
    PRESET_MINIMAL: ClassVar[Dict[str, Any]] = {
        "enable_gasa": False,
        "enable_monitoring": False,
        "enable_self_healing": False,
        "enable_tool_factory": False,
        "planner_budget_strategy": "fixed",
        "executor_validation_type": "execution",
    }

    PRESET_STANDARD: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "enable_monitoring": True,
        "enable_self_healing": False,
        "enable_tool_factory": True,
        "planner_budget_strategy": "proportional",
        "executor_validation_type": "execution",
    }

    PRESET_FULL_FEATURED: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "enable_monitoring": True,
        "enable_self_healing": True,
        "enable_tool_factory": True,
        "gasa_max_hops": 3,
        "retrieval_max_documents": 20,
        "planner_budget_strategy": "dynamic",
        "planner_total_budget": 2.0,
        "planner_allow_budget_overflow": True,
        "executor_validation_type": "judge",
    }

    PRESET_OPENAI: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": True,
        "gasa_shadow_model_name": "Qwen/Qwen3-0.6B",
        "gasa_fallback": "prompt_composer",
        "gasa_prompt_composer": True,
        "max_tokens": 4096,
    }

    PRESET_ANTHROPIC: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": True,
        "gasa_shadow_model_name": "Qwen/Qwen3-0.6B",
        "gasa_fallback": "prompt_composer",
        "gasa_prompt_composer": True,
        "max_tokens": 4096,
    }

    PRESET_VLLM: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": False,
        "gasa_fallback": "block_diagonal",
        "model_parameters": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "trust_remote_code": True,
        },
    }

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
        enable_self_healing: bool = True,
        self_healing_max_retries: int = 3,
        enable_tool_factory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        gasa_max_hops: int = 2,
        gasa_strategy: str = "binary",
        gasa_fallback: str = "block_diagonal",
        gasa_shadow_model: bool = False,
        gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B",
        gasa_prompt_composer: bool = False,
        retrieval_entropy_threshold: float = 0.1,
        retrieval_max_documents: int = 10,
        planner_budget_strategy: str = "proportional",
        planner_total_budget: float = 1.0,
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_validation_type: str = "judge",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: list[str] | None = None,
        tools: list[Any] | None = None,
        supported_modalities: list[str] | None = None,
        **model_parameters,
    ) -> None:
        """
        Initialize the agent configuration.

        Args:
        ----
            provider: Model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: Model name
            api_key: API key for cloud providers (optional, can be set via environment variable)
            memory_path: Path to store agent memory
            output_dir: Directory to save outputs
            enable_gasa: Whether to enable Graph-Aligned Sparse Attention
            enable_monitoring: Whether to enable monitoring and tracing
            enable_self_healing: Whether to enable self-healing capabilities
            self_healing_max_retries: Maximum number of retry attempts for self-healing operations
            enable_tool_factory: Whether to enable dynamic tool creation
            max_tokens: Maximum number of tokens for model responses
            temperature: Temperature for model generation
            gasa_max_hops: Maximum number of hops for GASA mask
            gasa_strategy: Strategy for GASA mask (binary, soft, learned)
            gasa_fallback: Fallback strategy for models that don't support sparse attention
            gasa_shadow_model: Whether to enable shadow model for tokenization
            gasa_shadow_model_name: Name of the shadow model to use
            gasa_prompt_composer: Whether to enable graph-aware prompt composer
            retrieval_entropy_threshold: Entropy threshold for retrieval termination
            retrieval_max_documents: Maximum number of documents to retrieve
            planner_budget_strategy: Strategy for budget allocation
            planner_total_budget: Total budget for the planner in USD
            planner_allow_budget_overflow: Whether to allow exceeding the budget
            planner_budget_overflow_margin: Margin by which the budget can be exceeded (as a fraction)
            executor_validation_type: Strategy for output validation
            tool_factory_sandbox_enabled: Whether to enable sandbox for tool execution
            allowed_imports: List of allowed imports for dynamic tools
            tools: List of tools to register with the agent
            supported_modalities: List of supported modalities (text, image, audio, video)
            **model_parameters: Additional model parameters

        Raises:
        ------
            ValueError: If required parameters are missing or invalid
            OSError: If paths cannot be created

        """
        # Validate required parameters with helpful error messages
        self._validate_required_parameters(provider, model_name)
        self._validate_provider(provider)
        self._validate_paths(memory_path, output_dir)
        # Store provider and model_name as primary fields
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self._model_parameters = model_parameters
        self.memory_path = memory_path
        self.output_dir = output_dir
        self.enable_gasa = enable_gasa
        self.enable_monitoring = enable_monitoring
        self.enable_self_healing = enable_self_healing
        self.self_healing_max_retries = self_healing_max_retries
        self.enable_tool_factory = enable_tool_factory
        self.max_tokens = max_tokens
        self.temperature = temperature

        # GASA configuration
        self.gasa_max_hops = gasa_max_hops
        self.gasa_strategy = gasa_strategy
        self.gasa_fallback = gasa_fallback
        self.gasa_shadow_model = gasa_shadow_model
        self.gasa_shadow_model_name = gasa_shadow_model_name
        self.gasa_prompt_composer = gasa_prompt_composer

        # Retrieval configuration
        self.retrieval_entropy_threshold = retrieval_entropy_threshold
        self.retrieval_max_documents = retrieval_max_documents

        # Planner configuration
        self.planner_budget_strategy = planner_budget_strategy
        self.planner_total_budget = planner_total_budget
        self.planner_allow_budget_overflow = planner_allow_budget_overflow
        self.planner_budget_overflow_margin = planner_budget_overflow_margin

        # Executor configuration
        self.executor_validation_type = executor_validation_type

        # Tool factory configuration
        self.tool_factory_sandbox_enabled = tool_factory_sandbox_enabled
        self.allowed_imports = allowed_imports or [
            "os",
            "datetime",
            "json",
            "math",
            "numpy",
            "pandas",
        ]
        self.tools = tools or []

        # Modality support
        self.supported_modalities = supported_modalities or ["text"]

        # Validate supported modalities
        supported_modalities_list = ["text", "image", "audio", "video"]
        for modality in self.supported_modalities:
            if modality not in supported_modalities_list:
                raise ValueError(
                    f"Unsupported modality '{modality}'. Supported modalities are: {', '.join(supported_modalities_list)}. "
                    f"Please specify valid modalities. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', supported_modalities=['text', 'image'])"
                )

        # Ensure text is always supported
        if "text" not in self.supported_modalities:
            self.supported_modalities.append("text")

        # Create output directory with better error handling
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Cannot create output directory '{output_dir}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', output_dir='./agent_output')"
            ) from e

        # Create memory directory with better error handling
        try:
            os.makedirs(memory_path, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Cannot create memory directory '{memory_path}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', memory_path='./agent_memory')"
            ) from e

    @property
    def model_parameters(self):
        """
        Get model parameters.

        Returns
        -------
            Dict[str, Any]: Model parameters

        """
        return self._model_parameters

    @classmethod
    def minimal(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a minimal configuration with only essential features enabled.

        This configuration is optimized for simplicity and performance, with most
        advanced features disabled.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Minimal configuration

        """
        # Start with minimal preset
        config_params = cls.PRESET_MINIMAL.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def standard(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a standard configuration with balanced features.

        This configuration is suitable for most use cases, with a good balance
        between features and performance.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Standard configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def full_featured(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a full-featured configuration with all features enabled.

        This configuration enables all advanced features for maximum capability,
        but may be more resource-intensive.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Full-featured configuration

        """
        # Start with full-featured preset
        config_params = cls.PRESET_FULL_FEATURED.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_openai(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for OpenAI models.

        This configuration includes settings that work well with OpenAI models,
        such as appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: OpenAI-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add OpenAI-specific settings
        config_params.update(cls.PRESET_OPENAI)

        # Add required parameters
        config_params["provider"] = "openai"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_anthropic(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for Anthropic models.

        This configuration includes settings that work well with Anthropic models,
        such as appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: Anthropic model name (e.g., "claude-3-opus", "claude-3-sonnet")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Anthropic-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add Anthropic-specific settings
        config_params.update(cls.PRESET_ANTHROPIC)

        # Add required parameters
        config_params["provider"] = "anthropic"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_vllm(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for vLLM models.

        This configuration includes settings that work well with vLLM models,
        such as appropriate GASA settings for self-hosted models.

        Args:
        ----
            model_name: Model name (e.g., "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: vLLM-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add vLLM-specific settings
        config_params.update(cls.PRESET_VLLM)

        # Add required parameters
        config_params["provider"] = "vllm"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    def _validate_required_parameters(self, provider: str, model_name: str) -> None:
        """Validate that required parameters are provided with helpful error messages."""
        if not provider:
            raise ValueError(
                "The 'provider' parameter is required. Please specify a model provider such as "
                "'openai', 'anthropic', 'vllm', or 'huggingface'. "
                "Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

        if not model_name:
            raise ValueError(
                "The 'model_name' parameter is required. Please specify the name of the model to use. "
                "Examples: 'gpt-4o' for OpenAI, 'claude-3-opus' for Anthropic, or 'Qwen/Qwen3-7B-Instruct' for vLLM. "
                "Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

    def _validate_provider(self, provider: str) -> None:
        """Validate that the provider is supported with helpful error messages."""
        supported_providers = ["openai", "anthropic", "vllm", "huggingface", "test", "mock"]
        if provider.lower() not in supported_providers:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported providers are: {', '.join(supported_providers)}. "
                f"Please choose one of the supported providers. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

    def _validate_paths(self, memory_path: str, output_dir: str) -> None:
        """Validate that paths are valid and can be created."""
        from pathlib import Path

        # Validate memory path
        try:
            memory_path_obj = Path(memory_path)
            if memory_path_obj.exists() and not memory_path_obj.is_dir():
                raise ValueError(
                    f"Memory path '{memory_path}' exists but is not a directory. "
                    f"Please specify a valid directory path for memory storage. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', memory_path='./agent_memory')"
                )
        except (OSError, ValueError) as e:
            if "not a directory" in str(e):
                raise e
            raise ValueError(
                f"Invalid memory path '{memory_path}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', memory_path='./agent_memory')"
            ) from e

        # Validate output directory
        try:
            output_dir_obj = Path(output_dir)
            if output_dir_obj.exists() and not output_dir_obj.is_dir():
                raise ValueError(
                    f"Output directory '{output_dir}' exists but is not a directory. "
                    f"Please specify a valid directory path for output storage. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', output_dir='./agent_output')"
                )
        except (OSError, ValueError) as e:
            if "not a directory" in str(e):
                raise e
            raise ValueError(
                f"Invalid output directory '{output_dir}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', output_dir='./agent_output')"
            ) from e

    def validate(self) -> "ValidationResult":
        """
        Validate the complete configuration.

        Returns
        -------
            ValidationResult: Result of the validation with errors and suggestions

        """
        errors = []
        suggestions = []

        # Check for API key for cloud providers
        if self.provider in ["openai", "anthropic"]:
            api_key = getattr(self, "api_key", None) or os.getenv(
                f"{self.provider.upper()}_API_KEY"
            )
            if not api_key:
                errors.append(f"{self.provider.title()} API key not found")
                suggestions.extend(
                    [
                        f"Set environment variable: export {self.provider.upper()}_API_KEY=your-key",
                        f"Pass api_key parameter: AgentConfig(provider='{self.provider}', api_key='your-key')",
                    ]
                )

        # Check model name validity (basic check for obviously invalid names)
        if self.model_name in ["invalid-model", "test-invalid"]:
            errors.append(f"Invalid model '{self.model_name}' for provider '{self.provider}'")
            if self.provider == "openai":
                suggestions.append("Valid models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
            elif self.provider == "anthropic":
                suggestions.append("Valid models: claude-3-opus, claude-3-sonnet, claude-3-haiku")

        if errors:
            return ValidationResult(
                is_valid=False,
                message="; ".join(errors),
                suggestions=suggestions,
                help_url=f"https://docs.saplings.ai/setup/{self.provider}",
            )

        return ValidationResult(is_valid=True)

    def check_dependencies(self) -> "DependencyResult":
        """
        Check that required dependencies are available.

        Returns
        -------
            DependencyResult: Result of dependency checking

        """
        missing = []

        # Check provider-specific dependencies
        if self.provider == "huggingface":
            try:
                import transformers
            except ImportError:
                missing.append("transformers")

        if self.provider == "vllm":
            try:
                import vllm
            except ImportError:
                missing.append("vllm")

        # Check optional feature dependencies
        if self.enable_gasa:
            try:
                import torch
            except ImportError:
                missing.append("torch")

        if missing:
            return DependencyResult(
                all_available=False,
                message=f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}",
                missing_dependencies=missing,
            )

        return DependencyResult(all_available=True)

    def suggest_fixes(self) -> List[str]:
        """
        Provide actionable suggestions for configuration issues.

        Returns
        -------
            List[str]: List of actionable suggestions

        """
        suggestions = []

        # Check for missing API keys
        if self.provider in ["openai", "anthropic"]:
            api_key = getattr(self, "api_key", None) or os.getenv(
                f"{self.provider.upper()}_API_KEY"
            )
            if not api_key:
                suggestions.extend(
                    [
                        f"export {self.provider.upper()}_API_KEY=your-api-key",
                        f"AgentConfig(provider='{self.provider}', api_key='your-key')",
                    ]
                )

        # Check dependencies
        dep_result = self.check_dependencies()
        if not dep_result.all_available:
            suggestions.append(f"pip install {' '.join(dep_result.missing_dependencies)}")

        return suggestions

    def explain(self) -> str:
        """
        Explain what each configuration option does.

        Returns
        -------
            str: Human-readable explanation of the configuration

        """
        return f"""Configuration for {self.provider} provider:
- Model: {self.model_name}
- GASA enabled: {self.enable_gasa}
- Monitoring enabled: {self.enable_monitoring}
- Memory path: {self.memory_path}
- Output directory: {self.output_dir}
- Tool factory enabled: {self.enable_tool_factory}
- Self-healing enabled: {self.enable_self_healing}
- Supported modalities: {', '.join(self.supported_modalities)}"""

    def compare(self, other: "AgentConfig") -> str:
        """
        Compare this configuration with another configuration.

        Args:
        ----
            other: Another AgentConfig to compare with

        Returns:
        -------
            str: Human-readable comparison showing differences

        """
        if not isinstance(other, AgentConfig):
            return "Cannot compare: other object is not an AgentConfig"

        differences = []

        # Compare basic settings
        if self.provider != other.provider:
            differences.append(f"Provider: {self.provider} vs {other.provider}")
        if self.model_name != other.model_name:
            differences.append(f"Model: {self.model_name} vs {other.model_name}")

        # Compare feature flags
        feature_comparisons = [
            ("GASA", self.enable_gasa, other.enable_gasa),
            ("Monitoring", self.enable_monitoring, other.enable_monitoring),
            ("Self-healing", self.enable_self_healing, other.enable_self_healing),
            ("Tool factory", self.enable_tool_factory, other.enable_tool_factory),
        ]

        for feature_name, self_value, other_value in feature_comparisons:
            if self_value != other_value:
                differences.append(f"{feature_name}: {self_value} vs {other_value}")

        # Compare paths
        if self.memory_path != other.memory_path:
            differences.append(f"Memory path: {self.memory_path} vs {other.memory_path}")
        if self.output_dir != other.output_dir:
            differences.append(f"Output dir: {self.output_dir} vs {other.output_dir}")

        if not differences:
            return "Configurations are identical"

        return "Configuration differences:\n" + "\n".join(f"- {diff}" for diff in differences)


class ValidationResult:
    """Result of configuration validation."""

    def __init__(
        self,
        is_valid: bool = True,
        message: str = "",
        suggestions: Optional[List[str]] = None,
        help_url: Optional[str] = None,
    ):
        self.is_valid = is_valid
        self.message = message
        self.suggestions = suggestions or []
        self.help_url = help_url


class DependencyResult:
    """Result of dependency checking."""

    def __init__(
        self,
        all_available: bool = True,
        message: str = "",
        missing_dependencies: Optional[List[str]] = None,
    ):
        self.all_available = all_available
        self.message = message
        self.missing_dependencies = missing_dependencies or []
