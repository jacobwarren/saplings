from __future__ import annotations

"""
Shadow model tokenizer for GASA with third-party LLM APIs.

This module provides a tokenizer implementation that uses a small local model
for tokenization when working with third-party LLM APIs like OpenAI and Anthropic.
This allows GASA to work effectively even when the underlying LLM is a "black-box" API.
"""


import logging

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Any, Union  # Import Union and TYPE_CHECKING

from saplings.core.tokenizer import TokenizerInterface, TokenizerOutput  # Moved import up
from saplings.tokenizers.simple_tokenizer import SimpleTokenizer  # Moved import up

# Try to import transformers
if TYPE_CHECKING:
    # These imports are only for type hinting if transformers is installed
    try:
        import torch
        from transformers.modeling_utils import PreTrainedModel
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

        # Type hints for static type checking only
        _TokenizerTypeHint = Union[SimpleTokenizer, PreTrainedTokenizerBase]
        _ModelTypeHint = Union[PreTrainedModel, None]
        _CallReturnTypeHint = Union[BatchEncoding, TokenizerOutput]
    except ImportError:
        # Define fallbacks if imports fail even during type checking (less likely)
        _TokenizerTypeHint = Any
        _ModelTypeHint = Any
        _CallReturnTypeHint = Any
else:
    # Runtime: Define as Any if transformers not available, otherwise rely on actual types
    # These are intentionally unreachable but needed for type checking
    _TokenizerTypeHint = Any
    _ModelTypeHint = Any
    _CallReturnTypeHint = Any


try:
    import torch
    from transformers.modeling_utils import PreTrainedModel

    # Removed duplicate torch and AutoTokenizer imports below
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

    # Use specific import paths as suggested by Pylance
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

    TRANSFORMERS_AVAILABLE = True

except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Runtime aliases remain Any if transformers is not available
    logger.warning("Transformers not installed. Please install it with: pip install transformers")

# Check if we're on Apple Silicon
import platform

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().startswith("arm")

# Try to import vLLM (for GPU acceleration)
try:
    import vllm

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning(
        "vLLM not installed or not compatible with this environment. Using CPU-only mode."
    )

# Try to import triton (for GPU acceleration)
try:
    import triton  # type: ignore # Suppress potential unresolved import error if triton is optional/env-specific

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    if not IS_APPLE_SILICON:
        logger.warning("Triton not installed. Some GPU acceleration features may not be available.")

# Imports moved above the try block


class ShadowModelTokenizer(TokenizerInterface):
    """
    Shadow model tokenizer for GASA with third-party LLM APIs.

    This class loads a small local model for tokenization when working with
    third-party LLM APIs like OpenAI and Anthropic. This allows GASA to work
    effectively even when the underlying LLM is a "black-box" API.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        use_fast: bool = True,
        device: str = "cpu",
        cache_dir: str | None = None,
        fallback_to_simple: bool = True,
        cpu_only: bool = False,
        alternative_models: list[str] | None = None,
    ) -> None:
        """
        Initialize the shadow model tokenizer.

        Args:
        ----
            model_name: Name of the model to use for tokenization
            use_fast: Whether to use the fast tokenizer implementation
            device: Device to use for the model (cpu or cuda)
            cache_dir: Directory to cache the model
            fallback_to_simple: Whether to fall back to SimpleTokenizer if transformers is not available
            cpu_only: Force CPU-only mode even if GPU is available
            alternative_models: List of alternative models to try if the primary model fails

        """
        self.model_name = model_name
        self.use_fast = use_fast
        self.device = device
        self.cache_dir = cache_dir
        self.fallback_to_simple = fallback_to_simple
        self.cpu_only = cpu_only
        self.alternative_models = alternative_models or [
            "distilgpt2",  # Small and widely compatible
            "gpt2",  # Widely available
            "EleutherAI/pythia-70m",  # Very small model
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small but capable
        ]

        # Force CPU mode on Apple Silicon
        if IS_APPLE_SILICON:
            self.device = "cpu"
            self.cpu_only = True
            logger.info("Detected Apple Silicon. Forcing CPU-only mode.")

        # Initialize tokenizer and model with simplified type hints
        self.tokenizer: Any | None = None
        self.model: Any | None = None  # Model loading is commented out, but hint is good practice

        # Try to load the tokenizer
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the tokenizer and model."""
        if not TRANSFORMERS_AVAILABLE:
            if self.fallback_to_simple:
                logger.warning("Transformers not available, falling back to SimpleTokenizer")
                self.tokenizer = SimpleTokenizer()
            else:
                msg = "Transformers not installed. Please install it with: pip install transformers"
                raise ImportError(msg)
            return

        # Try to load the primary model
        if not self._try_load_model(self.model_name):
            # If the primary model fails, try the alternative models
            logger.warning(
                f"Failed to load primary model: {self.model_name}. Trying alternative models."
            )
            for alt_model in self.alternative_models:
                if self._try_load_model(alt_model):
                    logger.info(f"Successfully loaded alternative model: {alt_model}")
                    break
            else:
                # If all models fail, fall back to SimpleTokenizer
                if self.fallback_to_simple:
                    logger.warning("All models failed to load. Falling back to SimpleTokenizer")
                    self.tokenizer = SimpleTokenizer()
                else:
                    msg = "Failed to load any model"
                    raise ValueError(msg)

    def _try_load_model(self, model_name: str) -> bool:
        """
        Try to load a model.

        Args:
        ----
            model_name: Name of the model to load

        Returns:
        -------
            bool: True if the model was loaded successfully, False otherwise

        """
        try:
            # Load the tokenizer
            logger.info(f"Loading tokenizer for model: {model_name}")

            # Set trust_remote_code based on the model name
            # Some models like Qwen require trust_remote_code=True
            trust_remote_code = "qwen" in model_name.lower() or "mistral" in model_name.lower()

            # Ensure transformers are available before calling AutoTokenizer
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library is required but not installed.")

            tokenizer_instance = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=self.use_fast,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer = tokenizer_instance  # Assign after successful load

            # We don't actually need to load the full model for tokenization,
            # but we'll keep this code commented out in case we need it later
            # if not self.cpu_only and not IS_APPLE_SILICON:
            #     logger.info(f"Loading model: {model_name}")
            #     self.model = AutoModelForCausalLM.from_pretrained(
            #         model_name,
            #         cache_dir=self.cache_dir,
            #         torch_dtype=torch.float16,
            #         device_map=self.device,
            #         trust_remote_code=trust_remote_code,
            #     )

            logger.info(f"Successfully loaded tokenizer for model: {model_name}")
            return True
        except Exception as e:
            logger.exception(f"Error loading tokenizer for {model_name}: {e}")
            return False

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into tokens.

        Args:
        ----
            text: Text to tokenize

        Returns:
        -------
            List[str]: List of tokens

        """
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # Use the tokenizer's tokenize method if available (covers SimpleTokenizer and HF)
        if hasattr(self.tokenizer, "tokenize"):
            result = self.tokenizer.tokenize(text)
            # Ensure result is list[str]
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
            logger.warning(
                f"Unexpected return type from tokenize: {type(result)}. Returning empty list."
            )
            return []

        # Fallback: Should ideally not be reached if TokenizerInterface is implemented
        logger.warning("Tokenizer does not have a 'tokenize' method. Returning empty list.")
        return []

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        Convert tokens to token IDs.

        Args:
        ----
            tokens: List of tokens

        Returns:
        -------
            List[int]: List of token IDs

        """
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # Use the tokenizer's convert_tokens_to_ids method if available (covers SimpleTokenizer and HF)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            result = self.tokenizer.convert_tokens_to_ids(tokens)
            # Ensure result is list[int]
            if isinstance(result, list) and all(isinstance(item, int) for item in result):
                return result
            if isinstance(result, int):  # Handle single token case if tokenizer returns int
                return [result]
            logger.warning(
                f"Unexpected return type from convert_tokens_to_ids: {type(result)}. Returning empty list."
            )
            return []

        # Fallback: Should ideally not be reached if TokenizerInterface is implemented
        logger.warning(
            "Tokenizer does not have a 'convert_tokens_to_ids' method. Returning empty list."
        )
        return []

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        Convert token IDs to tokens.

        Args:
        ----
            ids: List of token IDs

        Returns:
        -------
            List[str]: List of tokens

        """
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # Use the tokenizer's convert_ids_to_tokens method if available (covers SimpleTokenizer and HF)
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            # Handle both single ID and list of IDs input for HF tokenizers
            # SimpleTokenizer's method handles list[int] directly
            if isinstance(self.tokenizer, SimpleTokenizer):
                result = self.tokenizer.convert_ids_to_tokens(ids)
            # Ensure the underlying tokenizer actually has the method before list comprehension
            elif hasattr(self.tokenizer, "convert_ids_to_tokens"):
                result = [self.tokenizer.convert_ids_to_tokens(id_val) for id_val in ids]
            else:
                logger.warning(
                    "Underlying tokenizer missing 'convert_ids_to_tokens' method. Returning empty list."
                )
                return []

            # Ensure result is list[str]
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
            if isinstance(result, str):  # Handle single ID case if tokenizer returns str
                return [result]
            logger.warning(
                f"Unexpected return type from convert_ids_to_tokens: {type(result)}. Returning empty list."
            )
            return []

        # Fallback: Should ideally not be reached if TokenizerInterface is implemented
        logger.warning(
            "Tokenizer does not have a 'convert_ids_to_tokens' method. Returning empty list."
        )
        return []

    # Note: The first __call__ definition has been removed.

    def __call__(
        self, text: str, return_tensors: str | None = None
    ) -> Any:  # Simplified return type hint
        """
        Tokenize text and return a compatible object.

        Args:
        ----
            text: Text to tokenize
            return_tensors: Format of tensors to return

        Returns:
        -------
            Union[BatchEncoding, TokenizerOutput]: Object with input_ids attribute

        """
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # Use the tokenizer to tokenize the text
        if callable(self.tokenizer):
            # If the tokenizer has a __call__ method, use it directly
            return self.tokenizer(text, return_tensors=return_tensors)

        # Otherwise, create a compatible TokenizerOutput object manually
        # Note: This path might be less common if using standard HF tokenizers
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)

        # Ensure input_ids is list[int] before passing to TokenizerOutput
        final_input_ids: list[int] = []  # Initialize with correct type
        if isinstance(input_ids, list):
            if all(isinstance(i, int) for i in input_ids):
                final_input_ids = input_ids
            # Handle nested lists if they somehow occur
            elif input_ids and isinstance(input_ids[0], list):
                inner_list = input_ids[0]
                if all(isinstance(i, int) for i in inner_list):
                    final_input_ids = inner_list
                else:
                    logger.warning(
                        "Nested list found in input_ids, but inner list contains non-integers."
                    )
            else:
                logger.warning(f"List found for input_ids, but contains non-integers: {input_ids}")
        elif isinstance(input_ids, int):  # Should generally not happen
            final_input_ids = [input_ids]
        else:
            logger.warning(
                f"Unexpected type or structure for input_ids: {type(input_ids)}. Creating empty TokenizerOutput."
            )
            # final_input_ids remains []

        # Use the standard TokenizerOutput class from core.tokenizer
        return TokenizerOutput(final_input_ids)

    @property
    def vocab_size(self) -> int:  # Added return type hint
        """Get the vocabulary size."""
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        if hasattr(self.tokenizer, "vocab_size"):
            # Ensure the return value is an int
            size = getattr(self.tokenizer, "vocab_size", 50257)  # Provide default
            return int(size) if size is not None else 50257
        # If the tokenizer doesn't have a vocab_size attribute, use a default value
        return 50257  # Default value for GPT-2

    @property
    def unk_token_id(self) -> int:
        """Get the unknown token ID."""
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # If the underlying tokenizer has unk_token_id, use it
        if hasattr(self.tokenizer, "unk_token_id"):
            unk_id = self.tokenizer.unk_token_id
            if isinstance(unk_id, int):
                return unk_id

        # Otherwise, try to get it from special tokens
        special_tokens = self.special_tokens
        for key in ["<unk>", "[UNK]", "UNK"]:
            if key in special_tokens:
                return special_tokens[key]

        # Default fallback
        return 3  # Common default for unknown token ID

    @property
    def special_tokens(self) -> dict[str, int]:  # Added return type hint
        """Get the special tokens."""
        if self.tokenizer is None:
            msg = "Tokenizer not initialized"
            raise ValueError(msg)

        # If the underlying tokenizer is SimpleTokenizer, use its property
        if isinstance(self.tokenizer, SimpleTokenizer):
            return self.tokenizer.special_tokens

        # Otherwise, try the Hugging Face approach
        if hasattr(self.tokenizer, "special_tokens_map") and hasattr(
            self.tokenizer, "convert_tokens_to_ids"
        ):
            special_tokens_map = getattr(self.tokenizer, "special_tokens_map", {})
            special_tokens_dict = {}
            # Ensure convert_tokens_to_ids exists before proceeding
            convert_method = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            if convert_method is None:
                logger.warning("Underlying tokenizer missing convert_tokens_to_ids method.")
                # Fall through to default return

            else:
                for token_name, token_value in special_tokens_map.items():
                    # Handle AddedToken objects which might be in the map values
                    if hasattr(token_value, "content"):
                        token_str = token_value.content
                    elif isinstance(token_value, str):
                        token_str = token_value
                    else:
                        continue  # Skip if not a string or AddedToken

                    if isinstance(token_str, str):
                        try:
                            # Use the tokenizer's conversion method
                            token_id = convert_method(token_str)  # Call the retrieved method
                            if isinstance(token_id, int):  # Ensure it's an int
                                special_tokens_dict[token_str] = token_id
                            elif (
                                isinstance(token_id, list)
                                and len(token_id) == 1
                                and isinstance(token_id[0], int)
                            ):
                                # Some tokenizers might return a list for single token
                                special_tokens_dict[token_str] = token_id[0]

                        except Exception as e:
                            logger.warning(
                                f"Could not convert special token '{token_str}' to ID: {e}"
                            )

                # Add common special tokens if they weren't in the map explicitly
                # This ensures essential tokens like <unk> are present if possible
                common_special_tokens = ["<unk>", "[UNK]", "<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"]
                for token_str in common_special_tokens:
                    if token_str not in special_tokens_dict:
                        try:
                            token_id = convert_method(token_str)
                            if isinstance(token_id, int):
                                special_tokens_dict[token_str] = token_id
                            elif (
                                isinstance(token_id, list)
                                and len(token_id) == 1
                                and isinstance(token_id[0], int)
                            ):
                                special_tokens_dict[token_str] = token_id[0]
                        except Exception:
                            pass  # Ignore if conversion fails for these common ones

                # Add GASA specific token if not present
                if "[SUM]" not in special_tokens_dict:
                    special_tokens_dict["[SUM]"] = 999  # Default GASA summary token ID

                return special_tokens_dict

        # Fallback default values if no map is found or conversion method missing
        logger.warning("Could not determine special tokens from tokenizer, using defaults.")
        return {
            "<s>": 1,
            "</s>": 2,
            "<pad>": 0,
            "<unk>": 3,
            "[CLS]": 101,
            "[SEP]": 102,
            "[PAD]": 0,
            "[UNK]": 100,
            "[SUM]": 999,  # Special token for GASA summary
        }
