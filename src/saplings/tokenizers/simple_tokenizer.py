from __future__ import annotations

"""
Simple tokenizer implementation for Saplings.

This module provides a simple tokenizer that can be used when a more sophisticated
tokenizer is not available. It's primarily used for GASA mask generation.
"""


import re

from saplings.core.tokenizer import TokenizerInterface, TokenizerOutput


class SimpleTokenizer(TokenizerInterface):
    """
    A simple tokenizer for use with GASA when a model-specific tokenizer is not available.

    This tokenizer implements a basic word-level tokenization strategy with special token
    handling. It's not meant to be an accurate representation of any specific model's
    tokenization, but rather a lightweight alternative for GASA mask generation.
    """

    def __init__(self, vocab_size: int = 50257) -> None:
        """
        Initialize the simple tokenizer.

        Args:
        ----
            vocab_size: Size of the vocabulary

        """
        self._vocab_size = vocab_size

        # Special tokens
        self._special_tokens = {
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

        # Token to ID mapping (will be built on demand)
        self.token_to_id: dict[str, int] = self._special_tokens.copy()
        self.next_id = 1000  # Start regular token IDs after special tokens

        # Special attributes needed for compatibility
        self.unk_token_id = self._special_tokens["<unk>"]

    @property
    def vocab_size(self):
        """
        Get the vocabulary size.

        Returns
        -------
            int: The vocabulary size

        """
        return self._vocab_size

    @property
    def special_tokens(self):
        """
        Get the special tokens.

        Returns
        -------
            Dict[str, int]: The special tokens

        """
        return self._special_tokens

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
        # Handle special tokens first
        for token in self._special_tokens:
            text = text.replace(token, f" {token} ")

        # Simple word-level tokenization
        tokens = []
        for word in re.findall(r"\b\w+\b|[^\w\s]", text):
            # Check if it's a special token
            if word in self._special_tokens:
                tokens.append(word)
            else:
                # For regular words, we could split them further
                # but for simplicity, we'll just use the word as is
                tokens.append(word)

        return tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        Convert tokens to IDs.

        Args:
        ----
            tokens: Token or list of tokens to convert

        Returns:
        -------
            Union[int, List[int]]: Token ID or list of token IDs

        """
        if isinstance(tokens, str):
            return self._get_token_id(tokens)

        return [self._get_token_id(token) for token in tokens]

    def _get_token_id(self, token: str) -> int:
        """
        Get the ID for a token, assigning a new ID if not seen before.

        Args:
        ----
            token: Token to get ID for

        Returns:
        -------
            int: Token ID

        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        # Assign a new ID
        self.token_to_id[token] = self.next_id
        self.next_id += 1

        # If we exceed vocab size, wrap around
        if self.next_id >= self._vocab_size:
            self.next_id = 1000

        return self.token_to_id[token]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        """
        Convert token IDs to tokens.

        Args:
        ----
            ids: Token ID or list of token IDs to convert

        Returns:
        -------
            Union[str, List[str]]: Token or list of tokens

        """
        # If a single ID is provided, convert it to a list
        if isinstance(ids, int):
            ids = [ids]

        # Reverse lookup in token_to_id dictionary
        reverse_dict = {id: token for token, id in self.token_to_id.items()}

        # Convert IDs to tokens
        return [reverse_dict.get(id, f"<{id}>") for id in ids]

    def __call__(self, text: str, return_tensors: str | None = None) -> object:
        """
        Tokenize text and return a compatible object.

        Args:
        ----
            text: Text to tokenize
            return_tensors: Format of tensors to return (ignored in this implementation)

        Returns:
        -------
            object: Object with input_ids attribute

        """
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)

        # Ensure input_ids is a list
        if isinstance(input_ids, int):
            input_ids = [input_ids]

        # Use the standard TokenizerOutput class from core.tokenizer
        return TokenizerOutput(input_ids)
