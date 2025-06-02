from __future__ import annotations

"""
Core tokenizer interface and utilities for Saplings.

This module provides a standardized tokenizer interface that can be implemented
by different tokenizer implementations, reducing duplication and improving consistency.
"""


from abc import ABC, abstractmethod


class TokenizerInterface(ABC):
    """
    Abstract base class for tokenizers in Saplings.

    This interface defines the common methods that all tokenizers should implement,
    ensuring consistency across different tokenizer implementations and making them
    interchangeable.
    """

    @abstractmethod
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

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        Convert tokens to token IDs.

        Args:
        ----
            tokens: Token or list of tokens to convert

        Returns:
        -------
            Union[int, List[int]]: Token ID or list of token IDs

        """

    @abstractmethod
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

    @abstractmethod
    def __call__(self, text: str, return_tensors: str | None = None) -> object:
        """
        Tokenize text and return a compatible object.

        Args:
        ----
            text: Text to tokenize
            return_tensors: Format of tensors to return

        Returns:
        -------
            object: Object with input_ids attribute

        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns
        -------
            int: The vocabulary size

        """

    @property
    @abstractmethod
    def special_tokens(self) -> dict[str, int]:
        """
        Get the special tokens.

        Returns
        -------
            Dict[str, int]: The special tokens

        """


class TokenizerOutput:
    """
    Standard output class for tokenizers.

    This class provides a standardized output format that all tokenizers can use,
    ensuring consistency in how tokenized outputs are represented and accessed.
    """

    def __init__(self, input_ids: list[int]) -> None:
        """
        Initialize the tokenizer output.

        Args:
        ----
            input_ids: The token IDs

        """
        self.input_ids = [input_ids]

    def tolist(self) -> list[list[int]]:
        """
        Return a list representation of input_ids.

        Returns
        -------
            List[List[int]]: The input IDs as a list of lists

        """
        return self.input_ids

    def __getattr__(self, name: str) -> object | None:
        """
        Handle attribute access gracefully.

        Args:
        ----
            name: The attribute name

        Returns:
        -------
            Optional[object]: The attribute value or None

        """
        if name == "shape":
            return [1, len(self.input_ids[0])]
        return None
