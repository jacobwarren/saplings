from __future__ import annotations

"""
Cache key generation module for Saplings.

This module provides utilities for generating cache keys with a consistent
and efficient approach across different cache layers.
"""


import hashlib
import json
from typing import Any


class KeyBuilder:
    """
    Utility class for building cache keys with a consistent format.

    This class provides methods for building cache keys from various data types,
    ensuring consistency across different cache layers.
    """

    @staticmethod
    def _normalize_dict(data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a dictionary for consistent key generation.

        Args:
        ----
            data: Dictionary to normalize

        Returns:
        -------
            Dict[str, Any]: Normalized dictionary

        """
        result = {}

        # Sort keys to ensure consistent order
        for key in sorted(data.keys()):
            value = data[key]

            # Recursively normalize nested dictionaries
            if isinstance(value, dict):
                result[key] = KeyBuilder._normalize_dict(value)
            # Normalize lists
            elif isinstance(value, list):
                result[key] = KeyBuilder._normalize_list(value)
            # For simple types, use as is
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            # For complex types, use their string representation
            else:
                result[key] = str(value)

        return result

    @staticmethod
    def _normalize_list(data: list[Any]) -> list[Any]:
        """
        Normalize a list for consistent key generation.

        Args:
        ----
            data: List to normalize

        Returns:
        -------
            List[Any]: Normalized list

        """
        result = []

        for item in data:
            # Recursively normalize nested dictionaries
            if isinstance(item, dict):
                result.append(KeyBuilder._normalize_dict(item))
            # Recursively normalize nested lists
            elif isinstance(item, list):
                result.append(KeyBuilder._normalize_list(item))
            # For simple types, use as is
            elif isinstance(item, (str, int, float, bool, type(None))):
                result.append(item)
            # For complex types, use their string representation
            else:
                result.append(str(item))

        return result

    @staticmethod
    def build(namespace: str, data: Any, **additional_params) -> str:
        """
        Build a cache key from data and additional parameters.

        Args:
        ----
            namespace: Namespace for the key
            data: Primary data for the key
            **additional_params: Additional parameters to include in the key

        Returns:
        -------
            str: Cache key

        """
        # Create a dictionary for all parameters
        params: dict[str, Any] = {"namespace": namespace}

        # Add data based on its type
        if isinstance(data, dict):
            params["data"] = KeyBuilder._normalize_dict(data)  # type: ignore
        elif isinstance(data, list):
            params["data"] = KeyBuilder._normalize_list(data)  # type: ignore
        elif isinstance(data, str):
            params["data"] = data
        else:
            params["data"] = str(data)

        # Add additional parameters
        for key, value in additional_params.items():
            if isinstance(value, dict):
                params[key] = KeyBuilder._normalize_dict(value)  # type: ignore
            elif isinstance(value, list):
                params[key] = KeyBuilder._normalize_list(value)  # type: ignore
            elif isinstance(value, (str, int, float, bool, type(None))):
                params[key] = value  # type: ignore
            else:
                params[key] = str(value)

        # Serialize the parameters to a string
        params_str = json.dumps(params, sort_keys=True)

        # Hash the string to create a cache key
        return hashlib.md5(params_str.encode()).hexdigest()

    @staticmethod
    def build_model_key(cache_key: str, prompt: str | list[dict[str, Any]], **kwargs) -> str:
        """
        Build a cache key for a model request.

        Args:
        ----
            cache_key: Key identifying the model (provider:model_name)
            prompt: Prompt for the model
            **kwargs: Additional parameters

        Returns:
        -------
            str: Cache key

        """
        # Create data dictionary
        data = {
            "cache_key": cache_key,
        }

        # Add prompt based on its type
        if isinstance(prompt, list):
            data["prompt"] = KeyBuilder._normalize_list(prompt)  # type: ignore
        else:
            data["prompt"] = prompt

        return KeyBuilder.build("model", data, **kwargs)

    @staticmethod
    def build_vector_key(collection: str, query: str | list[float], **kwargs) -> str:
        """
        Build a cache key for a vector store query.

        Args:
        ----
            collection: Name of the vector collection
            query: Query text or embedding
            **kwargs: Additional parameters

        Returns:
        -------
            str: Cache key

        """
        # Create data dictionary
        data = {
            "collection": collection,
        }

        # Add query based on its type
        if isinstance(query, list):
            # For embeddings, we use a hash of the vector
            # (exact comparison would be too strict)
            query_str = json.dumps(query, sort_keys=True)
            data["query"] = hashlib.md5(query_str.encode()).hexdigest()
        else:
            data["query"] = query

        return KeyBuilder.build("vector", data, **kwargs)

    @staticmethod
    def build_function_key(func_name: str, args: tuple, kwargs: dict[str, Any]) -> str:
        """
        Build a cache key for a function call.

        Args:
        ----
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
        -------
            str: Cache key

        """
        # Create data dictionary
        data = {
            "func_name": func_name,
            "args": KeyBuilder._normalize_list(list(args)),
            "kwargs": KeyBuilder._normalize_dict(kwargs),
        }

        return KeyBuilder.build("function", data)
