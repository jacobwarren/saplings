"""
Function caching module for Saplings.

This module provides utilities for caching function calls.
"""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FunctionCache:
    """Cache for function calls."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[int] = 3600,
        namespace: str = "default"
    ):
        """
        Initialize the function cache.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            namespace: Namespace for the cache
        """
        self.max_size = max_size
        self.ttl = ttl
        self.namespace = namespace
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found or expired
        """
        if key not in self._cache:
            return None
        
        value, expiration = self._cache[key]
        
        # Check if expired
        if self.ttl is not None and time.time() > expiration:
            # Remove expired item
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return None
        
        # Update access time
        self._access_times[key] = time.time()
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Remove least recently used item
            self._remove_lru()
        
        # Calculate expiration time
        expiration = time.time() + self.ttl if self.ttl is not None else float("inf")
        
        # Store value and update access time
        self._cache[key] = (value, expiration)
        self._access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key was deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
    
    def _remove_lru(self) -> None:
        """Remove the least recently used item from the cache."""
        if not self._access_times:
            return
        
        # Find the least recently used key
        lru_key = min(self._access_times, key=self._access_times.get)
        
        # Remove it
        del self._cache[lru_key]
        del self._access_times[lru_key]


class FunctionCacheManager:
    """Manager for function caches."""
    
    def __init__(self):
        """Initialize the function cache manager."""
        self._caches: Dict[str, FunctionCache] = {}
    
    def get_cache(
        self,
        namespace: str = "default",
        max_size: int = 1000,
        ttl: Optional[int] = 3600
    ) -> FunctionCache:
        """
        Get a cache by namespace.
        
        Args:
            namespace: Namespace for the cache
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            FunctionCache: The cache
        """
        if namespace not in self._caches:
            self._caches[namespace] = FunctionCache(max_size, ttl, namespace)
        return self._caches[namespace]
    
    def clear_cache(self, namespace: str = "default") -> None:
        """
        Clear a cache by namespace.
        
        Args:
            namespace: Namespace for the cache
        """
        if namespace in self._caches:
            self._caches[namespace].clear()
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()


# Create a singleton instance
cache_manager = FunctionCacheManager()


def cached(
    ttl: Optional[int] = 3600,
    max_size: int = 1000,
    namespace: Optional[str] = None,
    key_generator: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching function calls.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        max_size: Maximum number of items in the cache
        namespace: Namespace for the cache (defaults to function name)
        key_generator: Function to generate cache keys
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        # Get function name and namespace
        func_name = func.__name__
        cache_namespace = namespace or func_name
        
        # Get cache
        cache = cache_manager.get_cache(cache_namespace, max_size, ttl)
        
        # Define key generator if not provided
        def default_key_generator(*args, **kwargs):
            # Convert args and kwargs to a string
            key_parts = [func_name]
            
            # Add args
            for arg in args:
                key_parts.append(str(arg))
            
            # Add kwargs (sorted for consistency)
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}={kwargs[k]}")
            
            # Join and hash
            key_str = ":".join(key_parts)
            return hashlib.md5(key_str.encode()).hexdigest()
        
        # Use provided key generator or default
        key_gen = key_generator or default_key_generator
        
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key_gen(*args, **kwargs)
                
                # Check cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func_name} with key {cache_key}")
                    return cached_value
                
                # Call function
                logger.debug(f"Cache miss for {func_name} with key {cache_key}")
                result = await func(*args, **kwargs)
                
                # Cache result
                cache.set(cache_key, result)
                
                return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key_gen(*args, **kwargs)
                
                # Check cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func_name} with key {cache_key}")
                    return cached_value
                
                # Call function
                logger.debug(f"Cache miss for {func_name} with key {cache_key}")
                result = func(*args, **kwargs)
                
                # Cache result
                cache.set(cache_key, result)
                
                return result
            
            return wrapper
    
    return decorator


def clear_cache(namespace: str = "default") -> None:
    """
    Clear a cache by namespace.
    
    Args:
        namespace: Namespace for the cache
    """
    cache_manager.clear_cache(namespace)


def clear_all_caches() -> None:
    """Clear all caches."""
    cache_manager.clear_all_caches()
