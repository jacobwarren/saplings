from __future__ import annotations

"""
Inject type for dependency injection.

This module provides the Inject type for explicitly marking dependencies
that should be injected by the container.
"""

import inspect
import logging
from typing import Any, Callable, Generic, Optional, Type, TypeVar, get_type_hints

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar("T")


class Inject(Generic[T]):
    """
    Explicit marker for dependencies that should be injected.

    This class is used to explicitly mark dependencies that should be
    injected by the container. It can be used in constructor parameters
    to make dependencies more explicit and to support optional dependencies.

    Example:
    -------
    ```python
    class Service:
        def __init__(self, required: IRequired, optional: Inject[IOptional] = None):
            self.required = required
            self.optional = optional
    ```

    """

    def __init__(self, value: Optional[T] = None):
        """
        Initialize the Inject marker.

        Args:
        ----
            value: The value to use if not injected

        """
        self.value = value

    @classmethod
    def extract_type(cls, annotation: Type) -> Optional[Type]:
        """
        Extract the type from an Inject annotation.

        Args:
        ----
            annotation: The type annotation to extract from

        Returns:
        -------
            The extracted type or None if not an Inject type

        """
        # Check if the annotation is an Inject type
        origin = getattr(annotation, "__origin__", None)
        if origin is Inject:
            # Extract the type argument
            args = getattr(annotation, "__args__", None)
            if args and len(args) == 1:
                return args[0]
        return None


def inject(func: Callable) -> Callable:
    """
    Decorator to auto-resolve constructor args.

    This decorator automatically resolves dependencies for a function
    based on its type annotations. It supports both regular type annotations
    and Inject[T] annotations for explicit dependency marking.

    Args:
    ----
        func: The function to inject dependencies into

    Returns:
    -------
        Wrapped function with auto-resolved dependencies

    """
    # Get the function's signature
    sig = inspect.signature(func)

    # Get type hints, handling forward references
    try:
        hints = get_type_hints(func)
    except (NameError, TypeError):
        # If we can't get type hints, use empty dict
        hints = {}

    def wrapper(*args, **kwargs):
        # Import here to avoid circular imports
        from saplings.di._internal.container import container

        # Create a mapping of parameter names to their values
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # Prepare the final kwargs
        final_kwargs = dict(bound_args.arguments)

        # Process parameters that aren't provided
        for name, _ in sig.parameters.items():
            if name in bound_args.arguments:
                # Check if the parameter is an Inject type with None value
                if isinstance(final_kwargs[name], Inject) and final_kwargs[name].value is None:
                    # Try to resolve the dependency
                    inject_type = Inject.extract_type(hints.get(name, Any))
                    if inject_type:
                        try:
                            final_kwargs[name] = container.resolve(inject_type)
                        except Exception as e:
                            logger.debug(f"Could not resolve optional dependency {name}: {e}")
            elif name in hints:
                # Check if the parameter is an Inject type
                inject_type = Inject.extract_type(hints.get(name, Any))
                if inject_type:
                    # Try to resolve the dependency
                    try:
                        final_kwargs[name] = container.resolve(inject_type)
                    except Exception as e:
                        logger.debug(f"Could not resolve optional dependency {name}: {e}")
                        # Use default value (None) for optional dependencies
                        final_kwargs[name] = None
                else:
                    # Regular type annotation, try to resolve
                    try:
                        final_kwargs[name] = container.resolve(hints[name])
                    except Exception as e:
                        logger.debug(f"Could not resolve dependency {name}: {e}")

        # Call the function with the injected dependencies
        return func(**final_kwargs)

    # Preserve the function's metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__

    return wrapper
