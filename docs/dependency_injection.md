# Dependency Injection

The Dependency Injection system in Saplings provides a flexible and powerful way to manage service dependencies, enabling loose coupling, testability, and configurability.

## Overview

The Dependency Injection system consists of several key components:

- **Container**: Central registry for services and their dependencies
- **Register Decorator**: Decorator for registering services with the container
- **Inject Decorator**: Decorator for injecting dependencies into functions
- **Lifecycle Scopes**: Different scopes for service instances (singleton, scoped, transient)
- **Providers**: Factory functions for creating service instances

This system enables the creation of modular, testable, and maintainable code by decoupling service implementations from their consumers.

## Core Concepts

### Dependency Inversion

Dependency Inversion is a principle that states:

1. High-level modules should not depend on low-level modules. Both should depend on abstractions.
2. Abstractions should not depend on details. Details should depend on abstractions.

In Saplings, this is achieved by defining interfaces (abstract classes or protocols) that high-level modules depend on, while low-level modules implement these interfaces.

### Inversion of Control

Inversion of Control (IoC) is a design principle where the control flow of a program is inverted: instead of the caller controlling the flow, the callee controls it. In the context of dependency injection, this means that instead of a class creating its dependencies, they are provided to it.

### Service Locator vs. Constructor Injection

Saplings supports both service locator and constructor injection patterns:

- **Service Locator**: Services are resolved from a central registry (the container)
- **Constructor Injection**: Dependencies are provided through constructors

While constructor injection is generally preferred for its explicitness, the service locator pattern can be useful in certain scenarios.

### Lifecycle Scopes

Saplings supports different lifecycle scopes for services:

- **Singleton**: One instance per container
- **Scoped**: One instance per scope (e.g., per request)
- **Transient**: New instance each time

These scopes allow for fine-grained control over service lifetimes.

## API Reference

### Container

```python
class SaplingsContainer:
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the container."""

    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        scope: LifecycleScope = LifecycleScope.SINGLETON,
        **kwargs,
    ) -> None:
        """Register a service with the container."""

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container."""

    def create_scope(self, parent_scope: Optional[Scope] = None) -> Scope:
        """Create a new dependency injection scope."""

    def enter_scope(self, scope: Optional[Scope] = None) -> Scope:
        """Enter a dependency injection scope."""

    def exit_scope(self) -> None:
        """Exit the current scope."""

    def get_service(
        self,
        service_provider: Callable[[], T],
        scope: LifecycleScope = LifecycleScope.SINGLETON,
    ) -> T:
        """Get a service from the container."""

    def dispose(self) -> None:
        """Dispose of all resources."""
```

### Register Decorator

```python
def register(cls: type[T], /, singleton: bool = True) -> Callable[[type[T]], type[T]]:
    """
    Class decorator to auto-register concrete types.

    Args:
        cls: The interface/abstract type to register
        singleton: Whether to register as singleton (default) or transient

    Returns:
        Decorator function that registers the concrete class
    """
```

### Inject Decorator

```python
def inject(func: Callable) -> Callable:
    """
    Decorator to auto-resolve constructor args.

    Args:
        func: The function to inject dependencies into

    Returns:
        Wrapped function with auto-resolved dependencies
    """
```

### Scope

```python
class Scope:
    def __init__(self, container: SaplingsContainer, parent_scope: Optional["Scope"] = None):
        """Initialize a scope."""

    def get(self, service_name: str) -> Optional[Any]:
        """Get a service from the scope."""

    def get_or_create(self, service_name: str, factory: Callable[[], Any]) -> Any:
        """Get an existing instance or create a new one."""

    def dispose(self) -> None:
        """Dispose of all resources in the scope."""
```

### LifecycleScope

```python
class LifecycleScope(str, Enum):
    """Lifecycle scopes for services."""

    SINGLETON = "singleton"  # One instance per container
    SCOPED = "scoped"  # One instance per scope
    TRANSIENT = "transient"  # New instance each time
```

### Provider

```python
class Provider(Generic[T]):
    """Base provider interface for dependencies."""

    def provide(self) -> T:
        """Provide an instance of the dependency."""

class FactoryProvider(Provider[T]):
    """Provider that uses a factory function to create instances."""

    def __init__(self, factory: Callable[..., T], **kwargs):
        """Initialize with a factory function and its arguments."""

    def provide(self) -> T:
        """Create an instance using the factory."""

class SingletonProvider(Provider[T]):
    """Provider that ensures only one instance is created."""

    def __init__(self, provider: Provider[T]):
        """Initialize with an underlying provider."""

    def provide(self) -> T:
        """Get or create the singleton instance."""

class ConfiguredProvider(Provider[T]):
    """Provider that configures an instance after creation."""

    def __init__(self, provider: Provider[T], configurator: Callable[[T], None]):
        """Initialize with a provider and a configurator."""

    def provide(self) -> T:
        """Create and configure an instance."""

class LazyProvider(Provider[T]):
    """Provider that defers creation until first access."""

    def __init__(self, provider: Provider[T]):
        """Initialize with an underlying provider."""

    def provide(self) -> T:
        """Get or create the instance on first access."""
```

## Usage Examples

### Basic Usage

```python
from saplings.di import container, register, inject
from typing import Protocol

# Define a service interface
class DataService(Protocol):
    def get_data(self) -> str:
        ...

# Implement the service
@register(DataService)
class FileDataService:
    def __init__(self, file_path: str = "data.txt"):
        self.file_path = file_path

    def get_data(self) -> str:
        with open(self.file_path, "r") as f:
            return f.read()

# Use the service with constructor injection
class DataProcessor:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def process(self) -> str:
        data = self.data_service.get_data()
        return data.upper()

# Use the inject decorator for automatic dependency resolution
@inject
def process_data(data_processor: DataProcessor) -> str:
    return data_processor.process()

# Register the processor with the container
container.register(
    DataProcessor,
    factory=lambda ds: DataProcessor(data_service=ds),
    ds=DataService
)

# Use the processor
result = process_data()
print(result)
```

### Service Registration

```python
from saplings.di import container
from typing import Protocol

# Define service interfaces
class Logger(Protocol):
    def log(self, message: str) -> None:
        ...

class DataStore(Protocol):
    def save(self, data: str) -> None:
        ...

# Implement services
class ConsoleLogger:
    def log(self, message: str) -> None:
        print(f"LOG: {message}")

class FileDataStore:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, data: str) -> None:
        with open(self.file_path, "w") as f:
            f.write(data)

# Register services with the container
container.register(Logger, factory=lambda: ConsoleLogger())
container.register(
    DataStore,
    factory=lambda: FileDataStore(file_path="data.txt")
)

# Use services
logger = container.resolve(Logger)
data_store = container.resolve(DataStore)

logger.log("Hello, world!")
data_store.save("Hello, world!")
```

### Lifecycle Scopes

```python
from saplings.di import container
from saplings.container import LifecycleScope
from typing import Protocol

# Define a service interface
class Counter(Protocol):
    def increment(self) -> int:
        ...

# Implement the service
class SimpleCounter:
    def __init__(self):
        self.count = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

# Register the service with different scopes
container.register(
    Counter,
    factory=lambda: SimpleCounter(),
    scope=LifecycleScope.SINGLETON
)

# Use the singleton service
counter1 = container.resolve(Counter)
counter2 = container.resolve(Counter)

print(counter1.increment())  # 1
print(counter2.increment())  # 2 (same instance)

# Register a transient service
container.register(
    "TransientCounter",
    factory=lambda: SimpleCounter(),
    scope=LifecycleScope.TRANSIENT
)

# Use the transient service
counter3 = container.resolve("TransientCounter")
counter4 = container.resolve("TransientCounter")

print(counter3.increment())  # 1
print(counter4.increment())  # 1 (different instance)
```

### Scoped Services

```python
from saplings.di import container
from saplings.container import LifecycleScope
from typing import Protocol

# Define a service interface
class RequestContext(Protocol):
    def get_user_id(self) -> str:
        ...

# Implement the service
class SimpleRequestContext:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def get_user_id(self) -> str:
        return self.user_id

# Register the service as scoped
container.register(
    RequestContext,
    factory=lambda: SimpleRequestContext(user_id="user1"),
    scope=LifecycleScope.SCOPED
)

# Create a scope
scope = container.enter_scope()

# Use the scoped service
context1 = container.resolve(RequestContext)
context2 = container.resolve(RequestContext)

print(context1.get_user_id())  # user1
print(context2.get_user_id())  # user1 (same instance within the scope)

# Exit the scope
container.exit_scope()

# Create a new scope
scope = container.enter_scope()

# Register a different context in the new scope
container.register(
    RequestContext,
    factory=lambda: SimpleRequestContext(user_id="user2"),
    scope=LifecycleScope.SCOPED
)

# Use the scoped service in the new scope
context3 = container.resolve(RequestContext)
print(context3.get_user_id())  # user2 (different instance in the new scope)

# Exit the scope
container.exit_scope()
```

### Integration with Agent

```python
from saplings import Agent, AgentConfig
from saplings.di import container, inject
from saplings.core.interfaces import IModelService, IMemoryManager

# Configure the container
container.register(
    AgentConfig,
    factory=lambda: AgentConfig(
        provider="openai",
        model_name="gpt-4o",
    )
)

# Use the inject decorator to get dependencies
@inject
def create_agent(config: AgentConfig, model_service: IModelService, memory_manager: IMemoryManager) -> Agent:
    return Agent(
        config=config,
        model_service=model_service,
        memory_manager=memory_manager,
    )

# Create an agent with injected dependencies
agent = create_agent()

# Use the agent
import asyncio
result = asyncio.run(agent.run("Explain the concept of dependency injection."))
print(result)
```

## Advanced Features

### Factory Registration

```python
from saplings.di import container
from typing import Protocol, List

# Define service interfaces
class DataSource(Protocol):
    def get_data(self) -> List[str]:
        ...

class DataProcessor(Protocol):
    def process(self, data: List[str]) -> List[str]:
        ...

class DataReporter(Protocol):
    def report(self, data: List[str]) -> None:
        ...

# Implement services
class FileDataSource:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self) -> List[str]:
        with open(self.file_path, "r") as f:
            return f.readlines()

class UppercaseProcessor:
    def process(self, data: List[str]) -> List[str]:
        return [line.upper() for line in data]

class ConsoleReporter:
    def report(self, data: List[str]) -> None:
        for line in data:
            print(f"REPORT: {line}")

# Register services with factories
container.register(
    DataSource,
    factory=lambda: FileDataSource(file_path="data.txt")
)

container.register(
    DataProcessor,
    factory=lambda: UppercaseProcessor()
)

container.register(
    DataReporter,
    factory=lambda: ConsoleReporter()
)

# Register a service that depends on other services
container.register(
    "DataManager",
    factory=lambda src, proc, rep: {
        "source": src,
        "processor": proc,
        "reporter": rep,
    },
    src=DataSource,
    proc=DataProcessor,
    rep=DataReporter
)

# Use the service
data_manager = container.resolve("DataManager")
data = data_manager["source"].get_data()
processed_data = data_manager["processor"].process(data)
data_manager["reporter"].report(processed_data)
```

### Custom Providers

```python
from saplings.di.providers import Provider, FactoryProvider, SingletonProvider, LazyProvider
from typing import Protocol

# Define a service interface
class ExpensiveService(Protocol):
    def perform_operation(self) -> str:
        ...

# Implement the service
class RealExpensiveService:
    def __init__(self):
        print("Initializing expensive service...")
        # Simulate expensive initialization
        import time
        time.sleep(1)

    def perform_operation(self) -> str:
        return "Operation completed"

# Create a factory provider
factory_provider = FactoryProvider(lambda: RealExpensiveService())

# Create a singleton provider
singleton_provider = SingletonProvider(factory_provider)

# Create a lazy provider
lazy_provider = LazyProvider(singleton_provider)

# Use the provider
print("Before accessing the service")
service = lazy_provider.provide()  # Service is created here
print("After accessing the service")
result = service.perform_operation()
print(result)

# Access the service again
print("Before accessing the service again")
service2 = lazy_provider.provide()  # Service is reused
print("After accessing the service again")
result2 = service2.perform_operation()
print(result2)
```

### Circular Dependency Detection

```python
from saplings.di import container
from typing import Protocol

# Define service interfaces with circular dependencies
class ServiceA(Protocol):
    def get_b(self) -> "ServiceB":
        ...

class ServiceB(Protocol):
    def get_a(self) -> ServiceA:
        ...

# Implement services
class RealServiceA:
    def __init__(self, service_b_factory):
        self.service_b_factory = service_b_factory

    def get_b(self) -> "ServiceB":
        return self.service_b_factory()

class RealServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a

    def get_a(self) -> ServiceA:
        return self.service_a

# Register services with circular dependency resolution
container.register(
    ServiceA,
    factory=lambda: RealServiceA(service_b_factory=lambda: container.resolve(ServiceB))
)

try:
    # This will cause a circular dependency error
    container.register(
        ServiceB,
        factory=lambda sa: RealServiceB(service_a=sa),
        sa=ServiceA
    )

    # Try to resolve ServiceA
    service_a = container.resolve(ServiceA)
except Exception as e:
    print(f"Circular dependency detected: {e}")
```

## Implementation Details

### Container Implementation

The `SaplingsContainer` class is the central component of the dependency injection system:

1. **Service Registration**: Services are registered with a type, factory function, and scope
2. **Service Resolution**: Services are resolved by type, with dependencies automatically injected
3. **Scope Management**: Scopes are created, entered, and exited to manage service lifetimes
4. **Circular Dependency Detection**: Circular dependencies are detected and reported
5. **Lifecycle Management**: Services are properly disposed when no longer needed

### Decorator Implementation

The `register` and `inject` decorators provide a convenient way to use the container:

1. **Register Decorator**: Registers a class with the container as an implementation of an interface
2. **Inject Decorator**: Automatically resolves dependencies for a function based on type annotations

### Provider Implementation

Providers are factory functions that create service instances:

1. **FactoryProvider**: Creates instances using a factory function
2. **SingletonProvider**: Ensures only one instance is created
3. **ConfiguredProvider**: Configures instances after creation
4. **LazyProvider**: Defers creation until first access

### Scope Implementation

Scopes manage the lifetime of service instances:

1. **Scope Creation**: Scopes are created with an optional parent scope
2. **Instance Management**: Instances are stored in the scope
3. **Hierarchical Resolution**: Instances are resolved from the current scope or parent scopes
4. **Disposal**: Instances are properly disposed when the scope is exited

## Extension Points

The Dependency Injection system is designed to be extensible:

### Custom Container

You can create a custom container by extending the `SaplingsContainer` class:

```python
from saplings.container import SaplingsContainer, LifecycleScope
from typing import Type, TypeVar, Callable, Any, Optional

T = TypeVar("T")

class CustomContainer(SaplingsContainer):
    def __init__(self, config=None):
        super().__init__(config)
        self._interceptors = {}

    def register_interceptor(self, service_type: Type[T], interceptor: Callable[[T], T]) -> None:
        """Register an interceptor for a service type."""
        self._interceptors[service_type] = interceptor

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service with interception."""
        # Resolve the service using the parent method
        service = super().resolve(service_type)

        # Apply interceptor if registered
        if service_type in self._interceptors:
            service = self._interceptors[service_type](service)

        return service
```

### Custom Provider

You can create a custom provider by implementing the `Provider` interface:

```python
from saplings.di.providers import Provider
from typing import TypeVar, Generic, Callable, Dict, Any

T = TypeVar("T")

class CachedProvider(Provider[T]):
    """Provider that caches instances based on a key."""

    def __init__(self, factory: Callable[[Any], T], key_selector: Callable[[Any], str]):
        """Initialize with a factory and key selector."""
        self._factory = factory
        self._key_selector = key_selector
        self._cache: Dict[str, T] = {}

    def provide(self, context: Any) -> T:
        """Get or create an instance based on the context."""
        key = self._key_selector(context)
        if key not in self._cache:
            self._cache[key] = self._factory(context)
        return self._cache[key]
```

### Custom Scope

You can create a custom scope by extending the `Scope` class:

```python
from saplings.container import Scope, SaplingsContainer
from typing import Optional, Any, Callable, Dict

class TimedScope(Scope):
    """Scope that automatically disposes after a timeout."""

    def __init__(self, container: SaplingsContainer, parent_scope: Optional["Scope"] = None, timeout_seconds: float = 300):
        """Initialize with a timeout."""
        super().__init__(container, parent_scope)
        self.timeout_seconds = timeout_seconds
        self.created_at = time.time()

    def get(self, service_name: str) -> Optional[Any]:
        """Get a service, checking for timeout."""
        if time.time() - self.created_at > self.timeout_seconds:
            self.dispose()
            raise ValueError(f"Scope has expired after {self.timeout_seconds} seconds")
        return super().get(service_name)

    def get_or_create(self, service_name: str, factory: Callable[[], Any]) -> Any:
        """Get or create a service, checking for timeout."""
        if time.time() - self.created_at > self.timeout_seconds:
            self.dispose()
            raise ValueError(f"Scope has expired after {self.timeout_seconds} seconds")
        return super().get_or_create(service_name, factory)
```

## Conclusion

The Dependency Injection system in Saplings provides a flexible and powerful way to manage service dependencies, enabling loose coupling, testability, and configurability. By using interfaces, factories, and scopes, it enables the creation of modular, testable, and maintainable code.
