# Hot-Loading System

Saplings includes a powerful hot-loading system that allows tools to be dynamically loaded, updated, and unloaded without restarting the application.

## Overview

The hot-loading system consists of several components:

1. **HotLoader**: Manages the loading, unloading, and reloading of tools
2. **ToolLifecycleManager**: Handles the initialization, update, and retirement of tools
3. **IntegrationManager**: Integrates the hot-loading system with other components
4. **EventSystem**: Provides event-based communication between components

## HotLoader

The `HotLoader` class is the main entry point for the hot-loading system:

```python
from saplings import HotLoader, HotLoaderConfig

# Create a hot loader
config = HotLoaderConfig(
    watch_directories=["tools"],
    auto_reload=True,
    reload_interval=5.0,
)
hot_loader = HotLoader(config=config)

# Start auto-reload
hot_loader.start_auto_reload()
```

## Loading and Unloading Tools

You can manually load and unload tools:

```python
from saplings import ToolPlugin

# Load a tool
class MyTool(ToolPlugin):
    name = "My Tool"
    description = "A custom tool"
    plugin_type = "tool"
    
    def execute(self, *args, **kwargs):
        return "Hello, world!"

hot_loader.load_tool(MyTool)

# Unload a tool
hot_loader.unload_tool("my_tool")
```

## Tool Lifecycle Management

The `ToolLifecycleManager` handles the lifecycle of tools:

```python
# Get the lifecycle manager
lifecycle_manager = hot_loader.lifecycle_manager

# Initialize a tool
lifecycle_manager.initialize_tool(MyTool)

# Update a tool
lifecycle_manager.update_tool(MyTool)

# Retire a tool
lifecycle_manager.retire_tool("my_tool")
```

## Integration with Other Components

The `IntegrationManager` integrates the hot-loading system with other components:

```python
from saplings import (
    IntegrationManager,
    Executor,
    ExecutorConfig,
    SequentialPlanner,
    PlannerConfig,
    GraphRunner,
    GraphRunnerConfig,
)

# Create components
executor = Executor(model=model, config=ExecutorConfig())
planner = SequentialPlanner(model=model, config=PlannerConfig())
graph_runner = GraphRunner(model=model, config=GraphRunnerConfig())

# Create an integration manager
integration_manager = IntegrationManager(
    executor=executor,
    planner=planner,
    graph_runner=graph_runner,
    hot_loader=hot_loader,
)

# Start the integration manager
integration_manager.start()
```

## Event System

The `EventSystem` provides event-based communication between components:

```python
from saplings import EventSystem, EventType, Event, EventListener

# Get the event system
event_system = EventSystem()

# Add a listener
def on_tool_loaded(event):
    print(f"Tool loaded: {event.data.get('tool_id')}")

event_system.add_listener(EventType.TOOL_LOADED, on_tool_loaded)

# Emit an event
event = Event(
    type=EventType.TOOL_LOADED,
    source="my_component",
    data={"tool_id": "my_tool"},
)
event_system.emit(event)
```

## Auto-Reload

The hot-loading system can automatically reload tools when they change:

```python
# Configure auto-reload
config = HotLoaderConfig(
    watch_directories=["tools"],
    auto_reload=True,
    reload_interval=5.0,
)
hot_loader = HotLoader(config=config)

# Start auto-reload
hot_loader.start_auto_reload()

# Stop auto-reload
hot_loader.stop_auto_reload()
```

## Integration with Tool Factory

The hot-loading system integrates seamlessly with the Tool Factory:

```python
from saplings import (
    ToolFactory,
    ToolFactoryConfig,
    ToolTemplate,
    ToolSpecification,
    SecurityLevel,
)

# Create a tool factory
tool_factory = ToolFactory(model=model, config=ToolFactoryConfig())

# Register a template
template = ToolTemplate(
    id="math_tool",
    name="Math Tool",
    description="A tool for mathematical operations",
    template_code="""
def {{function_name}}({{parameters}}):
    \"\"\"{{description}}\"\"\"
    {{code_body}}
""",
    required_parameters=["function_name", "parameters", "description", "code_body"],
)
tool_factory.register_template(template)

# Create a tool specification
spec = ToolSpecification(
    id="add_numbers",
    name="Add Numbers",
    description="A tool to add two numbers",
    template_id="math_tool",
    parameters={
        "function_name": "add_numbers",
        "parameters": "a: int, b: int",
        "description": "Add two numbers together",
        "code_body": "return a + b",
    },
)

# Create the tool
tool_class = await tool_factory.create_tool(spec)

# Load the tool
hot_loader.load_tool(tool_class)
```

## Advanced Usage

### Custom Tool Discovery

You can customize how tools are discovered:

```python
# Configure custom tool discovery
config = HotLoaderConfig(
    tool_discovery_method="directory",
    watch_directories=["tools"],
)
hot_loader = HotLoader(config=config)

# Scan a directory for tools
python_files = hot_loader.scan_directory("tools")
for file_path in python_files:
    module = hot_loader.load_module_from_file(file_path)
    # Process the module
```

### Callbacks

You can register callbacks for tool loading and unloading:

```python
# Configure callbacks
config = HotLoaderConfig(
    on_tool_load_callback=lambda tool_class: print(f"Loaded: {tool_class.name}"),
    on_tool_unload_callback=lambda tool_id: print(f"Unloaded: {tool_id}"),
)
hot_loader = HotLoader(config=config)
```

### Custom Event Listeners

You can create custom event listeners:

```python
# Create an async event listener
async def on_tool_executed(event):
    tool_id = event.data.get("tool_id")
    result = event.data.get("result")
    print(f"Tool {tool_id} executed with result: {result}")

# Register the listener
event_system.add_listener(
    EventType.TOOL_EXECUTED,
    on_tool_executed,
    is_async=True,
)
```

## Conclusion

The hot-loading system in Saplings provides a powerful way to dynamically load, update, and unload tools without restarting the application. By integrating with other components, it enables a flexible and extensible architecture for building agent-based systems.
