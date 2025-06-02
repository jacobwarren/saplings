# CLI Interface Guide

This guide covers the Saplings Command Line Interface (CLI), providing comprehensive documentation for all commands, configuration options, and usage patterns.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Basic Commands](#basic-commands)
- [Agent Management](#agent-management)
- [Configuration Management](#configuration-management)
- [Tool Operations](#tool-operations)
- [Memory & Document Management](#memory--document-management)
- [Monitoring & Debugging](#monitoring--debugging)
- [Scripting & Automation](#scripting--automation)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation & Setup

### Installing the CLI

```bash
# Install Saplings with CLI support
pip install saplings[cli]

# Verify installation
saplings --version
saplings --help
```

### Initial Configuration

```bash
# Initialize Saplings configuration
saplings init

# Set default provider
saplings config set provider openai
saplings config set api_key "your-openai-api-key"

# Verify configuration
saplings config show
```

### Environment Setup

```bash
# Create a new Saplings workspace
saplings workspace create my-project
cd my-project

# Initialize workspace with templates
saplings workspace init --template standard

# Show workspace status
saplings workspace status
```

## Basic Commands

### Core Command Structure

```bash
saplings [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS] [ARGS]
```

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Configuration file path | `~/.saplings/config.yaml` |
| `--workspace` | Workspace directory | Current directory |
| `--verbose` | Enable verbose output | `false` |
| `--quiet` | Suppress non-error output | `false` |
| `--format` | Output format (json, yaml, table) | `table` |
| `--no-color` | Disable colored output | `false` |

### Quick Start Commands

```bash
# Run a simple query
saplings run "What is machine learning?"

# Run with specific model
saplings run --model gpt-4o "Explain quantum computing"

# Run with tools enabled
saplings run --tools python,web "Calculate the fibonacci sequence"

# Interactive mode
saplings chat

# Exit interactive mode
> exit
```

## Agent Management

### Creating Agents

```bash
# Create a new agent with default settings
saplings agent create my-agent

# Create with specific configuration
saplings agent create research-agent \
  --provider openai \
  --model gpt-4o \
  --tools web,python \
  --memory-path ./research-memory

# Create from template
saplings agent create data-analyst --template analytics

# Create with custom configuration file
saplings agent create custom-agent --config agent-config.yaml
```

### Agent Configuration Templates

```yaml
# agent-config.yaml
name: "research-assistant"
provider: "openai"
model: "gpt-4o"
tools:
  - "PythonInterpreterTool"
  - "WebSearchTool"
  - "FileReadTool"
memory:
  path: "./agent-memory"
  max_documents: 1000
gasa:
  enabled: true
  strategy: "binary"
  max_hops: 2
monitoring:
  enabled: true
  tracing_backend: "console"
```

### Managing Agents

```bash
# List all agents
saplings agent list

# Show agent details
saplings agent show my-agent

# Update agent configuration
saplings agent update my-agent --model gpt-4o-mini
saplings agent update my-agent --add-tool WebSearchTool

# Clone an existing agent
saplings agent clone my-agent new-agent

# Delete agent
saplings agent delete old-agent

# Export agent configuration
saplings agent export my-agent > my-agent-config.yaml

# Import agent configuration
saplings agent import new-agent-config.yaml
```

### Agent Presets

```bash
# List available presets
saplings agent presets

# Create agent from preset
saplings agent create writer --preset creative-writing
saplings agent create analyst --preset data-analysis
saplings agent create researcher --preset research-assistant

# Show preset details
saplings agent preset show data-analysis
```

## Configuration Management

### Global Configuration

```bash
# Show all configuration
saplings config show

# Show specific configuration section
saplings config show providers
saplings config show models

# Set configuration values
saplings config set provider anthropic
saplings config set models.default claude-3-sonnet
saplings config set tools.python.timeout 30

# Unset configuration values
saplings config unset models.fallback

# Reset configuration to defaults
saplings config reset

# Edit configuration in default editor
saplings config edit
```

### Provider Configuration

```bash
# Configure OpenAI
saplings config set providers.openai.api_key "sk-..."
saplings config set providers.openai.base_url "https://api.openai.com/v1"

# Configure Anthropic
saplings config set providers.anthropic.api_key "sk-ant-..."

# Configure local providers
saplings config set providers.vllm.base_url "http://localhost:8000"
saplings config set providers.huggingface.token "hf_..."

# Test provider configuration
saplings config test-provider openai
saplings config test-provider anthropic
```

### Environment-Specific Configuration

```bash
# Set environment-specific configuration
saplings config set-env development providers.openai.api_key "dev-key"
saplings config set-env production providers.openai.api_key "prod-key"

# Switch between environments
saplings config use-env development
saplings config use-env production

# Show current environment
saplings config current-env

# List all environments
saplings config list-envs
```

## Tool Operations

### Tool Management

```bash
# List available tools
saplings tools list

# Show tool details
saplings tools show PythonInterpreterTool

# Search for tools
saplings tools search "web"
saplings tools search --category "data-analysis"

# Install tool from registry
saplings tools install WebSearchTool

# Install custom tool
saplings tools install ./my-custom-tool.py

# Uninstall tool
saplings tools uninstall CustomTool

# Update all tools
saplings tools update

# Update specific tool
saplings tools update PythonInterpreterTool
```

### Tool Development

```bash
# Create new tool from template
saplings tools create MyCustomTool --template basic
saplings tools create DataAnalyzer --template advanced

# Validate tool
saplings tools validate ./my-tool.py

# Test tool
saplings tools test MyCustomTool --input "test data"

# Package tool for distribution
saplings tools package MyCustomTool

# Publish tool to registry
saplings tools publish MyCustomTool --registry public
```

### Tool Factory

```bash
# Enable tool factory
saplings tools factory enable

# Configure tool factory security
saplings tools factory config --security-level high
saplings tools factory config --sandbox-enabled true

# List factory-generated tools
saplings tools factory list

# Clear factory cache
saplings tools factory clear-cache

# Show factory statistics
saplings tools factory stats
```

## Memory & Document Management

### Document Operations

```bash
# Add documents to agent memory
saplings memory add my-agent document.txt
saplings memory add my-agent --recursive ./documents/

# Add with metadata
saplings memory add my-agent file.pdf --metadata '{"type": "report", "year": 2024}'

# List documents in memory
saplings memory list my-agent

# Search documents
saplings memory search my-agent "machine learning"
saplings memory search my-agent --metadata-filter "type=report"

# Show document details
saplings memory show my-agent doc-id-123

# Remove documents
saplings memory remove my-agent doc-id-123
saplings memory remove my-agent --filter "type=draft"

# Clear all memory
saplings memory clear my-agent

# Export memory
saplings memory export my-agent memory-backup.json

# Import memory
saplings memory import my-agent memory-backup.json
```

### Memory Statistics

```bash
# Show memory statistics
saplings memory stats my-agent

# Show detailed breakdown
saplings memory stats my-agent --detailed

# Memory usage by type
saplings memory usage my-agent --by-type

# Memory health check
saplings memory health-check my-agent
```

### Document Processing

```bash
# Process documents with chunking
saplings memory process my-agent large-file.pdf --chunk-size 1000

# Process with custom embeddings
saplings memory process my-agent documents/ --embedding-model text-embedding-3-large

# Batch processing
saplings memory batch-process my-agent ./documents/ \
  --parallel 4 \
  --chunk-size 1500 \
  --progress

# Reprocess existing documents
saplings memory reprocess my-agent --filter "processed_before=2024-01-01"
```

## Monitoring & Debugging

### Execution Monitoring

```bash
# Run with monitoring enabled
saplings run "Analyze data" --monitor --trace

# Show execution traces
saplings traces list

# Show specific trace
saplings traces show trace-id-123

# Export traces
saplings traces export --output traces.json
saplings traces export trace-id-123 --format html

# Clear old traces
saplings traces clear --older-than 7d
```

### Performance Analysis

```bash
# Run performance analysis
saplings perf analyze my-agent "Complex analysis task"

# Show performance statistics
saplings perf stats my-agent

# Performance profiling
saplings perf profile my-agent --duration 60s

# Generate performance report
saplings perf report my-agent --output performance-report.html

# Benchmark agent performance
saplings perf benchmark my-agent benchmark-tasks.yaml
```

### Debugging

```bash
# Run in debug mode
saplings run "Debug this task" --debug --verbose

# Enable specific debug categories
saplings run "Task" --debug-categories gasa,memory,tools

# Show debug logs
saplings debug logs

# Show recent errors
saplings debug errors --last 10

# Debug agent configuration
saplings debug config my-agent

# Test agent connectivity
saplings debug test-connection my-agent

# Validate agent setup
saplings debug validate my-agent
```

### Health Monitoring

```bash
# Check system health
saplings health check

# Check specific agent health
saplings health check-agent my-agent

# Show health report
saplings health report

# Monitor continuously
saplings health monitor --interval 30s

# Health check with notifications
saplings health monitor --webhook http://localhost:8080/alerts
```

## Scripting & Automation

### Batch Operations

```bash
# Run batch commands from file
saplings batch run commands.txt

# Run with parallel execution
saplings batch run commands.txt --parallel 4

# Batch with error handling
saplings batch run commands.txt --continue-on-error
```

### Task Automation

```bash
# Create automation script
saplings script create data-pipeline --template etl

# Run automation script
saplings script run data-pipeline

# Schedule recurring tasks
saplings schedule create daily-report \
  --command "saplings run 'Generate daily report'" \
  --cron "0 9 * * *"

# List scheduled tasks
saplings schedule list

# Remove scheduled task
saplings schedule remove daily-report
```

### Pipeline Operations

```bash
# Create processing pipeline
saplings pipeline create analysis-pipeline

# Add pipeline steps
saplings pipeline add-step analysis-pipeline \
  --name "data-extraction" \
  --command "saplings run 'Extract data from documents'"

saplings pipeline add-step analysis-pipeline \
  --name "analysis" \
  --command "saplings run 'Analyze extracted data'"

# Run pipeline
saplings pipeline run analysis-pipeline

# Show pipeline status
saplings pipeline status analysis-pipeline

# Export pipeline definition
saplings pipeline export analysis-pipeline > pipeline.yaml
```

## Advanced Usage

### Multi-Agent Orchestration

```bash
# Create agent group
saplings group create research-team

# Add agents to group
saplings group add-agent research-team researcher
saplings group add-agent research-team analyst

# Run coordinated task
saplings group run research-team "Collaborative analysis task"

# Show group status
saplings group status research-team

# Remove agent from group
saplings group remove-agent research-team analyst
```

### Workflow Management

```bash
# Create workflow
saplings workflow create document-analysis

# Define workflow steps
saplings workflow define document-analysis workflow.yaml

# Run workflow
saplings workflow run document-analysis --input documents/

# Show workflow progress
saplings workflow status document-analysis

# Cancel running workflow
saplings workflow cancel workflow-id-123

# List all workflows
saplings workflow list

# Show workflow history
saplings workflow history document-analysis
```

### Custom Commands

```bash
# Create custom command
saplings command create summarize-docs \
  --script "saplings run 'Summarize these documents: {input}'"

# Run custom command
saplings summarize-docs ./documents/

# List custom commands
saplings command list

# Edit custom command
saplings command edit summarize-docs

# Remove custom command
saplings command remove summarize-docs
```

### Integration & Plugins

```bash
# List available plugins
saplings plugins list

# Install plugin
saplings plugins install langchain-integration
saplings plugins install ./custom-plugin/

# Enable plugin
saplings plugins enable langchain-integration

# Configure plugin
saplings plugins config langchain-integration --setting value

# Disable plugin
saplings plugins disable langchain-integration

# Uninstall plugin
saplings plugins uninstall langchain-integration
```

## Command Reference

### Complete Command List

```bash
# Core commands
saplings run <task>                    # Run a task
saplings chat                          # Interactive chat mode
saplings init                          # Initialize configuration

# Agent management
saplings agent create <name>           # Create new agent
saplings agent list                    # List agents
saplings agent show <name>             # Show agent details
saplings agent update <name>           # Update agent
saplings agent delete <name>           # Delete agent
saplings agent clone <src> <dst>       # Clone agent
saplings agent export <name>           # Export agent config
saplings agent import <file>           # Import agent config

# Configuration
saplings config show                   # Show configuration
saplings config set <key> <value>     # Set configuration
saplings config unset <key>           # Unset configuration
saplings config reset                  # Reset to defaults
saplings config edit                   # Edit configuration

# Tools
saplings tools list                    # List available tools
saplings tools show <name>             # Show tool details
saplings tools install <name>          # Install tool
saplings tools uninstall <name>        # Uninstall tool
saplings tools create <name>           # Create new tool
saplings tools validate <file>         # Validate tool

# Memory
saplings memory add <agent> <file>     # Add document
saplings memory list <agent>           # List documents
saplings memory search <agent> <query> # Search documents
saplings memory remove <agent> <id>    # Remove document
saplings memory clear <agent>          # Clear memory
saplings memory stats <agent>          # Memory statistics

# Monitoring
saplings traces list                   # List traces
saplings traces show <id>              # Show trace details
saplings perf stats <agent>            # Performance stats
saplings health check                  # Health check
saplings debug logs                    # Show debug logs

# Workspace
saplings workspace create <name>       # Create workspace
saplings workspace init               # Initialize workspace
saplings workspace status             # Show workspace status

# Batch operations
saplings batch run <file>              # Run batch commands
saplings script create <name>          # Create automation script
saplings pipeline create <name>        # Create pipeline
```

### Output Formats

```bash
# JSON output
saplings agent list --format json

# YAML output
saplings config show --format yaml

# Table output (default)
saplings memory list my-agent --format table

# CSV output
saplings perf stats my-agent --format csv

# Custom format with templates
saplings agent list --format "{{.Name}}: {{.Model}}"
```

### Configuration File Examples

#### Global Configuration (`~/.saplings/config.yaml`)

```yaml
providers:
  openai:
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
  anthropic:
    api_key: "sk-ant-..."

models:
  default: "gpt-4o"
  fallback: "gpt-3.5-turbo"

tools:
  python:
    timeout: 30
    max_memory: "1GB"
  web:
    timeout: 10
    max_results: 10

monitoring:
  enabled: true
  tracing_backend: "console"
  trace_sampling_rate: 1.0

workspace:
  default_template: "standard"
  auto_init: true
```

#### Agent Configuration

```yaml
name: "research-assistant"
provider: "openai"
model: "gpt-4o"
tools:
  - "PythonInterpreterTool"
  - "WebSearchTool"
  - "FileReadTool"

memory:
  path: "./research-memory"
  max_documents: 1000
  chunk_size: 1000

gasa:
  enabled: true
  strategy: "binary"
  max_hops: 2
  threshold: 0.1

monitoring:
  enabled: true
  trace_sampling_rate: 0.1

model_parameters:
  temperature: 0.3
  max_tokens: 2000
  top_p: 0.9
```

## Troubleshooting

### Common Issues

#### Configuration Problems

```bash
# Check configuration validity
saplings config validate

# Reset configuration if corrupted
saplings config reset --force

# Check provider connectivity
saplings config test-provider openai
```

#### Agent Issues

```bash
# Validate agent configuration
saplings debug validate my-agent

# Check agent connectivity
saplings debug test-connection my-agent

# Restart agent with fresh configuration
saplings agent restart my-agent --clean
```

#### Memory Problems

```bash
# Check memory health
saplings memory health-check my-agent

# Rebuild memory index
saplings memory reindex my-agent

# Clear corrupted memory
saplings memory clear my-agent --force
```

#### Performance Issues

```bash
# Profile agent performance
saplings perf profile my-agent --duration 60s

# Check system resources
saplings health system

# Optimize agent configuration
saplings agent optimize my-agent --target-response-time 2s
```

### Debug Information

```bash
# Collect debug information
saplings debug collect

# Show system information
saplings debug system-info

# Show logs with filtering
saplings debug logs --level error --since 1h

# Export debug bundle
saplings debug export debug-info.zip
```

### Getting Help

```bash
# General help
saplings help

# Command-specific help
saplings help agent create
saplings agent create --help

# Show examples
saplings help examples

# Show configuration reference
saplings help config

# Show troubleshooting guide
saplings help troubleshooting
```

This comprehensive CLI guide provides users with complete documentation for using Saplings from the command line, enabling efficient agent management, configuration, and automation.