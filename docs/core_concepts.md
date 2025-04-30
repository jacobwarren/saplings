# Core Concepts

This document explains the core concepts of the Saplings framework.

## Agent Architecture

Saplings uses a modular agent architecture with the following components:

1. **Memory**: Stores and indexes your knowledge base
2. **Retrieval**: Finds relevant information from memory
3. **Planning**: Creates a plan to accomplish a task
4. **Execution**: Executes the plan and generates output
5. **JudgeAgent & Validator**: Evaluates and improves the output
6. **Self-Healing**: Automatically fixes errors and improves over time

## Memory System

The memory system in Saplings combines vector embeddings with a graph structure to create a rich representation of your knowledge base. This allows for more effective retrieval and reasoning.

## Retrieval

Saplings uses a cascaded, entropy-aware retrieval system that combines TF-IDF, embeddings, and graph expansion to find the most relevant information for a given query.

## Planning

The planning system creates a structured plan to accomplish a task, taking into account budget constraints and available tools.

## Execution

The execution system carries out the plan, generating output and handling errors. It includes verification strategies to ensure the output meets quality standards.

## JudgeAgent & Validator Framework

The JudgeAgent & Validator framework evaluates the output and provides feedback for improvement. This enables the self-improvement loop that makes Saplings unique.

## Self-Healing & Adaptation

The self-healing system automatically fixes errors and adapts over time through fine-tuning. This allows Saplings to become more robust and effective with use.

## Graph-Aligned Sparse Attention (GASA)

GASA is a technique that uses the graph structure of your knowledge base to create more efficient attention patterns in the underlying language model. This leads to better grounding and more efficient computation.

## Extensibility

Saplings is designed to be extensible, with support for plug-in adapters, indexers, validators, and dynamic tool synthesis.
