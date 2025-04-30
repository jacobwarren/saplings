# Saplings Documentation

Welcome to the Saplings documentation! This guide will help you understand and use the Saplings framework effectively.

## Table of Contents

- [Getting Started](./getting_started.md)
- [Core Concepts](./core_concepts.md)
- [Agent](./agent.md)
- [Memory System](./memory.md)
- [Retrieval](./retrieval.md)
- [Planning](./planning.md)
- [Execution](./execution.md)
- [JudgeAgent and Validator Framework](./judge_validator.md)
- [Self-Healing and Adaptation](./self_healing.md)
- [Multi-Agent Orchestration](./orchestration.md)
- [Tool Factory](./tool_factory.md)
- [Hot-Loading System](./hot_loading.md)
- [Model Adapters](./model_adapters.md)
- [Advanced Features](./advanced_features.md)
- [Model Response Caching](./model_caching.md)
- [Graph-Aligned Sparse Attention (GASA)](./gasa.md)
- [Extensibility](./extensibility.md)
- [API Reference](./api_reference.md)

## Quick Links

- [Installation Guide](./getting_started.md#installation)
- [Quick Start](./getting_started.md#quick-start)
- [Examples](./examples.md)
- [Troubleshooting](./troubleshooting.md)
- [Contributing](./contributing.md)

## About Saplings

Saplings is a graphs-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day through automated critique → fine-tune loops.

Key features include:

- **Structural Memory** — Vector + graph store per corpus
- **Cascaded, Entropy-Aware Retrieval** — TF-IDF → embeddings → graph expansion
- **Guard-railed Generation** — Planner with budget, Executor with speculative draft/verify
- **JudgeAgent & Validator Loop** — Reflexive scoring, self-healing patches
- **Self-Healing & Adaptation** — Error analysis, automatic patching, and LoRA fine-tuning
- **Extensibility** — Hot-pluggable models, tools, validators
- **Graph-Aligned Sparse Attention (GASA)** — Graph-conditioned attention masks for faster, better-grounded reasoning
