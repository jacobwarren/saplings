from __future__ import annotations

try:
    from saplings.agent_config import AgentConfig

    print("Import AgentConfig successful!")

    # Try to create an instance
    config = AgentConfig(provider="test", model_name="test-model")
    print("Created AgentConfig instance successfully!")

except Exception as e:
    print(f"Import or instantiation failed: {e}")
