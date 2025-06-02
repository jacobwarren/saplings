from __future__ import annotations

try:
    from saplings import Agent
    from saplings.agent_config import AgentConfig

    print("Import Agent successful!")

    # Try to create an instance
    config = AgentConfig(provider="test", model_name="test-model")
    print("Created AgentConfig instance successfully!")

    # Try to create an Agent instance
    try:
        agent = Agent(config)
        print("Created Agent instance successfully!")
    except Exception as e:
        print(f"Agent instantiation failed: {e}")

except Exception as e:
    print(f"Import failed: {e}")
