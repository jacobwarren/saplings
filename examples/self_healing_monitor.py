"""
This example creates a system monitoring agent that self-improves
by learning from its errors and adapting its detection strategies.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime

from saplings import Agent, AgentConfig
from saplings.self_heal import SuccessPairCollector
from saplings.tools import PythonInterpreterTool


# Simulated system data functions
def get_system_metrics():
    """Simulate getting system metrics"""
    return {
        "cpu": random.uniform(5, 95),
        "memory": random.uniform(20, 90),
        "disk": random.uniform(30, 95),
        "network": random.uniform(0, 100),
        "errors": random.randint(0, 5),
        "timestamp": datetime.now().isoformat(),
    }


def inject_anomaly(metrics, anomaly_type=None):
    """Inject an anomaly into the metrics"""
    if not anomaly_type:
        anomaly_types = ["cpu_spike", "memory_leak", "disk_full", "network_drop", "error_surge"]
        anomaly_type = random.choice(anomaly_types)

    if anomaly_type == "cpu_spike":
        metrics["cpu"] = random.uniform(95, 100)
    elif anomaly_type == "memory_leak":
        metrics["memory"] = random.uniform(95, 100)
    elif anomaly_type == "disk_full":
        metrics["disk"] = random.uniform(98, 100)
    elif anomaly_type == "network_drop":
        metrics["network"] = random.uniform(0, 5)
    elif anomaly_type == "error_surge":
        metrics["errors"] = random.randint(20, 100)

    return metrics, anomaly_type


async def main():
    # Create tools
    python_tool = PythonInterpreterTool()

    # Create agent with self-healing enabled
    print("Creating self-healing agent...")
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=[python_tool],
            enable_self_healing=True,
        )
    )

    # Create success pair collector for self-improvement
    collector = SuccessPairCollector()

    # Initial detection prompt
    detection_prompt = """
    You are a system monitoring agent. Analyze the following metrics and determine:
    1. If there is an anomaly
    2. What type of anomaly it is
    3. Recommended actions to address it

    Metrics: {metrics}
    """

    # Monitoring loop
    print("Starting system monitoring...")
    iterations = 10  # Reduced for example purposes
    for i in range(iterations):
        # Get metrics
        metrics = get_system_metrics()

        # Every 3rd iteration, inject an anomaly
        inject_anomaly_this_time = i % 3 == 0 and i > 0
        actual_anomaly_type = None

        if inject_anomaly_this_time:
            metrics, actual_anomaly_type = inject_anomaly(metrics)
            print(f"\n[INJECTED ANOMALY: {actual_anomaly_type}]")

        # Format the prompt with current metrics
        current_prompt = detection_prompt.format(metrics=metrics)

        # Run the agent
        start_time = time.time()
        result = await agent.run(current_prompt)
        end_time = time.time()

        print(f"\nIteration {i+1}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Agent analysis:\n{result}")

        # Evaluate the result and collect success/failure pairs
        if inject_anomaly_this_time:
            # Check if agent detected the correct anomaly
            detected_correctly = actual_anomaly_type.lower() in result.lower()

            if detected_correctly:
                print("✅ Agent correctly identified the anomaly")
                collector.add_success_pair(current_prompt, result)
            else:
                print("❌ Agent failed to identify the correct anomaly")
                # Provide corrected response for learning
                corrected_result = f"Analysis: There is an anomaly of type {actual_anomaly_type}."
                collector.add_failure_pair(current_prompt, result, corrected_result)

        # Every 6 iterations, trigger self-improvement
        if i > 0 and i % 6 == 0:
            print("\n--- Triggering self-improvement ---")
            await agent.self_heal(collector.get_pairs())
            print("Self-improvement completed")

        # Pause between iterations
        await asyncio.sleep(1)

    print("\nMonitoring complete. Agent has learned from experience.")
    print(f"Success pairs collected: {len(collector.success_pairs)}")
    print(f"Failure pairs collected: {len(collector.failure_pairs)}")


if __name__ == "__main__":
    asyncio.run(main())
