"""
This example demonstrates a self-improving agent that uses the JudgeAgent to evaluate its performance,
generate patches, and adapt over time through continuous learning.
"""

from __future__ import annotations

import asyncio

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, RubricCriterion
from saplings.self_heal import PatchGenerator

# Create a dataset of questions and expected answers
QUESTIONS = [
    {
        "question": "What is the square root of 144?",
        "expected_answer": "12",
        "difficulty": "easy",
        "category": "math",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "expected_answer": "Jane Austen",
        "difficulty": "medium",
        "category": "literature",
    },
    {
        "question": "What is the capital of Brazil?",
        "expected_answer": "Brasília",
        "difficulty": "medium",
        "category": "geography",
    },
    {
        "question": "What is the time complexity of quicksort in the average case?",
        "expected_answer": "O(n log n)",
        "difficulty": "hard",
        "category": "computer science",
    },
    {
        "question": "What causes rust on metal?",
        "expected_answer": "Oxidation, specifically when iron reacts with oxygen and water",
        "difficulty": "medium",
        "category": "chemistry",
    },
]


async def main():
    # Create model
    print("Creating model...")
    model = LLM.create("openai", "gpt-4o")

    # Create a evaluation rubric
    print("Creating evaluation rubric...")
    rubric = Rubric(
        name="QA Evaluation Rubric",
        description="Evaluates the quality of answers to questions",
        criteria=[
            RubricCriterion(
                name="correctness",
                description="Factual accuracy of the answer",
                weight=0.5,
                scoring_guide="5: Perfect accuracy, 4: Minor errors, 3: Partially correct, 2: Mostly incorrect, 1: Completely incorrect",
            ),
            RubricCriterion(
                name="completeness",
                description="How complete the answer is",
                weight=0.3,
                scoring_guide="5: Comprehensive, 4: Covers most aspects, 3: Covers main points, 2: Missing key elements, 1: Extremely incomplete",
            ),
            RubricCriterion(
                name="conciseness",
                description="How concise and to-the-point the answer is",
                weight=0.2,
                scoring_guide="5: Perfectly concise, 4: Mostly concise, 3: Somewhat verbose, 2: Very verbose, 1: Extremely verbose",
            ),
        ],
    )

    # Create JudgeAgent
    print("Creating JudgeAgent...")
    judge_config = JudgeConfig(enable_detailed_feedback=True)
    judge = JudgeAgent(model=model, config=judge_config)

    # Create patch generator
    print("Creating patch generator...")
    patch_generator = PatchGenerator(model=model)

    # Create agent
    print("Creating agent...")
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_self_healing=True,
        )
    )

    # Learning loop
    print("\nStarting learning loop...")

    # Only run 2 iterations for demo purposes
    for iteration in range(2):
        print(f"\n=== Iteration {iteration + 1} ===")

        all_results = []
        all_judgments = []

        # Test agent on all questions
        for q_idx, question_data in enumerate(QUESTIONS):
            question = question_data["question"]
            expected = question_data["expected_answer"]

            print(f"\nQuestion {q_idx + 1}: {question}")
            print(f"Expected: {expected}")

            # Get agent's answer
            result = await agent.run(question)
            print(f"Agent: {result}")

            # Judge the answer
            judgment = await judge.judge_with_rubric(
                prompt=question, response=result, rubric=rubric, reference=expected
            )

            # Store results and judgments
            all_results.append(
                {"question": question, "expected": expected, "answer": result, "judgment": judgment}
            )
            all_judgments.append(judgment)

            # Print judgment
            total_score = judgment.get("total_score", 0)
            print(f"Score: {total_score:.2f}/5.0")

            for criterion, score in judgment.get("criteria_scores", {}).items():
                print(f"- {criterion}: {score:.2f}/5.0")

        # Calculate overall performance
        avg_score = sum(j.get("total_score", 0) for j in all_judgments) / len(all_judgments)
        print(f"\nAverage score: {avg_score:.2f}/5.0")

        # Generate patches based on judgments
        print("\nGenerating improvement patches...")
        patches = await patch_generator.generate_patches(
            results=all_results, judgments=all_judgments
        )

        # Apply patches to the agent
        print("Applying patches to the agent...")
        for i, patch in enumerate(patches):
            print(f"Applying patch {i+1}: {patch.name}")
            await agent.apply_patch(patch)

        print(f"Completed iteration {iteration + 1}")

    print("\nLearning complete. Final evaluation:")

    # Final evaluation
    final_scores = []
    for q_idx, question_data in enumerate(QUESTIONS):
        question = question_data["question"]
        expected = question_data["expected_answer"]

        result = await agent.run(question)
        judgment = await judge.judge_with_rubric(
            prompt=question, response=result, rubric=rubric, reference=expected
        )

        final_scores.append(judgment.get("total_score", 0))

        print(f"\nQuestion: {question}")
        print(f"Answer: {result}")
        print(f"Score: {judgment.get('total_score', 0):.2f}/5.0")

    final_avg = sum(final_scores) / len(final_scores)
    print(f"\nFinal average score: {final_avg:.2f}/5.0")

    print("\nThis example demonstrates how Saplings can create self-improving agents")
    print("that learn from their mistakes and adapt over time through continuous feedback.")


if __name__ == "__main__":
    asyncio.run(main())
