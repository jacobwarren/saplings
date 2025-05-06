from __future__ import annotations

import pathlib
import re
import sys
from subprocess import run
from typing import NamedTuple


class ComplexityResult(NamedTuple):
    """Result of complexity analysis."""

    filename: str
    lineno: int
    name: str
    complexity: int


def main() -> None:
    """
    Check the cyclomatic complexity of all Python files in the src directory.
    Fails if the worst complexity is > 10 or the average complexity is > 7.
    """
    src_path = pathlib.Path("src")
    if not src_path.exists():
        print("src directory not found")
        sys.exit(0)

    # Run radon cc command directly
    process = run(
        ["radon", "cc", "-s", "--no-assert", "src"], capture_output=True, text=True, check=False
    )

    if process.returncode != 0:
        print(f"Error running radon: {process.stderr}")
        sys.exit(1)

    # Parse results
    results = []
    output = process.stdout

    # Pattern to match results
    # Format example: "F 122:0 main - 20"
    pattern = r"([A-F]) (\S+):(\d+) (\S+) - (\d+)"

    for line in output.splitlines():
        match = re.match(pattern, line)
        if not match:
            continue

        _, filename, lineno, name, complexity = match.groups()
        results.append(
            ComplexityResult(
                filename=filename, lineno=int(lineno), name=name, complexity=int(complexity)
            )
        )

    if not results:
        print("No complexity results found")
        sys.exit(0)

    # Calculate metrics
    complexities = [r.complexity for r in results]
    worst = max(complexities) if complexities else 0
    avg = sum(complexities) / len(complexities) if complexities else 0

    # Print summary
    print(f"Total blocks analyzed: {len(results)}")
    print(f"Average complexity: {avg:.2f}")
    print(f"Worst complexity: {worst}")

    # Check against thresholds
    if worst > 10 or avg > 7:
        print(f"FAIL: worst={worst}, avg={avg:.2f}")
        print("Threshold exceeded: worst must be <= 10 and average must be <= 7")

        # Print worst offenders
        print("\nWorst offenders:")
        sorted_results = sorted(results, key=lambda r: r.complexity, reverse=True)
        for r in sorted_results[:5]:  # Show top 5 worst offenders
            print(f"{r.filename}:{r.lineno} - {r.name} (Complexity: {r.complexity})")

        sys.exit(1)

    print("SUCCESS: Code complexity is within acceptable limits")


if __name__ == "__main__":
    main()
