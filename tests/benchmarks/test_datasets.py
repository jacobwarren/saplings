"""
Test datasets for benchmarking.

This module provides standardized datasets for benchmarking the Saplings framework.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from saplings.memory import Document, DocumentMetadata, DependencyGraph


class TestDatasets:
    """Standardized datasets for benchmarking."""

    @staticmethod
    def create_document_corpus(
        num_documents: int = 10,
        document_length: int = 200,
        with_metadata: bool = True,
        with_embeddings: bool = False,
        embedding_dim: int = 768,
    ) -> List[Document]:
        """
        Create a corpus of test documents.

        Args:
            num_documents: Number of documents to create
            document_length: Approximate length of each document in characters
            with_metadata: Whether to include metadata
            with_embeddings: Whether to include embeddings
            embedding_dim: Dimension of embeddings

        Returns:
            List[Document]: List of documents
        """
        documents = []
        for i in range(num_documents):
            # Create document content
            content = f"This is test document {i+1} with content for benchmarking. "
            content += f"It contains information that relates to other documents in the set. "
            content += f"Specifically, it references concepts from documents {max(1, i-1)} "
            content += f"and {min(num_documents, i+2)}. "

            # Pad content to reach desired length
            while len(content) < document_length:
                content += "Additional content to reach desired length. "

            # Create metadata if requested
            metadata = None
            if with_metadata:
                # Use modulo to ensure valid dates (1-28 for January)
                day = (i % 28) + 1
                metadata = DocumentMetadata(
                    source=f"test_doc_{i+1}.txt",
                    document_id=f"doc_{i+1}",
                    created_at=f"2023-01-{day:02d}T00:00:00Z",
                    author="Benchmark Test",
                    tags=["test", "benchmark", f"doc{i+1}"],
                )

            # Create embedding if requested
            embedding = None
            if with_embeddings:
                # Create a random embedding
                embedding = np.random.randn(embedding_dim).astype(np.float32)
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)

            # Create document
            doc = Document(
                id=f"doc_{i+1}",
                content=content,
                metadata=metadata,
                embedding=embedding,
            )

            documents.append(doc)

        return documents

    @staticmethod
    def create_document_graph(
        documents: List[Document],
        connection_density: float = 0.3,
    ) -> DependencyGraph:
        """
        Create a dependency graph for a set of documents.

        Args:
            documents: List of documents
            connection_density: Density of connections (0.0-1.0)

        Returns:
            DependencyGraph: Dependency graph
        """
        # Create graph
        graph = DependencyGraph()

        # Add documents to graph
        nodes = []
        for doc in documents:
            node = graph.add_document_node(doc)
            nodes.append(node)

        # Add relationships
        num_documents = len(documents)
        for i in range(num_documents):
            # Connect to previous document
            if i > 0:
                graph.add_edge(
                    source_id=nodes[i].id,
                    target_id=nodes[i-1].id,
                    relationship_type="references"
                )

            # Connect to next document
            if i < num_documents - 1:
                graph.add_edge(
                    source_id=nodes[i].id,
                    target_id=nodes[i+1].id,
                    relationship_type="references"
                )

            # Add random connections based on density
            for j in range(num_documents):
                if i != j and i != j-1 and i != j+1:  # Skip already connected nodes
                    if np.random.random() < connection_density:
                        graph.add_edge(
                            source_id=nodes[i].id,
                            target_id=nodes[j].id,
                            relationship_type="references"
                        )

        return graph

    @staticmethod
    def create_query_set(
        num_queries: int = 10,
        documents: Optional[List[Document]] = None,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Create a set of test queries with ground truth relevance.

        Args:
            num_queries: Number of queries to create
            documents: List of documents to reference in queries

        Returns:
            List[Dict[str, Union[str, List[str]]]]: List of queries with ground truth
        """
        queries = []

        # Define query templates
        templates = [
            "What information is available about {topic}?",
            "Summarize the key points about {topic}.",
            "How does {topic} relate to other concepts?",
            "What are the main characteristics of {topic}?",
            "Explain the relationship between {topic} and {related_topic}.",
        ]

        # Define topics
        topics = [
            "machine learning",
            "artificial intelligence",
            "natural language processing",
            "computer vision",
            "deep learning",
            "reinforcement learning",
            "neural networks",
            "data science",
            "big data",
            "robotics",
        ]

        for i in range(num_queries):
            # Select template and topics
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            related_topic = topics[(i + 3) % len(topics)]

            # Create query
            query = template.format(topic=topic, related_topic=related_topic)

            # Determine relevant documents if provided
            relevant_docs = []
            if documents:
                for doc in documents:
                    # Simple relevance check: if topic appears in content
                    if topic.lower() in doc.content.lower():
                        relevant_docs.append(doc.id)

            # Create query entry
            query_entry = {
                "query": query,
                "relevant_docs": relevant_docs,
            }

            queries.append(query_entry)

        return queries

    @staticmethod
    def create_code_samples(
        num_samples: int = 10,
        with_errors: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Create a set of code samples for testing.

        Args:
            num_samples: Number of code samples to create
            with_errors: Whether to include errors in some samples

        Returns:
            List[Dict[str, str]]: List of code samples
        """
        samples = []

        # Define code templates
        templates = [
            # Function to calculate factorial
            {
                "correct": """
def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
                "error": """
def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n < 1:
        return 1
    return n * factorial(n)  # Missing decrement, will cause infinite recursion
""",
                "error_type": "RecursionError",
                "error_message": "maximum recursion depth exceeded",
            },
            # Function to find maximum in a list
            {
                "correct": """
def find_max(numbers):
    \"\"\"Find the maximum value in a list of numbers.\"\"\"
    if not numbers:
        return None
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value
""",
                "error": """
def find_max(numbers):
    \"\"\"Find the maximum value in a list of numbers.\"\"\"
    max_value = numbers[0]  # Will fail if list is empty
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value
""",
                "error_type": "IndexError",
                "error_message": "list index out of range",
            },
            # Function to reverse a string
            {
                "correct": """
def reverse_string(s):
    \"\"\"Reverse a string.\"\"\"
    return s[::-1]
""",
                "error": """
def reverse_string(s):
    \"\"\"Reverse a string.\"\"\"
    result = ""
    for i in range(len(s)):
        result = s[i] + result
    return result  # Inefficient but not an error
""",
                "error_type": None,
                "error_message": None,
            },
            # Function to check if a number is prime
            {
                "correct": """
def is_prime(n):
    \"\"\"Check if a number is prime.\"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
""",
                "error": """
def is_prime(n):
    \"\"\"Check if a number is prime.\"\"\"
    if n <= 1:
        return False
    for i in range(2, n):  # Inefficient, should check up to sqrt(n)
        if n % i == 0:
            return False
    return True
""",
                "error_type": None,
                "error_message": None,
            },
            # Function to calculate Fibonacci numbers
            {
                "correct": """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
""",
                "error": """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)  # Will fail for negative n
""",
                "error_type": "RecursionError",
                "error_message": "maximum recursion depth exceeded",
            },
        ]

        for i in range(num_samples):
            # Select template
            template = templates[i % len(templates)]

            # Determine whether to use correct or error version
            use_error = with_errors and i % 3 == 0  # Every 3rd sample has an error

            code = template["error"] if use_error else template["correct"]
            error_type = template["error_type"] if use_error else None
            error_message = template["error_message"] if use_error else None

            # Create sample entry
            sample = {
                "code": code,
                "error_type": error_type,
                "error_message": error_message,
                "has_error": use_error,
            }

            samples.append(sample)

        return samples

    @staticmethod
    def create_evaluation_tasks(
        num_tasks: int = 10,
    ) -> List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]:
        """
        Create a set of evaluation tasks with human judgments.

        Args:
            num_tasks: Number of tasks to create

        Returns:
            List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]: List of tasks
        """
        tasks = []

        # Define task templates
        templates = [
            {
                "task": "Summarize the following text: {text}",
                "text": "The artificial intelligence field has seen rapid advancement in recent years, "
                        "particularly in the areas of deep learning and natural language processing. "
                        "These technologies have enabled new applications such as autonomous vehicles, "
                        "virtual assistants, and advanced recommendation systems.",
                "responses": [
                    {
                        "response": "AI has advanced quickly, especially in deep learning and NLP, "
                                    "enabling autonomous vehicles, virtual assistants, and recommendation systems.",
                        "score": 5,
                        "feedback": "Concise and captures all key points.",
                    },
                    {
                        "response": "AI has made progress in recent years.",
                        "score": 2,
                        "feedback": "Too vague and missing specific advancements and applications.",
                    },
                    {
                        "response": "Artificial intelligence has advanced rapidly, with deep learning and NLP "
                                    "being key areas of progress. These technologies have led to applications "
                                    "like self-driving cars, virtual assistants, and recommendation systems.",
                        "score": 4,
                        "feedback": "Good coverage but could be more concise.",
                    },
                ],
            },
            {
                "task": "Write a function to calculate the factorial of a number.",
                "responses": [
                    {
                        "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                        "score": 5,
                        "feedback": "Correct implementation with proper base case.",
                    },
                    {
                        "response": "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
                        "score": 5,
                        "feedback": "Correct iterative implementation.",
                    },
                    {
                        "response": "def factorial(n):\n    return n * factorial(n - 1)",
                        "score": 1,
                        "feedback": "Missing base case, will cause infinite recursion.",
                    },
                ],
            },
            {
                "task": "Explain the concept of machine learning to a non-technical person.",
                "responses": [
                    {
                        "response": "Machine learning is a type of artificial intelligence that allows computers "
                                    "to learn from data without being explicitly programmed. It's like teaching "
                                    "a child to recognize animals by showing them many examples, rather than "
                                    "giving them specific rules.",
                        "score": 5,
                        "feedback": "Clear explanation with a relatable analogy.",
                    },
                    {
                        "response": "Machine learning involves training models on data using algorithms like "
                                    "gradient descent to minimize loss functions and optimize parameters.",
                        "score": 1,
                        "feedback": "Too technical for a non-technical audience.",
                    },
                    {
                        "response": "Machine learning is when computers learn from examples.",
                        "score": 3,
                        "feedback": "Accurate but too brief and lacks depth.",
                    },
                ],
            },
        ]

        for i in range(num_tasks):
            # Select template
            template = templates[i % len(templates)]

            # Create task
            task = template["task"]
            if "{text}" in task:
                task = task.format(text=template["text"])

            # Create task entry
            task_entry = {
                "task": task,
                "responses": template["responses"],
            }

            tasks.append(task_entry)

        return tasks
