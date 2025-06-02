#!/usr/bin/env python3
"""
FAISS Vector Store Example

This example demonstrates using FAISS (Facebook AI Similarity Search) 
for advanced vector storage and similarity search in Saplings agents.
"""

import asyncio
import os
import numpy as np
from typing import List, Dict, Any
from saplings import AgentBuilder

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu")


class FAISSVectorStore:
    """Custom FAISS-based vector store for document embeddings."""
    
    def __init__(self, dimension: int = 1536):  # OpenAI ada-002 embedding dimension
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.is_trained = False
        
    def initialize_index(self, index_type: str = "flat"):
        """Initialize FAISS index with specified type."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not installed")
            
        if index_type == "flat":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "ivf":
            # Inverted File index for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World for very fast search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add documents with their embeddings to the store."""
        if self.index is None:
            self.initialize_index()
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeddings) >= 100:  # Need enough data to train
                self.index.train(embeddings)
                self.is_trained = True
            else:
                print("Warning: Not enough data to train IVF index, using flat index instead")
                self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None or len(self.documents) == 0:
            return []
        
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity_score": 1.0 / (1.0 + distance),  # Convert distance to similarity
                    "rank": i + 1
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "index_type": type(self.index).__name__ if self.index else None,
            "dimension": self.dimension,
            "is_trained": getattr(self.index, 'is_trained', True) if self.index else False,
            "memory_usage_mb": (self.index.ntotal * self.dimension * 4) / (1024 * 1024) if self.index else 0
        }


def create_mock_embeddings(texts: List[str], dimension: int = 1536) -> np.ndarray:
    """Create mock embeddings for demonstration (in real usage, use OpenAI embeddings API)."""
    # Generate consistent fake embeddings based on text content
    embeddings = []
    for text in texts:
        # Simple hash-based embedding for demo
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.normal(0, 1, dimension)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    return np.array(embeddings)


async def demonstrate_basic_faiss_usage():
    """Demonstrate basic FAISS vector store usage."""
    print("=== Basic FAISS Vector Store Usage ===\n")
    
    if not FAISS_AVAILABLE:
        print("FAISS not available - skipping example")
        return
    
    print("1. Creating FAISS vector store...")
    
    vector_store = FAISSVectorStore(dimension=1536)
    vector_store.initialize_index("flat")
    
    print("2. Adding documents to vector store...")
    
    # Sample documents about AI/ML topics
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations.",
        "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.",
        "Supervised learning uses labeled data to train models that can make predictions on new, unseen data.",
        "Unsupervised learning finds hidden patterns in data without using labeled examples.",
        "Transfer learning leverages pre-trained models to solve new but related tasks with less data.",
        "Ensemble methods combine multiple models to create a stronger predictor than individual models.",
        "Feature engineering involves selecting and transforming variables to improve model performance."
    ]
    
    metadata = [
        {"topic": "ml_basics", "category": "definition", "complexity": "beginner"},
        {"topic": "deep_learning", "category": "definition", "complexity": "intermediate"},
        {"topic": "nlp", "category": "definition", "complexity": "intermediate"},
        {"topic": "computer_vision", "category": "definition", "complexity": "intermediate"},
        {"topic": "reinforcement_learning", "category": "definition", "complexity": "advanced"},
        {"topic": "supervised_learning", "category": "definition", "complexity": "beginner"},
        {"topic": "unsupervised_learning", "category": "definition", "complexity": "intermediate"},
        {"topic": "transfer_learning", "category": "definition", "complexity": "advanced"},
        {"topic": "ensemble_methods", "category": "definition", "complexity": "intermediate"},
        {"topic": "feature_engineering", "category": "definition", "complexity": "intermediate"}
    ]
    
    # Create embeddings (in real usage, use OpenAI embeddings API)
    embeddings = create_mock_embeddings(documents)
    
    # Add to vector store
    vector_store.add_documents(documents, embeddings, metadata)
    
    print("3. Testing similarity search...")
    
    # Test queries
    test_queries = [
        "How do neural networks learn representations?",
        "What techniques work with limited labeled data?",
        "How can I improve my model's accuracy?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Create query embedding
        query_embedding = create_mock_embeddings([query])[0]
        
        # Search for similar documents
        results = vector_store.search(query_embedding, k=3)
        
        print("Top 3 similar documents:")
        for result in results:
            print(f"  Rank {result['rank']}: {result['document'][:80]}...")
            print(f"    Topic: {result['metadata']['topic']}, Similarity: {result['similarity_score']:.3f}")
    
    # Show stats
    print(f"\nVector Store Stats: {vector_store.get_stats()}")


async def demonstrate_agent_with_faiss():
    """Demonstrate Saplings agent with FAISS-based memory."""
    print("\n=== Agent with FAISS Memory ===\n")
    
    if not FAISS_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        print("Skipping - FAISS or OpenAI API key not available")
        return
    
    print("1. Setting up agent with FAISS-enhanced memory...")
    
    # Create agent with GASA and memory optimization
    agent = AgentBuilder.for_openai(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        gasa_enabled=True,
        gasa_strategy="binary",
        memory_path="./faiss_agent_memory"
    ).build()
    
    print("2. Creating specialized FAISS store for agent knowledge...")
    
    # Create FAISS store for AI research papers abstracts
    research_store = FAISSVectorStore(dimension=1536)
    research_store.initialize_index("hnsw")  # Fast for real-time search
    
    # Sample research paper abstracts
    research_papers = [
        {
            "abstract": "We present Transformer, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "title": "Attention Is All You Need",
            "year": 2017,
            "field": "nlp"
        },
        {
            "abstract": "We introduce BERT, a new language representation model which stands for Bidirectional Encoder Representations from Transformers.",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2018,
            "field": "nlp"
        },
        {
            "abstract": "ResNet introduces skip connections that allow training of very deep neural networks by addressing the vanishing gradient problem.",
            "title": "Deep Residual Learning for Image Recognition",
            "year": 2016,
            "field": "computer_vision"
        },
        {
            "abstract": "We demonstrate that reinforcement learning agents can achieve superhuman performance in complex video games through deep Q-learning.",
            "title": "Human-level control through deep reinforcement learning",
            "year": 2015,
            "field": "reinforcement_learning"
        },
        {
            "abstract": "Generative Adversarial Networks consist of two neural networks competing in a zero-sum game framework to generate realistic data.",
            "title": "Generative Adversarial Networks",
            "year": 2014,
            "field": "generative_ai"
        }
    ]
    
    # Prepare data for FAISS
    abstracts = [paper["abstract"] for paper in research_papers]
    metadata = [{"title": p["title"], "year": p["year"], "field": p["field"]} for p in research_papers]
    
    # Add to vector store
    embeddings = create_mock_embeddings(abstracts)
    research_store.add_documents(abstracts, embeddings, metadata)
    
    # Add documents to agent memory as well
    for paper in research_papers:
        await agent.add_document(
            content=f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
            metadata={"type": "research_paper", **paper}
        )
    
    print("3. Testing agent with FAISS-enhanced retrieval...")
    
    query = "How do attention mechanisms work in neural networks and what impact have they had on NLP?"
    
    # First, use FAISS to find relevant papers
    query_embedding = create_mock_embeddings([query])[0]
    relevant_papers = research_store.search(query_embedding, k=3)
    
    print("FAISS found these relevant papers:")
    for paper in relevant_papers:
        print(f"  - {paper['metadata']['title']} (similarity: {paper['similarity_score']:.3f})")
    
    # Now query the agent with context
    response = await agent.run(f"""
    Based on the research papers in your memory, please answer: {query}
    
    Pay special attention to papers about attention mechanisms and transformers.
    Provide a comprehensive answer that explains both the technical concepts and their impact.
    """)
    
    print(f"\nAgent response with FAISS-enhanced memory:\n{response}")


async def demonstrate_faiss_index_types():
    """Demonstrate different FAISS index types and their characteristics."""
    print("\n=== FAISS Index Types Comparison ===\n")
    
    if not FAISS_AVAILABLE:
        print("FAISS not available - skipping example")
        return
    
    print("1. Comparing different FAISS index types...")
    
    # Prepare test data
    documents = [f"Document {i}: This is a test document about topic {i%5}" for i in range(1000)]
    embeddings = create_mock_embeddings(documents, dimension=512)  # Smaller dimension for speed
    metadata = [{"doc_id": i, "topic": i%5} for i in range(1000)]
    
    # Test different index types
    index_configs = [
        {"type": "flat", "name": "Flat (Exact Search)"},
        {"type": "ivf", "name": "IVF (Approximate Search)"},
        {"type": "hnsw", "name": "HNSW (Very Fast Approximate)"}
    ]
    
    results = {}
    
    for config in index_configs:
        print(f"\n2. Testing {config['name']}...")
        
        # Create vector store with specific index type
        store = FAISSVectorStore(dimension=512)
        store.initialize_index(config["type"])
        
        # Measure indexing time
        import time
        start_time = time.time()
        store.add_documents(documents, embeddings, metadata)
        index_time = time.time() - start_time
        
        # Measure search time
        query_embedding = create_mock_embeddings(["test query"], dimension=512)[0]
        
        start_time = time.time()
        search_results = store.search(query_embedding, k=10)
        search_time = time.time() - start_time
        
        # Store results
        results[config["type"]] = {
            "name": config["name"],
            "index_time": index_time,
            "search_time": search_time,
            "results_count": len(search_results),
            "stats": store.get_stats()
        }
        
        print(f"   Indexing time: {index_time:.4f}s")
        print(f"   Search time: {search_time:.6f}s")
        print(f"   Memory usage: {store.get_stats()['memory_usage_mb']:.1f} MB")
    
    print("\n3. Performance Summary:")
    print(f"{'Index Type':<25} {'Index Time':<12} {'Search Time':<12} {'Memory (MB)':<12}")
    print("-" * 65)
    
    for result in results.values():
        print(f"{result['name']:<25} {result['index_time']:<12.4f} {result['search_time']:<12.6f} {result['stats']['memory_usage_mb']:<12.1f}")


async def demonstrate_faiss_with_filtering():
    """Demonstrate FAISS with metadata filtering."""
    print("\n=== FAISS with Metadata Filtering ===\n")
    
    if not FAISS_AVAILABLE:
        print("FAISS not available - skipping example")
        return
    
    print("1. Creating filtered vector store...")
    
    # Enhanced vector store with filtering capability
    class FilteredFAISSStore(FAISSVectorStore):
        def search_with_filter(self, query_embedding: np.ndarray, k: int = 5, 
                             filter_func=None) -> List[Dict[str, Any]]:
            """Search with metadata filtering."""
            # Get more results to account for filtering
            search_k = min(k * 3, len(self.documents))
            initial_results = self.search(query_embedding, search_k)
            
            # Apply filter if provided
            if filter_func:
                filtered_results = [r for r in initial_results if filter_func(r['metadata'])]
                return filtered_results[:k]
            
            return initial_results[:k]
    
    # Create enhanced store
    store = FilteredFAISSStore(dimension=384)  # Smaller embeddings for demo
    store.initialize_index("flat")
    
    # Add diverse content
    programming_docs = [
        ("Python is a high-level programming language with dynamic typing.", {"language": "python", "difficulty": "beginner", "category": "programming"}),
        ("JavaScript enables dynamic web page interactions and is essential for frontend development.", {"language": "javascript", "difficulty": "beginner", "category": "programming"}),
        ("Rust provides memory safety without garbage collection through ownership.", {"language": "rust", "difficulty": "advanced", "category": "programming"}),
        ("Machine learning algorithms can automatically learn patterns from data.", {"language": "python", "difficulty": "intermediate", "category": "ai"}),
        ("Deep neural networks require careful initialization and regularization.", {"language": "python", "difficulty": "advanced", "category": "ai"}),
        ("React creates reusable UI components for building user interfaces.", {"language": "javascript", "difficulty": "intermediate", "category": "frontend"}),
        ("CSS Grid provides powerful layout capabilities for web design.", {"language": "css", "difficulty": "intermediate", "category": "frontend"}),
        ("Kubernetes orchestrates containerized applications at scale.", {"language": "yaml", "difficulty": "advanced", "category": "devops"}),
    ]
    
    documents = [doc[0] for doc in programming_docs]
    metadata = [doc[1] for doc in programming_docs]
    embeddings = create_mock_embeddings(documents, dimension=384)
    
    store.add_documents(documents, embeddings, metadata)
    
    print("2. Testing filtered searches...")
    
    # Test queries with different filters
    query = "How to learn programming effectively?"
    query_embedding = create_mock_embeddings([query], dimension=384)[0]
    
    # Search without filter
    print("\nAll results:")
    all_results = store.search(query_embedding, k=5)
    for i, result in enumerate(all_results, 1):
        print(f"  {i}. {result['document'][:60]}... (Category: {result['metadata']['category']})")
    
    # Search only programming content
    print("\nFiltered - Programming only:")
    programming_results = store.search_with_filter(
        query_embedding, k=5,
        filter_func=lambda meta: meta['category'] == 'programming'
    )
    for i, result in enumerate(programming_results, 1):
        print(f"  {i}. {result['document'][:60]}... (Language: {result['metadata']['language']})")
    
    # Search only beginner content
    print("\nFiltered - Beginner level only:")
    beginner_results = store.search_with_filter(
        query_embedding, k=5,
        filter_func=lambda meta: meta['difficulty'] == 'beginner'
    )
    for i, result in enumerate(beginner_results, 1):
        print(f"  {i}. {result['document'][:60]}... (Difficulty: {result['metadata']['difficulty']})")


async def main():
    """Run all FAISS vector store examples."""
    await demonstrate_basic_faiss_usage()
    await demonstrate_agent_with_faiss()
    await demonstrate_faiss_index_types()
    await demonstrate_faiss_with_filtering()
    
    print("\n=== FAISS Vector Store Examples Complete ===")
    print("\nKey Benefits Demonstrated:")
    print("- High-performance similarity search with FAISS")
    print("- Multiple index types for different performance/accuracy trade-offs")
    print("- Integration with Saplings agent memory systems")
    print("- Metadata filtering for targeted search")
    print("- Scalable vector storage for large document collections")
    print("\nNext Steps:")
    print("- Use real embeddings from OpenAI or other providers")
    print("- Implement persistent storage for FAISS indices")
    print("- Add more sophisticated filtering and ranking")
    print("- Optimize index parameters for your specific use case")


if __name__ == "__main__":
    asyncio.run(main())