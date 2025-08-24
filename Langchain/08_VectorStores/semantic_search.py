"""
Semantic Search and Advanced Retrieval Patterns

This module demonstrates advanced semantic search patterns and retrieval strategies:
1. Multi-vector search approaches
2. Hybrid search (dense + sparse)
3. Contextual retrieval with re-ranking
4. Query expansion and refinement
5. Similarity threshold optimization
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import time
from collections import defaultdict
import re

load_dotenv()

def multi_vector_search_example():
    """Demonstrate multi-vector search approaches"""
    print("=== Multi-Vector Search Approaches ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Create diverse knowledge base
    knowledge_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            metadata={"topic": "Programming", "language": "Python", "difficulty": "Beginner", "type": "Overview"}
        ),
        Document(
            page_content="Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Supervised learning uses labeled data to train models for prediction tasks.",
            metadata={"topic": "AI/ML", "subtopic": "Machine Learning", "difficulty": "Intermediate", "type": "Classification"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using mathematical operations.",
            metadata={"topic": "AI/ML", "subtopic": "Deep Learning", "difficulty": "Advanced", "type": "Architecture"}
        ),
        Document(
            page_content="Data preprocessing is a crucial step in machine learning that involves cleaning, transforming, and organizing raw data before feeding it to algorithms.",
            metadata={"topic": "Data Science", "subtopic": "Preprocessing", "difficulty": "Intermediate", "type": "Process"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It combines computational linguistics with machine learning.",
            metadata={"topic": "AI/ML", "subtopic": "NLP", "difficulty": "Advanced", "type": "Field"}
        ),
        Document(
            page_content="Version control systems like Git help developers track changes in code, collaborate on projects, and manage different versions of software.",
            metadata={"topic": "Programming", "subtopic": "DevOps", "difficulty": "Beginner", "type": "Tools"}
        ),
        Document(
            page_content="Cloud computing provides on-demand access to computing resources including servers, storage, databases, and software over the internet.",
            metadata={"topic": "Technology", "subtopic": "Cloud", "difficulty": "Intermediate", "type": "Infrastructure"}
        ),
        Document(
            page_content="API (Application Programming Interface) design involves creating interfaces that allow different software applications to communicate and share data effectively.",
            metadata={"topic": "Programming", "subtopic": "Architecture", "difficulty": "Intermediate", "type": "Design"}
        )
    ]
    
    print(f"📚 Created knowledge base with {len(knowledge_docs)} documents")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        knowledge_docs,
        embeddings,
        collection_name="multi_vector_knowledge"
    )
    
    # 1. Standard Semantic Search
    print(f"\n🔍 1. Standard Semantic Search:")
    query = "How do I learn programming?"
    
    standard_results = vector_store.similarity_search(query, k=3)
    print(f"Query: '{query}'")
    print("Standard Results:")
    for i, doc in enumerate(standard_results, 1):
        print(f"  {i}. Topic: {doc.metadata['topic']} | {doc.page_content[:80]}...")
    
    # 2. Metadata-Filtered Search
    print(f"\n🏷️  2. Metadata-Filtered Search:")
    
    # Search only in Programming topic
    programming_results = vector_store.similarity_search(
        query, 
        k=3,
        filter={"topic": "Programming"}
    )
    print(f"Filtered Results (Programming only):")
    for i, doc in enumerate(programming_results, 1):
        print(f"  {i}. {doc.metadata['topic']} - {doc.metadata.get('subtopic', 'General')} | {doc.page_content[:80]}...")
    
    # Search by difficulty level
    beginner_results = vector_store.similarity_search(
        "programming concepts for beginners",
        k=3,
        filter={"difficulty": "Beginner"}
    )
    print(f"\nFiltered Results (Beginner level):")
    for i, doc in enumerate(beginner_results, 1):
        print(f"  {i}. {doc.metadata['difficulty']} - {doc.metadata['topic']} | {doc.page_content[:80]}...")
    
    # 3. Multi-Query Search
    print(f"\n🔄 3. Multi-Query Search (Query Expansion):")
    
    related_queries = [
        "programming languages for beginners",
        "how to start coding",
        "best programming practices"
    ]
    
    multi_query_results = {}
    all_results = []
    
    for q in related_queries:
        results = vector_store.similarity_search(q, k=2)
        multi_query_results[q] = results
        all_results.extend(results)
    
    # Remove duplicates and rank by frequency
    seen_content = {}
    for doc in all_results:
        content_key = doc.page_content[:50]  # Use first 50 chars as key
        if content_key not in seen_content:
            seen_content[content_key] = {"doc": doc, "count": 1}
        else:
            seen_content[content_key]["count"] += 1
    
    # Sort by frequency (popularity across queries)
    ranked_results = sorted(seen_content.values(), key=lambda x: x["count"], reverse=True)
    
    print("Multi-Query Aggregated Results:")
    for i, result in enumerate(ranked_results[:3], 1):
        doc = result["doc"]
        count = result["count"]
        print(f"  {i}. (Found in {count} queries) {doc.metadata['topic']} | {doc.page_content[:80]}...")
    
    # 4. Contextual Search with Conversation History
    print(f"\n💬 4. Contextual Search with Conversation History:")
    
    conversation_context = [
        "I'm new to programming and want to learn Python",
        "I'm interested in AI and machine learning applications"
    ]
    
    # Combine context with current query
    contextual_query = " ".join(conversation_context) + " " + query
    
    contextual_results = vector_store.similarity_search(contextual_query, k=3)
    print(f"Contextual Query: '{contextual_query[:100]}...'")
    print("Contextual Results:")
    for i, doc in enumerate(contextual_results, 1):
        print(f"  {i}. {doc.metadata['topic']} ({doc.metadata['difficulty']}) | {doc.page_content[:80]}...")
    
    print("\n" + "="*60 + "\n")

def hybrid_search_example():
    """Demonstrate hybrid search combining dense and sparse retrieval"""
    print("=== Hybrid Search (Dense + Sparse) ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Technical documents with specific terminology
    tech_docs = [
        Document(
            page_content="FastAPI is a modern web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides automatic data validation, serialization, and documentation.",
            metadata={"framework": "FastAPI", "language": "Python", "type": "Web Framework"}
        ),
        Document(
            page_content="React is a JavaScript library for building user interfaces, particularly single-page applications. It uses a component-based architecture and virtual DOM for efficient rendering.",
            metadata={"framework": "React", "language": "JavaScript", "type": "Frontend Library"}
        ),
        Document(
            page_content="PostgreSQL is an advanced open-source relational database management system that supports SQL and JSON querying. It offers ACID compliance and extensibility.",
            metadata={"database": "PostgreSQL", "type": "RDBMS", "features": "SQL, JSON"}
        ),
        Document(
            page_content="Docker containers provide lightweight virtualization by packaging applications with their dependencies. This ensures consistent deployment across different environments.",
            metadata={"technology": "Docker", "type": "Containerization", "benefit": "Portability"}
        ),
        Document(
            page_content="Kubernetes orchestrates containerized applications across clusters of machines. It handles deployment, scaling, service discovery, and load balancing automatically.",
            metadata={"technology": "Kubernetes", "type": "Orchestration", "scope": "Cluster Management"}
        ),
        Document(
            page_content="TensorFlow is an open-source machine learning framework developed by Google. It supports deep learning, neural networks, and deployment on various platforms.",
            metadata={"framework": "TensorFlow", "domain": "Machine Learning", "developer": "Google"}
        ),
        Document(
            page_content="Redis is an in-memory data structure store used as a database, cache, and message broker. It supports various data types like strings, hashes, lists, and sets.",
            metadata={"database": "Redis", "type": "In-Memory", "use_cases": "Cache, Database, Broker"}
        ),
        Document(
            page_content="GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need. It provides a complete description of API data.",
            metadata={"technology": "GraphQL", "type": "Query Language", "benefit": "Flexible Data Fetching"}
        )
    ]
    
    print(f"📚 Created technical knowledge base with {len(tech_docs)} documents")
    
    # 1. Create Dense Vector Retriever (Semantic Search)
    print(f"\n🔄 Setting up retrievers...")
    
    vector_store = FAISS.from_documents(tech_docs, embeddings)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 2. Create Sparse Retriever (BM25 - Keyword Based)
    sparse_retriever = BM25Retriever.from_documents(tech_docs)
    sparse_retriever.k = 4
    
    # 3. Create Ensemble Retriever (Hybrid)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]  # 60% semantic, 40% keyword
    )
    
    print("✅ Retrievers configured: Dense (60%) + Sparse (40%)")
    
    # Test different types of queries
    test_queries = [
        {
            "query": "web framework for building APIs",
            "type": "General Semantic",
            "expected": "Should find FastAPI and React"
        },
        {
            "query": "PostgreSQL database JSON",
            "type": "Specific Keywords",
            "expected": "Should prioritize PostgreSQL with keyword match"
        },
        {
            "query": "container deployment orchestration",
            "type": "Conceptual",
            "expected": "Should find Docker and Kubernetes"
        },
        {
            "query": "TensorFlow machine learning Google",
            "type": "Mixed Keywords + Semantic",
            "expected": "Should find TensorFlow with high relevance"
        }
    ]
    
    print(f"\n🔍 Testing hybrid search performance:")
    
    for test in test_queries:
        query = test["query"]
        query_type = test["type"]
        
        print(f"\n📋 {query_type} Query: '{query}'")
        print(f"Expected: {test['expected']}")
        print("-" * 50)
        
        # Dense retrieval (semantic only)
        dense_start = time.time()
        dense_results = dense_retriever.get_relevant_documents(query)
        dense_time = time.time() - dense_start
        
        print(f"🎯 Dense Retrieval ({dense_time*1000:.1f}ms):")
        for i, doc in enumerate(dense_results[:3], 1):
            framework = doc.metadata.get('framework', doc.metadata.get('technology', doc.metadata.get('database', 'Unknown')))
            print(f"  {i}. {framework} | {doc.page_content[:60]}...")
        
        # Sparse retrieval (keyword only)
        sparse_start = time.time()
        sparse_results = sparse_retriever.get_relevant_documents(query)
        sparse_time = time.time() - sparse_start
        
        print(f"\n🔤 Sparse Retrieval ({sparse_time*1000:.1f}ms):")
        for i, doc in enumerate(sparse_results[:3], 1):
            framework = doc.metadata.get('framework', doc.metadata.get('technology', doc.metadata.get('database', 'Unknown')))
            print(f"  {i}. {framework} | {doc.page_content[:60]}...")
        
        # Hybrid retrieval
        hybrid_start = time.time()
        hybrid_results = ensemble_retriever.get_relevant_documents(query)
        hybrid_time = time.time() - hybrid_start
        
        print(f"\n🔄 Hybrid Retrieval ({hybrid_time*1000:.1f}ms):")
        for i, doc in enumerate(hybrid_results[:3], 1):
            framework = doc.metadata.get('framework', doc.metadata.get('technology', doc.metadata.get('database', 'Unknown')))
            print(f"  {i}. {framework} | {doc.page_content[:60]}...")
    
    # Performance comparison
    print(f"\n📊 Retrieval Method Comparison:")
    print(f"{'Method':<15} {'Strengths':<30} {'Best For'}")
    print("-" * 70)
    print(f"{'Dense':<15} {'Semantic understanding':<30} {'Conceptual queries'}")
    print(f"{'Sparse':<15} {'Exact keyword matching':<30} {'Technical terms'}")
    print(f"{'Hybrid':<15} {'Best of both worlds':<30} {'General purpose'}")
    
    print("\n" + "="*60 + "\n")

def query_expansion_and_refinement_example():
    """Demonstrate query expansion and refinement techniques"""
    print("=== Query Expansion and Refinement ===")
    
    # Initialize LLM and embeddings using configuration helpers
    llm = create_azure_chat_openai(temperature=0.3)
    embeddings = create_azure_openai_embeddings()
    
    # Domain-specific knowledge base
    domain_docs = [
        Document(
            page_content="Agile methodology emphasizes iterative development, customer collaboration, and responding to change. Scrum is a popular agile framework with sprints, daily standups, and retrospectives.",
            metadata={"domain": "Project Management", "methodology": "Agile", "framework": "Scrum"}
        ),
        Document(
            page_content="DevOps practices integrate development and operations teams to improve collaboration and productivity. CI/CD pipelines automate testing and deployment processes.",
            metadata={"domain": "Software Engineering", "practice": "DevOps", "automation": "CI/CD"}
        ),
        Document(
            page_content="Microservices architecture breaks applications into small, independent services that communicate via APIs. This approach improves scalability and maintainability.",
            metadata={"domain": "Software Architecture", "pattern": "Microservices", "benefit": "Scalability"}
        ),
        Document(
            page_content="Test-Driven Development (TDD) involves writing tests before implementing functionality. This practice improves code quality and reduces debugging time.",
            metadata={"domain": "Software Engineering", "practice": "TDD", "benefit": "Quality"}
        ),
        Document(
            page_content="Cloud-native applications are designed specifically for cloud environments. They leverage containerization, orchestration, and cloud services for optimal performance.",
            metadata={"domain": "Cloud Computing", "approach": "Cloud-Native", "technology": "Containers"}
        ),
        Document(
            page_content="API-first design prioritizes creating well-defined APIs before building applications. This approach improves integration and enables better collaboration between teams.",
            metadata={"domain": "Software Architecture", "approach": "API-First", "benefit": "Integration"}
        )
    ]
    
    print(f"📚 Created domain knowledge base with {len(domain_docs)} documents")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        domain_docs,
        embeddings,
        collection_name="domain_knowledge"
    )
    
    def expand_query_with_llm(original_query: str) -> List[str]:
        """Use LLM to expand query with related terms"""
        expansion_prompt = f"""
        Given the following query, generate 3-4 related queries that would help find more comprehensive information on the topic. 
        Focus on synonyms, related concepts, and different ways to phrase the question.
        
        Original query: "{original_query}"
        
        Generate related queries (one per line):
        """
        
        try:
            response = llm.invoke(expansion_prompt)
            expanded_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            return expanded_queries[:4]  # Limit to 4 queries
        except Exception as e:
            print(f"LLM expansion failed: {e}")
            return [original_query]
    
    def extract_keywords(query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple keyword extraction
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords
    
    # Test query expansion
    original_queries = [
        "agile project management",
        "continuous integration deployment",
        "microservices best practices"
    ]
    
    print(f"\n🔄 Testing Query Expansion:")
    
    for original_query in original_queries:
        print(f"\n📋 Original Query: '{original_query}'")
        
        # 1. Keyword extraction
        keywords = extract_keywords(original_query)
        print(f"🔑 Extracted Keywords: {', '.join(keywords)}")
        
        # 2. LLM-based expansion
        print(f"🤖 LLM Expanding query...")
        expanded_queries = expand_query_with_llm(original_query)
        print(f"📝 Expanded Queries:")
        for i, exp_q in enumerate(expanded_queries, 1):
            print(f"  {i}. {exp_q}")
        
        # 3. Multi-query search
        all_queries = [original_query] + expanded_queries
        
        # Collect results from all queries
        all_results = []
        query_scores = defaultdict(list)
        
        for query in all_queries:
            results = vector_store.similarity_search_with_score(query, k=3)
            for doc, score in results:
                content_key = doc.page_content[:50]  # Use content as key
                all_results.append((doc, score))
                query_scores[content_key].append(score)
        
        # Aggregate scores (lower is better for distance scores)
        doc_rankings = {}
        seen_docs = {}
        
        for doc, score in all_results:
            content_key = doc.page_content[:50]
            if content_key not in seen_docs:
                seen_docs[content_key] = doc
                doc_rankings[content_key] = []
            doc_rankings[content_key].append(score)
        
        # Calculate average scores and rank
        final_rankings = []
        for content_key, scores in doc_rankings.items():
            avg_score = sum(scores) / len(scores)
            doc = seen_docs[content_key]
            final_rankings.append((doc, avg_score, len(scores)))  # (doc, avg_score, frequency)
        
        # Sort by average score (lower is better)
        final_rankings.sort(key=lambda x: x[1])
        
        print(f"\n🏆 Aggregated Results (Top 3):")
        for i, (doc, avg_score, frequency) in enumerate(final_rankings[:3], 1):
            print(f"  {i}. Score: {avg_score:.4f} (found in {frequency} queries)")
            print(f"     Domain: {doc.metadata.get('domain', 'Unknown')}")
            print(f"     Content: {doc.page_content[:80]}...")
        
        # Compare with single query
        single_query_results = vector_store.similarity_search_with_score(original_query, k=3)
        print(f"\n🔍 Single Query Results for comparison:")
        for i, (doc, score) in enumerate(single_query_results, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Domain: {doc.metadata.get('domain', 'Unknown')}")
            print(f"     Content: {doc.page_content[:80]}...")
    
    print("\n" + "="*60 + "\n")

def similarity_threshold_optimization_example():
    """Demonstrate similarity threshold optimization for quality control"""
    print("=== Similarity Threshold Optimization ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Mixed quality knowledge base (some relevant, some not)
    mixed_docs = [
        Document(page_content="Python programming language syntax and best practices for writing clean, maintainable code", metadata={"quality": "high", "relevance": "high"}),
        Document(page_content="Machine learning algorithms including supervised, unsupervised, and reinforcement learning approaches", metadata={"quality": "high", "relevance": "high"}),
        Document(page_content="Web development frameworks for building scalable applications with modern technologies", metadata={"quality": "high", "relevance": "medium"}),
        Document(page_content="Data structures and algorithms are fundamental concepts in computer science and programming", metadata={"quality": "high", "relevance": "high"}),
        Document(page_content="The weather today is sunny with a chance of rain in the afternoon", metadata={"quality": "low", "relevance": "none"}),
        Document(page_content="Database design principles for relational and NoSQL database systems", metadata={"quality": "high", "relevance": "medium"}),
        Document(page_content="Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor", metadata={"quality": "low", "relevance": "none"}),
        Document(page_content="Software engineering practices including version control, testing, and deployment strategies", metadata={"quality": "high", "relevance": "high"}),
        Document(page_content="Random text about cooking recipes and kitchen utensils for meal preparation", metadata={"quality": "low", "relevance": "none"}),
        Document(page_content="Artificial intelligence and neural networks for pattern recognition and prediction tasks", metadata={"quality": "high", "relevance": "high"})
    ]
    
    print(f"📚 Created mixed quality knowledge base with {len(mixed_docs)} documents")
    
    # Analyze ground truth
    high_quality_docs = [doc for doc in mixed_docs if doc.metadata["quality"] == "high"]
    relevant_docs = [doc for doc in mixed_docs if doc.metadata["relevance"] in ["high", "medium"]]
    
    print(f"📊 Ground Truth Analysis:")
    print(f"   High Quality: {len(high_quality_docs)}/{len(mixed_docs)} documents")
    print(f"   Relevant: {len(relevant_docs)}/{len(mixed_docs)} documents")
    
    # Create vector store
    vector_store = FAISS.from_documents(mixed_docs, embeddings)
    
    # Test different similarity thresholds
    test_query = "programming and software development"
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\n🔍 Testing Query: '{test_query}'")
    print(f"🎯 Optimizing similarity thresholds...")
    
    threshold_analysis = []
    
    for threshold in thresholds:
        # Get results with scores
        all_results = vector_store.similarity_search_with_score(test_query, k=len(mixed_docs))
        
        # Filter by threshold
        filtered_results = [(doc, score) for doc, score in all_results if score <= threshold]
        
        # Analyze quality
        total_returned = len(filtered_results)
        high_quality_returned = sum(1 for doc, _ in filtered_results if doc.metadata["quality"] == "high")
        relevant_returned = sum(1 for doc, _ in filtered_results if doc.metadata["relevance"] in ["high", "medium"])
        
        # Calculate metrics
        precision_quality = high_quality_returned / total_returned if total_returned > 0 else 0
        recall_quality = high_quality_returned / len(high_quality_docs)
        
        precision_relevance = relevant_returned / total_returned if total_returned > 0 else 0
        recall_relevance = relevant_returned / len(relevant_docs)
        
        # F1 scores
        f1_quality = 2 * (precision_quality * recall_quality) / (precision_quality + recall_quality) if (precision_quality + recall_quality) > 0 else 0
        f1_relevance = 2 * (precision_relevance * recall_relevance) / (precision_relevance + recall_relevance) if (precision_relevance + recall_relevance) > 0 else 0
        
        threshold_analysis.append({
            "threshold": threshold,
            "total_returned": total_returned,
            "precision_quality": precision_quality,
            "recall_quality": recall_quality,
            "f1_quality": f1_quality,
            "precision_relevance": precision_relevance,
            "recall_relevance": recall_relevance,
            "f1_relevance": f1_relevance,
            "avg_score": sum(score for _, score in filtered_results) / total_returned if total_returned > 0 else 0
        })
    
    # Display threshold analysis
    print(f"\n📊 Threshold Analysis Results:")
    print(f"{'Threshold':<10} {'Returned':<8} {'Quality F1':<12} {'Relevance F1':<13} {'Avg Score'}")
    print("-" * 65)
    
    for analysis in threshold_analysis:
        print(f"{analysis['threshold']:<10} {analysis['total_returned']:<8} {analysis['f1_quality']:<12.3f} {analysis['f1_relevance']:<13.3f} {analysis['avg_score']:<.3f}")
    
    # Find optimal threshold
    best_quality_threshold = max(threshold_analysis, key=lambda x: x['f1_quality'])
    best_relevance_threshold = max(threshold_analysis, key=lambda x: x['f1_relevance'])
    
    print(f"\n🏆 Optimal Thresholds:")
    print(f"   Best for Quality: {best_quality_threshold['threshold']} (F1: {best_quality_threshold['f1_quality']:.3f})")
    print(f"   Best for Relevance: {best_relevance_threshold['threshold']} (F1: {best_relevance_threshold['f1_relevance']:.3f})")
    
    # Demonstrate adaptive threshold
    print(f"\n🎛️  Adaptive Threshold Demonstration:")
    
    def adaptive_search(query: str, min_results: int = 3, max_results: int = 8, quality_threshold: float = 0.6):
        """Adaptive search with dynamic threshold adjustment"""
        all_results = vector_store.similarity_search_with_score(query, k=len(mixed_docs))
        
        # Start with quality threshold
        filtered_results = [(doc, score) for doc, score in all_results if score <= quality_threshold]
        
        # If too few results, gradually increase threshold
        if len(filtered_results) < min_results:
            for relaxed_threshold in [0.7, 0.8, 0.9, 1.0]:
                filtered_results = [(doc, score) for doc, score in all_results if score <= relaxed_threshold]
                if len(filtered_results) >= min_results:
                    break
        
        # If too many results, limit to max_results with best scores
        if len(filtered_results) > max_results:
            filtered_results = filtered_results[:max_results]
        
        return filtered_results, quality_threshold
    
    adaptive_results, used_threshold = adaptive_search(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Used threshold: {used_threshold}")
    print(f"Returned {len(adaptive_results)} results:")
    
    for i, (doc, score) in enumerate(adaptive_results, 1):
        quality = doc.metadata["quality"]
        relevance = doc.metadata["relevance"]
        quality_indicator = "✅" if quality == "high" else "❌"
        relevance_indicator = "🎯" if relevance in ["high", "medium"] else "❌"
        
        print(f"  {i}. Score: {score:.3f} {quality_indicator} {relevance_indicator}")
        print(f"     Quality: {quality}, Relevance: {relevance}")
        print(f"     Content: {doc.page_content[:60]}...")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Semantic Search and Advanced Retrieval Patterns")
    print("=" * 60)
    
    try:
        # Run examples
        multi_vector_search_example()
        hybrid_search_example()
        query_expansion_and_refinement_example()
        similarity_threshold_optimization_example()
        
        print("🎉 Advanced retrieval examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings")
        print("2. Installed required packages: chromadb, faiss-cpu, rank-bm25")
