"""
Performance Optimization and Scaling Vector Stores

This module demonstrates performance optimization techniques for vector stores:
1. Index optimization strategies
2. Batch processing for large datasets
3. Memory management and efficiency
4. Caching and persistence patterns
5. Scaling strategies for production
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
import numpy as np
import time
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import psutil
import gc
from functools import wraps

load_dotenv()

def measure_performance(func):
    """Decorator to measure function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_diff = end_memory - start_memory
        
        print(f"⏱️  Performance: {execution_time:.2f}s, Memory: {memory_diff:+.1f}MB")
        
        return result
    return wrapper

def index_optimization_example():
    """Demonstrate different indexing strategies and their performance"""
    print("=== Index Optimization Strategies ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Generate large dataset for testing
    def generate_test_documents(count: int) -> List[Document]:
        """Generate synthetic documents for performance testing"""
        topics = [
            "machine learning algorithms and applications",
            "web development frameworks and tools",
            "data science and analytics techniques",
            "cloud computing and infrastructure",
            "artificial intelligence and neural networks",
            "database design and optimization",
            "software engineering best practices",
            "cybersecurity and privacy protection"
        ]
        
        variations = [
            "introduction to", "advanced concepts in", "practical guide to",
            "comprehensive overview of", "best practices for", "modern approaches to",
            "fundamentals of", "expert techniques in"
        ]
        
        documents = []
        for i in range(count):
            topic = topics[i % len(topics)]
            variation = variations[i % len(variations)]
            
            content = f"{variation} {topic}. This document covers important concepts, methodologies, and practical applications in the field. It provides detailed explanations and examples for better understanding."
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "doc_id": i,
                    "topic_id": i % len(topics),
                    "variation_id": i % len(variations),
                    "category": topic.split()[0],
                    "length": len(content)
                }
            ))
        
        return documents
    
    # Test with different dataset sizes
    dataset_sizes = [100, 500, 1000, 2000]
    index_types = ["FAISS", "Chroma"]
    
    performance_results = {}
    
    print(f"🔧 Testing index performance with different dataset sizes...")
    
    for size in dataset_sizes:
        print(f"\n📊 Dataset Size: {size} documents")
        performance_results[size] = {}
        
        # Generate test data
        print(f"   Generating {size} test documents...")
        test_docs = generate_test_documents(size)
        
        for index_type in index_types:
            print(f"\n   Testing {index_type} index:")
            
            # Measure index creation time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            if index_type == "FAISS":
                vector_store = FAISS.from_documents(test_docs, embeddings)
            else:  # Chroma
                vector_store = Chroma.from_documents(
                    test_docs,
                    embeddings,
                    collection_name=f"perf_test_{size}"
                )
            
            creation_time = time.time() - start_time
            creation_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            print(f"     Index creation: {creation_time:.2f}s, Memory: +{creation_memory:.1f}MB")
            
            # Measure search performance
            test_queries = [
                "machine learning algorithms",
                "web development frameworks",
                "data science techniques",
                "cloud computing infrastructure"
            ]
            
            search_times = []
            for query in test_queries:
                search_start = time.time()
                results = vector_store.similarity_search(query, k=5)
                search_time = time.time() - search_start
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            
            print(f"     Average search: {avg_search_time*1000:.1f}ms")
            
            # Store results
            performance_results[size][index_type] = {
                "creation_time": creation_time,
                "creation_memory": creation_memory,
                "avg_search_time": avg_search_time,
                "throughput": size / creation_time  # docs per second
            }
            
            # Cleanup memory
            del vector_store
            gc.collect()
    
    # Performance analysis
    print(f"\n📈 Index Performance Summary:")
    print(f"{'Size':<6} {'Type':<8} {'Creation':<12} {'Search':<10} {'Throughput':<12} {'Memory'}")
    print("-" * 70)
    
    for size in dataset_sizes:
        for index_type in index_types:
            results = performance_results[size][index_type]
            print(f"{size:<6} {index_type:<8} {results['creation_time']:<12.2f} {results['avg_search_time']*1000:<10.1f} {results['throughput']:<12.1f} {results['creation_memory']:<.1f}MB")
    
    # Recommendations
    print(f"\n💡 Performance Recommendations:")
    
    for index_type in index_types:
        large_dataset_perf = performance_results[max(dataset_sizes)][index_type]
        
        if index_type == "FAISS":
            print(f"   🔧 FAISS: Excellent for large datasets, fastest search, higher memory usage")
            print(f"      Best for: Production systems with frequent searches")
        else:
            print(f"   🔧 Chroma: Good balance, persistent storage, moderate performance")
            print(f"      Best for: Development, smaller datasets, persistent storage needs")
    
    print("\n" + "="*60 + "\n")

@measure_performance
def batch_processing_example():
    """Demonstrate efficient batch processing for large datasets"""
    print("=== Batch Processing for Large Datasets ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Generate large dataset
    def generate_large_dataset(count: int) -> List[str]:
        """Generate a large dataset of text for batch processing"""
        base_texts = [
            "Advanced machine learning techniques for data analysis",
            "Modern web development with React and Node.js",
            "Cloud infrastructure design and deployment strategies",
            "Database optimization and performance tuning",
            "Artificial intelligence applications in healthcare",
            "Cybersecurity best practices for enterprise systems",
            "DevOps automation and continuous integration",
            "Mobile app development for iOS and Android"
        ]
        
        dataset = []
        for i in range(count):
            base_text = base_texts[i % len(base_texts)]
            variation = f"Document {i+1}: {base_text} with specific focus on implementation details and practical examples."
            dataset.append(variation)
        
        return dataset
    
    dataset_size = 500
    print(f"📊 Generating dataset with {dataset_size} documents...")
    
    large_dataset = generate_large_dataset(dataset_size)
    
    # 1. Sequential Processing (Baseline)
    print(f"\n🔄 1. Sequential Processing:")
    
    sequential_start = time.time()
    sequential_embeddings = []
    
    for i, text in enumerate(large_dataset):
        if i % 100 == 0:
            print(f"   Processing document {i+1}/{len(large_dataset)}")
        
        embedding = embeddings.embed_query(text)
        sequential_embeddings.append(embedding)
    
    sequential_time = time.time() - sequential_start
    print(f"   Sequential processing: {sequential_time:.2f}s")
    print(f"   Rate: {len(large_dataset)/sequential_time:.1f} docs/second")
    
    # 2. Batch Processing
    print(f"\n📦 2. Batch Processing:")
    
    batch_start = time.time()
    batch_size = 50
    batch_embeddings = []
    
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(large_dataset)-1)//batch_size + 1}")
        
        batch_result = embeddings.embed_documents(batch)
        batch_embeddings.extend(batch_result)
    
    batch_time = time.time() - batch_start
    print(f"   Batch processing: {batch_time:.2f}s")
    print(f"   Rate: {len(large_dataset)/batch_time:.1f} docs/second")
    print(f"   Speedup: {sequential_time/batch_time:.1f}x faster")
    
    # 3. Parallel Batch Processing
    print(f"\n⚡ 3. Parallel Batch Processing:")
    
    def process_batch(batch_data):
        """Process a batch of documents"""
        batch_texts, batch_id = batch_data
        return batch_id, embeddings.embed_documents(batch_texts)
    
    parallel_start = time.time()
    
    # Create batches
    batches = []
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i+batch_size]
        batches.append((batch, i//batch_size))
    
    # Process batches in parallel
    parallel_embeddings = [None] * len(batches)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch = {executor.submit(process_batch, batch_data): batch_data for batch_data in batches}
        
        for future in as_completed(future_to_batch):
            batch_id, batch_result = future.result()
            parallel_embeddings[batch_id] = batch_result
            print(f"   Completed batch {batch_id + 1}/{len(batches)}")
    
    # Flatten results
    flat_parallel_embeddings = []
    for batch_result in parallel_embeddings:
        flat_parallel_embeddings.extend(batch_result)
    
    parallel_time = time.time() - parallel_start
    print(f"   Parallel processing: {parallel_time:.2f}s")
    print(f"   Rate: {len(large_dataset)/parallel_time:.1f} docs/second")
    print(f"   Speedup over sequential: {sequential_time/parallel_time:.1f}x faster")
    print(f"   Speedup over batch: {batch_time/parallel_time:.1f}x faster")
    
    # 4. Create vector store with optimized batch
    print(f"\n🏗️  4. Creating Vector Store with Optimized Batching:")
    
    # Convert to documents
    documents = [
        Document(
            page_content=text,
            metadata={"doc_id": i, "batch_id": i // batch_size}
        )
        for i, text in enumerate(large_dataset)
    ]
    
    store_start = time.time()
    
    # Use the fastest embedding approach (parallel batch)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    store_time = time.time() - store_start
    print(f"   Vector store creation: {store_time:.2f}s")
    
    # Test search performance
    test_queries = [
        "machine learning algorithms",
        "web development frameworks",
        "cloud infrastructure",
        "database optimization"
    ]
    
    search_times = []
    for query in test_queries:
        search_start = time.time()
        results = vector_store.similarity_search(query, k=10)
        search_time = time.time() - search_start
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"   Average search time: {avg_search_time*1000:.1f}ms")
    
    # Performance summary
    print(f"\n📊 Batch Processing Performance Summary:")
    print(f"   Dataset size: {dataset_size} documents")
    print(f"   Sequential: {sequential_time:.2f}s ({len(large_dataset)/sequential_time:.1f} docs/sec)")
    print(f"   Batch: {batch_time:.2f}s ({len(large_dataset)/batch_time:.1f} docs/sec)")
    print(f"   Parallel: {parallel_time:.2f}s ({len(large_dataset)/parallel_time:.1f} docs/sec)")
    print(f"   Best speedup: {sequential_time/parallel_time:.1f}x improvement")
    
    print("\n" + "="*60 + "\n")

def caching_and_persistence_example():
    """Demonstrate caching and persistence strategies"""
    print("=== Caching and Persistence Strategies ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Sample documents
    sample_docs = [
        Document(
            page_content="Python is a versatile programming language used for web development, data science, and automation.",
            metadata={"category": "Programming", "language": "Python", "id": 1}
        ),
        Document(
            page_content="Machine learning enables computers to learn patterns from data without explicit programming.",
            metadata={"category": "AI/ML", "topic": "Machine Learning", "id": 2}
        ),
        Document(
            page_content="React is a popular JavaScript library for building user interfaces and single-page applications.",
            metadata={"category": "Web Development", "framework": "React", "id": 3}
        ),
        Document(
            page_content="Docker containers provide lightweight virtualization for consistent application deployment.",
            metadata={"category": "DevOps", "technology": "Docker", "id": 4}
        ),
        Document(
            page_content="PostgreSQL is a powerful relational database with advanced features and SQL compliance.",
            metadata={"category": "Database", "type": "Relational", "id": 5}
        )
    ]
    
    print(f"📚 Working with {len(sample_docs)} sample documents")
    
    # 1. Embedding Caching
    print(f"\n💾 1. Embedding Caching Strategy:")
    
    class EmbeddingCache:
        def __init__(self, cache_file="embedding_cache.pkl"):
            self.cache_file = cache_file
            self.cache = self._load_cache()
        
        def _load_cache(self):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return {}
        
        def _save_cache(self):
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        
        def get_embedding(self, text: str, embeddings_model):
            text_hash = str(hash(text))
            
            if text_hash in self.cache:
                print(f"   Cache hit for text: {text[:50]}...")
                return self.cache[text_hash]
            
            print(f"   Cache miss, generating embedding for: {text[:50]}...")
            embedding = embeddings_model.embed_query(text)
            self.cache[text_hash] = embedding
            self._save_cache()
            
            return embedding
        
        def cache_size(self):
            return len(self.cache)
    
    # Test caching
    cache = EmbeddingCache()
    print(f"   Current cache size: {cache.cache_size()} embeddings")
    
    # First pass - should generate embeddings
    print(f"\n   First pass (cache misses expected):")
    cached_embeddings = []
    for doc in sample_docs:
        embedding = cache.get_embedding(doc.page_content, embeddings)
        cached_embeddings.append(embedding)
    
    # Second pass - should use cache
    print(f"\n   Second pass (cache hits expected):")
    for doc in sample_docs:
        embedding = cache.get_embedding(doc.page_content, embeddings)
    
    print(f"   Final cache size: {cache.cache_size()} embeddings")
    
    # 2. Vector Store Persistence
    print(f"\n💾 2. Vector Store Persistence:")
    
    # Create and save FAISS index
    print(f"   Creating FAISS vector store...")
    faiss_store = FAISS.from_documents(sample_docs, embeddings)
    
    # Save to disk
    print(f"   Saving FAISS index to disk...")
    faiss_store.save_local("faiss_persistent_store")
    
    # Load from disk
    print(f"   Loading FAISS index from disk...")
    loaded_faiss_store = FAISS.load_local("faiss_persistent_store", embeddings, allow_dangerous_deserialization=True)
    
    # Verify loaded store works
    test_results = loaded_faiss_store.similarity_search("programming languages", k=2)
    print(f"   Loaded store test: Found {len(test_results)} results")
    for i, doc in enumerate(test_results, 1):
        print(f"     {i}. {doc.metadata['category']}: {doc.page_content[:50]}...")
    
    # Create persistent Chroma store
    print(f"\n   Creating persistent Chroma store...")
    chroma_store = Chroma.from_documents(
        sample_docs,
        embeddings,
        collection_name="persistent_collection",
        persist_directory="./chroma_persistent"
    )
    
    # Simulate restart - load existing collection
    print(f"   Loading existing Chroma collection...")
    loaded_chroma_store = Chroma(
        collection_name="persistent_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_persistent"
    )
    
    # Test loaded collection
    chroma_results = loaded_chroma_store.similarity_search("web development", k=2)
    print(f"   Loaded Chroma test: Found {len(chroma_results)} results")
    for i, doc in enumerate(chroma_results, 1):
        print(f"     {i}. {doc.metadata['category']}: {doc.page_content[:50]}...")
    
    # 3. Query Result Caching
    print(f"\n💾 3. Query Result Caching:")
    
    class QueryCache:
        def __init__(self):
            self.query_cache = {}
        
        def get_cached_results(self, query: str, vector_store, k: int = 5):
            cache_key = f"{query}_{k}"
            
            if cache_key in self.query_cache:
                print(f"   Query cache hit: '{query}'")
                return self.query_cache[cache_key]
            
            print(f"   Query cache miss, searching: '{query}'")
            results = vector_store.similarity_search(query, k=k)
            self.query_cache[cache_key] = results
            
            return results
        
        def cache_stats(self):
            return {"cached_queries": len(self.query_cache)}
    
    query_cache = QueryCache()
    
    test_queries = [
        "programming languages",
        "machine learning algorithms",
        "web development frameworks",
        "programming languages"  # Repeat to test cache
    ]
    
    for query in test_queries:
        results = query_cache.get_cached_results(query, loaded_faiss_store, k=2)
        print(f"     Results for '{query}': {len(results)} documents")
    
    print(f"   Query cache stats: {query_cache.cache_stats()}")
    
    # 4. Memory Management
    print(f"\n🧠 4. Memory Management:")
    
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    initial_memory = get_memory_usage()
    print(f"   Initial memory usage: {initial_memory:.1f}MB")
    
    # Create multiple vector stores to show memory growth
    stores = []
    for i in range(3):
        store = FAISS.from_documents(sample_docs, embeddings)
        stores.append(store)
        current_memory = get_memory_usage()
        print(f"   After creating store {i+1}: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB)")
    
    # Cleanup
    print(f"   Cleaning up stores...")
    for store in stores:
        del store
    
    gc.collect()  # Force garbage collection
    
    final_memory = get_memory_usage()
    print(f"   After cleanup: {final_memory:.1f}MB")
    print(f"   Memory recovered: {(initial_memory + (current_memory - initial_memory)) - final_memory:.1f}MB")
    
    print("\n" + "="*60 + "\n")

def scaling_strategies_example():
    """Demonstrate scaling strategies for production"""
    print("=== Scaling Strategies for Production ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # 1. Partitioning Strategy
    print(f"\n📂 1. Document Partitioning Strategy:")
    
    # Create documents with different categories
    partitioned_docs = []
    categories = ["Technology", "Science", "Business", "Healthcare", "Education"]
    
    for category in categories:
        for i in range(10):  # 10 docs per category
            doc = Document(
                page_content=f"This is a {category.lower()} document about various topics and concepts relevant to the {category.lower()} domain. Document number {i+1}.",
                metadata={
                    "category": category,
                    "doc_id": f"{category}_{i+1}",
                    "partition": category.lower()
                }
            )
            partitioned_docs.append(doc)
    
    print(f"   Created {len(partitioned_docs)} documents across {len(categories)} categories")
    
    # Create separate stores per partition
    partition_stores = {}
    
    for category in categories:
        category_docs = [doc for doc in partitioned_docs if doc.metadata["category"] == category]
        partition_stores[category] = FAISS.from_documents(category_docs, embeddings)
        print(f"   Created {category} partition: {len(category_docs)} documents")
    
    # Distributed search across partitions
    def distributed_search(query: str, partitions: Dict[str, Any], k_per_partition: int = 2):
        """Search across multiple partitions and aggregate results"""
        all_results = []
        
        for partition_name, store in partitions.items():
            partition_results = store.similarity_search_with_score(query, k=k_per_partition)
            
            # Add partition info to results
            for doc, score in partition_results:
                doc.metadata["search_partition"] = partition_name
                all_results.append((doc, score))
        
        # Sort by score (lower is better for distance)
        all_results.sort(key=lambda x: x[1])
        
        return all_results
    
    test_query = "technology and innovation"
    distributed_results = distributed_search(test_query, partition_stores, k_per_partition=2)
    
    print(f"\n   Distributed search for '{test_query}':")
    for i, (doc, score) in enumerate(distributed_results[:5], 1):
        category = doc.metadata["category"]
        partition = doc.metadata["search_partition"]
        print(f"     {i}. Score: {score:.3f}, Category: {category}, Partition: {partition}")
    
    # 2. Hierarchical Search
    print(f"\n🌳 2. Hierarchical Search Strategy:")
    
    # Create summary documents for each category
    category_summaries = {
        "Technology": "Technology documents cover software, hardware, programming, and digital innovations.",
        "Science": "Science documents include research, discoveries, methodologies, and scientific principles.",
        "Business": "Business documents focus on strategy, management, economics, and commercial activities.",
        "Healthcare": "Healthcare documents discuss medical practices, treatments, health policies, and patient care.",
        "Education": "Education documents cover teaching methods, learning theories, and educational systems."
    }
    
    summary_docs = [
        Document(
            page_content=summary,
            metadata={"type": "summary", "category": category, "doc_count": 10}
        )
        for category, summary in category_summaries.items()
    ]
    
    # Create summary index
    summary_store = FAISS.from_documents(summary_docs, embeddings)
    
    def hierarchical_search(query: str, summary_store, detail_stores: Dict[str, Any], k_summaries: int = 2, k_details: int = 3):
        """Hierarchical search: first find relevant categories, then search within them"""
        
        # Step 1: Find relevant categories
        summary_results = summary_store.similarity_search(query, k=k_summaries)
        relevant_categories = [doc.metadata["category"] for doc in summary_results]
        
        print(f"   Step 1: Found relevant categories: {relevant_categories}")
        
        # Step 2: Search within relevant categories
        detailed_results = []
        for category in relevant_categories:
            if category in detail_stores:
                category_results = detail_stores[category].similarity_search_with_score(query, k=k_details)
                detailed_results.extend(category_results)
        
        # Sort by score
        detailed_results.sort(key=lambda x: x[1])
        
        return detailed_results, relevant_categories
    
    hierarchical_results, found_categories = hierarchical_search(
        "educational technology innovations",
        summary_store,
        partition_stores,
        k_summaries=2,
        k_details=2
    )
    
    print(f"   Hierarchical search results:")
    print(f"   Relevant categories: {found_categories}")
    for i, (doc, score) in enumerate(hierarchical_results[:3], 1):
        category = doc.metadata["category"]
        print(f"     {i}. Score: {score:.3f}, Category: {category}")
        print(f"        Content: {doc.page_content[:60]}...")
    
    # 3. Load Balancing Simulation
    print(f"\n⚖️  3. Load Balancing Simulation:")
    
    class LoadBalancer:
        def __init__(self, stores: Dict[str, Any]):
            self.stores = stores
            self.store_loads = {name: 0 for name in stores.keys()}
            self.total_requests = 0
        
        def get_least_loaded_store(self):
            return min(self.store_loads.items(), key=lambda x: x[1])
        
        def route_query(self, query: str, strategy: str = "round_robin"):
            self.total_requests += 1
            
            if strategy == "round_robin":
                store_names = list(self.stores.keys())
                selected_store = store_names[(self.total_requests - 1) % len(store_names)]
            
            elif strategy == "least_loaded":
                selected_store, _ = self.get_least_loaded_store()
            
            else:  # hash-based
                import hashlib
                hash_value = int(hashlib.md5(query.encode()).hexdigest(), 16)
                store_names = list(self.stores.keys())
                selected_store = store_names[hash_value % len(store_names)]
            
            # Simulate load
            self.store_loads[selected_store] += 1
            
            # Perform search
            results = self.stores[selected_store].similarity_search(query, k=2)
            
            # Simulate load decrease after processing
            self.store_loads[selected_store] -= 1
            
            return selected_store, results
    
    # Create load balancer with partition stores
    load_balancer = LoadBalancer(partition_stores)
    
    test_queries_lb = [
        "technology innovation",
        "scientific research",
        "business strategy",
        "healthcare solutions",
        "educational methods"
    ]
    
    print(f"   Testing load balancing strategies:")
    
    for strategy in ["round_robin", "least_loaded", "hash_based"]:
        print(f"\n   Strategy: {strategy}")
        
        load_balancer.store_loads = {name: 0 for name in partition_stores.keys()}
        load_balancer.total_requests = 0
        
        for query in test_queries_lb:
            selected_store, results = load_balancer.route_query(query, strategy)
            print(f"     Query: '{query}' → {selected_store}")
        
        print(f"     Final loads: {load_balancer.store_loads}")
        load_variance = max(load_balancer.store_loads.values()) - min(load_balancer.store_loads.values())
        print(f"     Load variance: {load_variance}")
    
    # 4. Performance Monitoring
    print(f"\n📊 4. Performance Monitoring:")
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {
                "total_queries": 0,
                "total_time": 0,
                "avg_response_time": 0,
                "query_times": [],
                "error_count": 0
            }
        
        def log_query(self, query: str, response_time: float, success: bool = True):
            self.metrics["total_queries"] += 1
            
            if success:
                self.metrics["total_time"] += response_time
                self.metrics["query_times"].append(response_time)
                self.metrics["avg_response_time"] = self.metrics["total_time"] / self.metrics["total_queries"]
            else:
                self.metrics["error_count"] += 1
        
        def get_stats(self):
            if self.metrics["query_times"]:
                times = self.metrics["query_times"]
                return {
                    "total_queries": self.metrics["total_queries"],
                    "avg_response_time": self.metrics["avg_response_time"],
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "error_rate": self.metrics["error_count"] / self.metrics["total_queries"],
                    "success_rate": 1 - (self.metrics["error_count"] / self.metrics["total_queries"])
                }
            return {"total_queries": 0}
    
    monitor = PerformanceMonitor()
    
    # Simulate monitored queries
    for query in test_queries_lb:
        start_time = time.time()
        
        try:
            results = partition_stores["Technology"].similarity_search(query, k=3)
            response_time = time.time() - start_time
            monitor.log_query(query, response_time, success=True)
        except Exception as e:
            response_time = time.time() - start_time
            monitor.log_query(query, response_time, success=False)
    
    stats = monitor.get_stats()
    print(f"   Performance Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Performance Optimization and Scaling Vector Stores")
    print("=" * 60)
    
    try:
        # Run optimization examples
        index_optimization_example()
        batch_processing_example()
        caching_and_persistence_example()
        scaling_strategies_example()
        
        print("🎉 Performance optimization examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings")
        print("2. Installed required packages: chromadb, faiss-cpu, psutil")
