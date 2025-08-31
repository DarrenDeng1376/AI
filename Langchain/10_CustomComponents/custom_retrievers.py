"""
Custom Retriever Implementations with Azure Integration

This module demonstrates how to build custom retrievers for LangChain
applications, including Azure-backed retrievers and advanced retrieval patterns.

Key concepts covered:
1. Custom retriever base classes
2. Azure Cognitive Search integration
3. Hybrid search (vector + keyword)
4. Multi-modal retrieval
5. Contextual re-ranking
6. Retrieval optimization and caching
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_openai_embeddings
from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3


# ============================================================================
# 1. AZURE COGNITIVE SEARCH RETRIEVER
# ============================================================================

class AzureCognitiveSearchRetriever(BaseRetriever):
    """
    Custom retriever that integrates with Azure Cognitive Search
    for enterprise-grade document retrieval
    """
    
    search_service_name: str = Field(description="Azure Search service name")
    search_index_name: str = Field(description="Search index name")
    api_key: str = Field(description="Azure Search API key")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    search_mode: str = Field(default="hybrid", description="Search mode: semantic, vector, keyword, or hybrid")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = create_azure_openai_embeddings()
        self._init_search_client()
    
    def _init_search_client(self):
        """Initialize Azure Search client"""
        try:
            # Simulated Azure Search client for demo
            # In production, use: from azure.search.documents import SearchClient
            print(f"🔍 Initializing Azure Search client for service: {self.search_service_name}")
            self.search_client = None  # Placeholder
        except Exception as e:
            print(f"⚠️ Azure Search client initialization failed: {e}")
            self.search_client = None
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from Azure Cognitive Search"""
        try:
            if self.search_mode == "vector":
                return self._vector_search(query)
            elif self.search_mode == "keyword":
                return self._keyword_search(query)
            elif self.search_mode == "semantic":
                return self._semantic_search(query)
            else:  # hybrid
                return self._hybrid_search(query)
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return self._fallback_search(query)
    
    def _vector_search(self, query: str) -> List[Document]:
        """Perform vector similarity search"""
        print(f"🔍 Performing vector search for: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Simulated vector search results
        # In production, use Azure Search vector capabilities
        mock_results = [
            {
                "content": f"This document discusses {query} in detail with technical implementation guides and best practices.",
                "metadata": {"source": "technical_guide.pdf", "score": 0.85, "page": 1},
                "id": "doc_1"
            },
            {
                "content": f"Advanced concepts related to {query} including troubleshooting and optimization strategies.",
                "metadata": {"source": "advanced_guide.pdf", "score": 0.78, "page": 5},
                "id": "doc_2"
            },
            {
                "content": f"Case study: Implementing {query} in enterprise environments with Azure services.",
                "metadata": {"source": "case_study.pdf", "score": 0.72, "page": 3},
                "id": "doc_3"
            }
        ]
        
        return [
            Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            for result in mock_results[:self.top_k]
        ]
    
    def _keyword_search(self, query: str) -> List[Document]:
        """Perform traditional keyword-based search"""
        print(f"🔍 Performing keyword search for: {query[:50]}...")
        
        # Simulated keyword search
        mock_results = [
            {
                "content": f"Documentation containing exact keywords: {query}. This provides comprehensive coverage of the topic.",
                "metadata": {"source": "keyword_doc.pdf", "score": 0.9, "page": 2},
                "id": "kw_doc_1"
            },
            {
                "content": f"Reference material with terminology related to {query} and practical examples.",
                "metadata": {"source": "reference.pdf", "score": 0.75, "page": 1},
                "id": "kw_doc_2"
            }
        ]
        
        return [
            Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            for result in mock_results[:self.top_k]
        ]
    
    def _semantic_search(self, query: str) -> List[Document]:
        """Perform semantic search using Azure's semantic ranking"""
        print(f"🔍 Performing semantic search for: {query[:50]}...")
        
        # Simulated semantic search with better understanding
        mock_results = [
            {
                "content": f"Comprehensive guide that semantically relates to {query}, covering underlying concepts and principles.",
                "metadata": {"source": "semantic_guide.pdf", "score": 0.92, "semantic_score": 0.88, "page": 1},
                "id": "sem_doc_1"
            },
            {
                "content": f"Related topics and conceptual frameworks that align with {query} from a semantic perspective.",
                "metadata": {"source": "concepts.pdf", "score": 0.81, "semantic_score": 0.82, "page": 4},
                "id": "sem_doc_2"
            }
        ]
        
        return [
            Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            for result in mock_results[:self.top_k]
        ]
    
    def _hybrid_search(self, query: str) -> List[Document]:
        """Perform hybrid search combining vector, keyword, and semantic"""
        print(f"🔍 Performing hybrid search for: {query[:50]}...")
        
        # Get results from different search methods
        vector_docs = self._vector_search(query)
        keyword_docs = self._keyword_search(query)
        semantic_docs = self._semantic_search(query)
        
        # Combine and deduplicate results
        all_docs = {}
        
        # Add vector results with high weight for semantic similarity
        for doc in vector_docs:
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            all_docs[doc_id] = {
                "document": doc,
                "vector_score": doc.metadata.get("score", 0.0),
                "keyword_score": 0.0,
                "semantic_score": 0.0
            }
        
        # Add keyword results
        for doc in keyword_docs:
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            if doc_id in all_docs:
                all_docs[doc_id]["keyword_score"] = doc.metadata.get("score", 0.0)
            else:
                all_docs[doc_id] = {
                    "document": doc,
                    "vector_score": 0.0,
                    "keyword_score": doc.metadata.get("score", 0.0),
                    "semantic_score": 0.0
                }
        
        # Add semantic results
        for doc in semantic_docs:
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            if doc_id in all_docs:
                all_docs[doc_id]["semantic_score"] = doc.metadata.get("semantic_score", 0.0)
            else:
                all_docs[doc_id] = {
                    "document": doc,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_score": doc.metadata.get("semantic_score", 0.0)
                }
        
        # Calculate hybrid scores
        for doc_id, doc_data in all_docs.items():
            # Weighted combination of scores
            hybrid_score = (
                0.4 * doc_data["vector_score"] +
                0.3 * doc_data["keyword_score"] +
                0.3 * doc_data["semantic_score"]
            )
            
            # Update document metadata
            doc_data["document"].metadata["hybrid_score"] = hybrid_score
            doc_data["document"].metadata["vector_score"] = doc_data["vector_score"]
            doc_data["document"].metadata["keyword_score"] = doc_data["keyword_score"]
            doc_data["document"].metadata["semantic_score"] = doc_data["semantic_score"]
        
        # Sort by hybrid score and return top k
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["document"].metadata["hybrid_score"],
            reverse=True
        )
        
        return [doc_data["document"] for doc_data in sorted_docs[:self.top_k]]
    
    def _fallback_search(self, query: str) -> List[Document]:
        """Fallback search when main search fails"""
        print(f"🔄 Using fallback search for: {query[:50]}...")
        
        return [
            Document(
                page_content=f"Fallback result for query: {query}. This is a basic result when the main search service is unavailable.",
                metadata={"source": "fallback", "score": 0.5, "type": "fallback"}
            )
        ]


# ============================================================================
# 2. CACHED RETRIEVER WITH INTELLIGENT REFRESH
# ============================================================================

class CachedRetriever(BaseRetriever):
    """
    Retriever with intelligent caching and refresh strategies
    """
    
    base_retriever: BaseRetriever = Field(description="Base retriever to cache")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache entries")
    cache_file: str = Field(default="retriever_cache.db", description="Cache file path")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_cache()
        self.cache_stats = defaultdict(int)
    
    def _init_cache(self):
        """Initialize SQLite cache"""
        self.conn = sqlite3.connect(self.cache_file)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results TEXT NOT NULL,
                timestamp REAL NOT NULL,
                hit_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON retrieval_cache(timestamp)
        """)
        
        self.conn.commit()
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents with caching"""
        query_hash = self._get_query_hash(query)
        current_time = time.time()
        
        # Check cache first
        cached_result = self._get_from_cache(query_hash, current_time)
        if cached_result:
            self.cache_stats["hits"] += 1
            print(f"💾 Cache hit for query: {query[:50]}...")
            return cached_result
        
        # Cache miss - retrieve from base retriever
        self.cache_stats["misses"] += 1
        print(f"🔍 Cache miss, retrieving for: {query[:50]}...")
        
        try:
            documents = self.base_retriever.get_relevant_documents(query)
            self._save_to_cache(query_hash, query, documents, current_time)
            return documents
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []
    
    def _get_from_cache(self, query_hash: str, current_time: float) -> Optional[List[Document]]:
        """Get results from cache if valid"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT results, timestamp, hit_count 
            FROM retrieval_cache 
            WHERE query_hash = ?
        """, (query_hash,))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        results_json, timestamp, hit_count = result
        
        # Check if cache is still valid
        if current_time - timestamp > self.cache_ttl:
            return None
        
        # Update hit count
        cursor.execute("""
            UPDATE retrieval_cache 
            SET hit_count = hit_count + 1 
            WHERE query_hash = ?
        """, (query_hash,))
        self.conn.commit()
        
        # Deserialize documents
        try:
            results_data = json.loads(results_json)
            return [
                Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                for doc_data in results_data
            ]
        except Exception as e:
            print(f"❌ Cache deserialization error: {e}")
            return None
    
    def _save_to_cache(self, query_hash: str, query: str, documents: List[Document], timestamp: float):
        """Save results to cache"""
        try:
            # Serialize documents
            results_data = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            results_json = json.dumps(results_data)
            
            # Calculate average score
            scores = [doc.metadata.get("score", 0.0) for doc in documents]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Save to cache
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO retrieval_cache 
                (query_hash, query, results, timestamp, hit_count, avg_score)
                VALUES (?, ?, ?, ?, 0, ?)
            """, (query_hash, query, results_json, timestamp, avg_score))
            
            self.conn.commit()
            
            # Manage cache size
            self._manage_cache_size()
            
        except Exception as e:
            print(f"❌ Cache save error: {e}")
    
    def _manage_cache_size(self):
        """Remove old cache entries if cache is too large"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM retrieval_cache")
        count = cursor.fetchone()[0]
        
        if count > self.max_cache_size:
            # Remove 20% of oldest entries
            remove_count = int(self.max_cache_size * 0.2)
            cursor.execute("""
                DELETE FROM retrieval_cache 
                WHERE query_hash IN (
                    SELECT query_hash FROM retrieval_cache 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (remove_count,))
            self.conn.commit()
            print(f"🗑️ Removed {remove_count} old cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_entries,
                AVG(hit_count) as avg_hit_count,
                MAX(hit_count) as max_hit_count,
                AVG(avg_score) as avg_score
            FROM retrieval_cache
        """)
        
        db_stats = cursor.fetchone()
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if self.cache_stats["hits"] + self.cache_stats["misses"] > 0 else 0,
            "total_cache_entries": db_stats[0] if db_stats else 0,
            "avg_hit_count": db_stats[1] if db_stats else 0,
            "max_hit_count": db_stats[2] if db_stats else 0,
            "avg_document_score": db_stats[3] if db_stats else 0
        }
    
    def clear_cache(self):
        """Clear all cache entries"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM retrieval_cache")
        self.conn.commit()
        self.cache_stats.clear()
        print("🗑️ Cache cleared")


# ============================================================================
# 3. CONTEXTUAL RE-RANKING RETRIEVER
# ============================================================================

class ContextualReRankingRetriever(BaseRetriever):
    """
    Retriever that re-ranks results based on conversation context
    """
    
    base_retriever: BaseRetriever = Field(description="Base retriever")
    conversation_context: List[str] = Field(default_factory=list, description="Recent conversation context")
    max_context_length: int = Field(default=5, description="Maximum context messages to consider")
    rerank_top_k: int = Field(default=10, description="Number of initial results to rerank")
    final_top_k: int = Field(default=5, description="Final number of results to return")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = create_azure_openai_embeddings()
    
    def add_context(self, message: str):
        """Add a message to conversation context"""
        self.conversation_context.append(message)
        
        # Maintain context length
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve and re-rank documents based on context"""
        print(f"🔍 Contextual retrieval for: {query[:50]}...")
        
        # Get initial results from base retriever
        # Temporarily increase top_k for reranking
        original_top_k = getattr(self.base_retriever, 'top_k', 5)
        if hasattr(self.base_retriever, 'top_k'):
            self.base_retriever.top_k = self.rerank_top_k
        
        try:
            initial_docs = self.base_retriever.get_relevant_documents(query)
        finally:
            # Restore original top_k
            if hasattr(self.base_retriever, 'top_k'):
                self.base_retriever.top_k = original_top_k
        
        if not initial_docs or not self.conversation_context:
            return initial_docs[:self.final_top_k]
        
        # Re-rank based on conversation context
        reranked_docs = self._rerank_by_context(query, initial_docs)
        
        return reranked_docs[:self.final_top_k]
    
    def _rerank_by_context(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents based on conversation context"""
        print(f"🔄 Re-ranking {len(documents)} documents using context...")
        
        try:
            # Create context embedding
            context_text = " ".join(self.conversation_context[-3:])  # Use last 3 messages
            combined_query = f"{context_text} {query}"
            context_embedding = self.embeddings.embed_query(combined_query)
            
            # Calculate contextual relevance for each document
            scored_docs = []
            for doc in documents:
                # Get document embedding
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                
                # Calculate contextual similarity
                contextual_score = self._cosine_similarity(context_embedding, doc_embedding)
                
                # Combine with original score
                original_score = doc.metadata.get("score", 0.0)
                combined_score = 0.6 * original_score + 0.4 * contextual_score
                
                # Update metadata
                doc.metadata["contextual_score"] = contextual_score
                doc.metadata["combined_score"] = combined_score
                doc.metadata["reranked"] = True
                
                scored_docs.append((combined_score, doc))
            
            # Sort by combined score
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            print(f"✅ Re-ranking complete. Top score: {scored_docs[0][0]:.3f}")
            
            return [doc for _, doc in scored_docs]
            
        except Exception as e:
            print(f"❌ Re-ranking error: {e}")
            return documents
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1, vec2 = np.array(vec1), np.array(vec2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except ImportError:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


# ============================================================================
# 4. MULTI-MODAL RETRIEVER
# ============================================================================

class MultiModalRetriever(BaseRetriever):
    """
    Retriever that handles multiple content types (text, code, tables, etc.)
    """
    
    text_retriever: BaseRetriever = Field(description="Retriever for text content")
    specialized_retrievers: Dict[str, BaseRetriever] = Field(default_factory=dict, description="Specialized retrievers by content type")
    content_type_weights: Dict[str, float] = Field(default_factory=lambda: {"text": 0.5, "code": 0.3, "table": 0.2}, description="Weights for different content types")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve from multiple content types and combine results"""
        print(f"🔍 Multi-modal retrieval for: {query[:50]}...")
        
        all_documents = []
        
        # Detect query intent for content type prioritization
        content_preferences = self._detect_content_preferences(query)
        
        # Retrieve from text retriever
        try:
            text_docs = self.text_retriever.get_relevant_documents(query)
            for doc in text_docs:
                doc.metadata["content_type"] = "text"
                doc.metadata["type_weight"] = content_preferences.get("text", 0.5)
            all_documents.extend(text_docs)
        except Exception as e:
            print(f"❌ Text retrieval error: {e}")
        
        # Retrieve from specialized retrievers
        for content_type, retriever in self.specialized_retrievers.items():
            try:
                specialized_docs = retriever.get_relevant_documents(query)
                for doc in specialized_docs:
                    doc.metadata["content_type"] = content_type
                    doc.metadata["type_weight"] = content_preferences.get(content_type, 0.1)
                all_documents.extend(specialized_docs)
            except Exception as e:
                print(f"❌ {content_type} retrieval error: {e}")
        
        # Re-score and combine results
        return self._combine_multimodal_results(all_documents, content_preferences)
    
    def _detect_content_preferences(self, query: str) -> Dict[str, float]:
        """Detect what type of content the query is looking for"""
        query_lower = query.lower()
        preferences = self.content_type_weights.copy()
        
        # Code-related keywords
        code_keywords = ["function", "method", "class", "api", "code", "implementation", "syntax", "library", "framework"]
        if any(keyword in query_lower for keyword in code_keywords):
            preferences["code"] = min(preferences["code"] + 0.3, 1.0)
            preferences["text"] = max(preferences["text"] - 0.1, 0.1)
        
        # Table/data-related keywords
        table_keywords = ["table", "data", "statistics", "metrics", "comparison", "chart", "graph"]
        if any(keyword in query_lower for keyword in table_keywords):
            preferences["table"] = min(preferences["table"] + 0.3, 1.0)
            preferences["text"] = max(preferences["text"] - 0.1, 0.1)
        
        # Documentation/explanation keywords
        doc_keywords = ["explain", "documentation", "guide", "tutorial", "overview", "introduction"]
        if any(keyword in query_lower for keyword in doc_keywords):
            preferences["text"] = min(preferences["text"] + 0.2, 1.0)
        
        return preferences
    
    def _combine_multimodal_results(self, documents: List[Document], preferences: Dict[str, float]) -> List[Document]:
        """Combine and score results from different content types"""
        if not documents:
            return []
        
        # Calculate combined scores
        for doc in documents:
            original_score = doc.metadata.get("score", 0.0)
            content_type = doc.metadata.get("content_type", "text")
            type_weight = preferences.get(content_type, 0.1)
            
            # Combine original score with content type preference
            combined_score = original_score * type_weight
            doc.metadata["multimodal_score"] = combined_score
        
        # Sort by combined score
        documents.sort(key=lambda x: x.metadata.get("multimodal_score", 0.0), reverse=True)
        
        # Group by content type for diversity
        content_groups = defaultdict(list)
        for doc in documents:
            content_type = doc.metadata.get("content_type", "text")
            content_groups[content_type].append(doc)
        
        # Select diverse results
        final_docs = []
        max_per_type = 3
        
        # First, add top results from each content type
        for content_type, docs in content_groups.items():
            final_docs.extend(docs[:max_per_type])
        
        # Sort final results by score
        final_docs.sort(key=lambda x: x.metadata.get("multimodal_score", 0.0), reverse=True)
        
        print(f"✅ Combined {len(documents)} results from {len(content_groups)} content types")
        
        return final_docs[:10]  # Return top 10


# ============================================================================
# 5. DEMONSTRATION FUNCTIONS
# ============================================================================

async def demonstrate_azure_search_retriever():
    """Demonstrate Azure Cognitive Search retriever"""
    print("🔍 Azure Cognitive Search Retriever Demo")
    print("=" * 50)
    
    # Initialize retriever (with mock credentials for demo)
    retriever = AzureCognitiveSearchRetriever(
        search_service_name="demo-search-service",
        search_index_name="documents",
        api_key="demo-key",
        top_k=3,
        search_mode="hybrid"
    )
    
    test_queries = [
        "Azure OpenAI integration best practices",
        "Python authentication with Azure services",
        "LangChain custom components tutorial"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        docs = retriever.get_relevant_documents(query)
        
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. [{doc.metadata.get('source', 'unknown')}] Score: {doc.metadata.get('hybrid_score', 0):.3f}")
            print(f"     {doc.page_content[:100]}...")


async def demonstrate_cached_retriever():
    """Demonstrate cached retriever"""
    print("\n💾 Cached Retriever Demo")
    print("=" * 40)
    
    # Create base retriever
    base_retriever = AzureCognitiveSearchRetriever(
        search_service_name="demo-search-service",
        search_index_name="documents", 
        api_key="demo-key",
        top_k=2
    )
    
    # Wrap with caching
    cached_retriever = CachedRetriever(
        base_retriever=base_retriever,
        cache_ttl=60,  # 1 minute for demo
        max_cache_size=100
    )
    
    test_query = "Azure authentication methods"
    
    print(f"📋 First retrieval (should be cache miss): {test_query}")
    docs1 = cached_retriever.get_relevant_documents(test_query)
    print(f"   Retrieved {len(docs1)} documents")
    
    print(f"\n📋 Second retrieval (should be cache hit): {test_query}")
    docs2 = cached_retriever.get_relevant_documents(test_query)
    print(f"   Retrieved {len(docs2)} documents")
    
    # Show cache stats
    stats = cached_retriever.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Total entries: {stats['total_cache_entries']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")


async def demonstrate_contextual_retriever():
    """Demonstrate contextual re-ranking retriever"""
    print("\n🧠 Contextual Re-ranking Retriever Demo")
    print("=" * 50)
    
    # Create base retriever
    base_retriever = AzureCognitiveSearchRetriever(
        search_service_name="demo-search-service",
        search_index_name="documents",
        api_key="demo-key",
        top_k=3
    )
    
    # Wrap with contextual re-ranking
    contextual_retriever = ContextualReRankingRetriever(
        base_retriever=base_retriever,
        rerank_top_k=5,
        final_top_k=3
    )
    
    # Simulate conversation context
    conversation = [
        ("I'm working on a Python project with Azure services", "Great! What specific Azure services are you using?"),
        ("I need to integrate Azure OpenAI for chat functionality", "Perfect! Azure OpenAI provides excellent chat capabilities. What authentication method do you prefer?"),
        ("I want to use managed identity for security", "Excellent choice! Managed identity is very secure. Let me help you with the implementation.")
    ]
    
    print("Building conversation context:")
    for human, ai in conversation:
        contextual_retriever.add_context(human)
        contextual_retriever.add_context(ai)
        print(f"  Human: {human}")
        print(f"  AI: {ai}")
    
    # Test retrieval with context
    test_query = "authentication implementation example"
    print(f"\n📋 Query with context: {test_query}")
    docs = contextual_retriever.get_relevant_documents(test_query)
    
    for i, doc in enumerate(docs, 1):
        contextual_score = doc.metadata.get("contextual_score", 0)
        combined_score = doc.metadata.get("combined_score", 0)
        print(f"  {i}. Contextual: {contextual_score:.3f}, Combined: {combined_score:.3f}")
        print(f"     {doc.page_content[:80]}...")


async def demonstrate_multimodal_retriever():
    """Demonstrate multi-modal retriever"""
    print("\n🎭 Multi-modal Retriever Demo")
    print("=" * 40)
    
    # Create specialized retrievers (mock for demo)
    text_retriever = AzureCognitiveSearchRetriever(
        search_service_name="demo-search-service",
        search_index_name="text_documents",
        api_key="demo-key",
        top_k=2
    )
    
    code_retriever = AzureCognitiveSearchRetriever(
        search_service_name="demo-search-service", 
        search_index_name="code_examples",
        api_key="demo-key",
        top_k=2
    )
    
    # Create multi-modal retriever
    multimodal_retriever = MultiModalRetriever(
        text_retriever=text_retriever,
        specialized_retrievers={"code": code_retriever},
        content_type_weights={"text": 0.4, "code": 0.6}
    )
    
    test_queries = [
        "How to authenticate with Azure OpenAI",
        "Show me Python code for Azure authentication",
        "Explain the authentication flow documentation"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        docs = multimodal_retriever.get_relevant_documents(query)
        
        for i, doc in enumerate(docs, 1):
            content_type = doc.metadata.get("content_type", "unknown")
            score = doc.metadata.get("multimodal_score", 0)
            print(f"  {i}. [{content_type}] Score: {score:.3f}")
            print(f"     {doc.page_content[:60]}...")


# ============================================================================
# 6. MAIN DEMONSTRATION
# ============================================================================

async def main():
    """Main demonstration of custom retrievers"""
    print("🚀 Custom Retriever Implementations with Azure Integration")
    print("=" * 70)
    
    try:
        # Demonstrate Azure Cognitive Search retriever
        await demonstrate_azure_search_retriever()
        
        # Demonstrate caching
        await demonstrate_cached_retriever()
        
        # Demonstrate contextual re-ranking  
        await demonstrate_contextual_retriever()
        
        # Demonstrate multi-modal retrieval
        await demonstrate_multimodal_retriever()
        
        print("\n🎉 All retriever demos completed!")
        print("\n💡 Key Features Demonstrated:")
        print("✅ Azure Cognitive Search integration (hybrid search)")
        print("✅ Intelligent caching with performance tracking")
        print("✅ Contextual re-ranking based on conversation history")
        print("✅ Multi-modal content retrieval and scoring")
        print("✅ Configurable search modes (vector, keyword, semantic, hybrid)")
        print("✅ Performance optimization and error handling")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Make sure:")
        print("1. Azure OpenAI is configured properly")
        print("2. Required packages are installed")
        print("3. Storage directory is writable")

if __name__ == "__main__":
    asyncio.run(main())
