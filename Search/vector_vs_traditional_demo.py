"""
Vector Search vs Traditional Search Side-by-Side Comparison
This example creates both indexes and demonstrates the differences in search results
"""

import os
import time
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class VectorVsTraditionalDemo:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError("Please set AZURE_SEARCH_SERVICE_ENDPOINT and AZURE_SEARCH_ADMIN_KEY in .env file")
        
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        
        # Index names
        self.traditional_index = "traditional-comparison"
        self.vector_index = "vector-comparison"
        
        # Search clients
        self.traditional_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.traditional_index,
            credential=self.credential
        )
        self.vector_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.vector_index,
            credential=self.credential
        )

    def create_traditional_index(self):
        """Create traditional text-only search index"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        ]
        
        index = SearchIndex(name=self.traditional_index, fields=fields)
        
        try:
            self.index_client.create_index(index)
            print("✅ Traditional search index created successfully")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("ℹ️  Traditional index already exists")
            else:
                print(f"❌ Error creating traditional index: {e}")

    def create_vector_index(self):
        """Create vector search index with embeddings"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=384,  # Using smaller dimension for demo (sentence-transformers compatible)
                vector_search_profile_name="vector-profile"
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )],
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
        )
        
        index = SearchIndex(
            name=self.vector_index,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            self.index_client.create_index(index)
            print("✅ Vector search index created successfully")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("ℹ️  Vector index already exists")
            else:
                print(f"❌ Error creating vector index: {e}")

    def generate_mock_embedding(self, text, dimensions=384):
        """Generate a mock embedding vector for demonstration purposes"""
        # In a real implementation, you would use a proper embedding model
        # This creates a simple hash-based vector for demonstration
        import hashlib
        
        # Create a hash of the text
        hash_obj = hashlib.md5(text.lower().encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector (simplified approach for demo)
        vector = []
        for i in range(dimensions):
            # Use hash bytes cyclically to create vector values
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            vector.append((byte_val - 128) / 128.0)
        
        return vector

    def get_sample_documents(self):
        """Get sample documents that demonstrate search differences"""
        documents = [
            {
                "id": "1",
                "title": "Automotive Maintenance Guide",
                "content": "Regular vehicle servicing includes engine diagnostics, brake inspection, and fluid checks. Professional mechanics recommend routine maintenance to prevent costly repairs.",
                "category": "Transportation",
                "tags": ["automotive", "maintenance", "vehicle", "repair"]
            },
            {
                "id": "2",
                "title": "Computer Troubleshooting Manual",
                "content": "System diagnostics help identify hardware failures. Boot issues often require motherboard inspection, memory testing, and power supply verification.",
                "category": "Technology",
                "tags": ["computer", "troubleshooting", "hardware", "diagnostics"]
            },
            {
                "id": "3",
                "title": "Medical Professional Guidelines",
                "content": "Healthcare providers must follow diagnostic protocols. Physicians use clinical assessment and laboratory tests for accurate patient evaluation and treatment planning.",
                "category": "Healthcare",
                "tags": ["medical", "healthcare", "physician", "diagnosis"]
            },
            {
                "id": "4",
                "title": "Artificial Intelligence and Neural Networks",
                "content": "Deep learning models utilize artificial neural networks with multiple layers. These systems automatically learn complex patterns from training datasets without explicit programming.",
                "category": "Technology",
                "tags": ["AI", "neural networks", "deep learning", "machine learning"]
            },
            {
                "id": "5",
                "title": "Data Science and Analytics",
                "content": "Statistical analysis and predictive modeling extract meaningful insights from large datasets. Advanced algorithms process information to support data-driven decision making.",
                "category": "Technology",
                "tags": ["data science", "analytics", "statistics", "algorithms"]
            },
            {
                "id": "6",
                "title": "Mobile Device Repair",
                "content": "Smartphone and tablet restoration involves screen replacement, battery servicing, and component troubleshooting. Technical specialists use precision tools for electronic device maintenance.",
                "category": "Technology",
                "tags": ["mobile", "smartphone", "tablet", "repair", "electronics"]
            },
            {
                "id": "7",
                "title": "Financial Investment Strategies",
                "content": "Portfolio management requires risk assessment and market analysis. Investment advisors recommend diversified asset allocation for long-term wealth building.",
                "category": "Finance",
                "tags": ["investment", "finance", "portfolio", "wealth"]
            },
            {
                "id": "8",
                "title": "Cooking and Recipe Development",
                "content": "Culinary arts involve ingredient preparation, cooking techniques, and flavor balancing. Professional chefs create innovative dishes through experimentation and skill development.",
                "category": "Food",
                "tags": ["cooking", "recipes", "culinary", "chef"]
            }
        ]
        
        return documents

    def upload_documents(self):
        """Upload documents to both indexes"""
        documents = self.get_sample_documents()
        
        # Upload to traditional index
        try:
            self.traditional_client.upload_documents(documents)
            print("✅ Documents uploaded to traditional index")
        except Exception as e:
            print(f"❌ Error uploading to traditional index: {e}")
        
        # Prepare documents with embeddings for vector index
        vector_documents = []
        for doc in documents:
            vector_doc = doc.copy()
            # Generate embedding for title + content
            content_for_embedding = f"{doc['title']} {doc['content']}"
            vector_doc['content_vector'] = self.generate_mock_embedding(content_for_embedding)
            vector_documents.append(vector_doc)
        
        # Upload to vector index
        try:
            self.vector_client.upload_documents(vector_documents)
            print("✅ Documents uploaded to vector index")
        except Exception as e:
            print(f"❌ Error uploading to vector index: {e}")

    def search_traditional(self, query):
        """Perform traditional text search"""
        try:
            results = self.traditional_client.search(
                search_text=query,
                top=5,
                include_total_count=True
            )
            return list(results)
        except Exception as e:
            print(f"❌ Error in traditional search: {e}")
            return []

    def search_vector(self, query):
        """Perform vector search"""
        try:
            # Generate query embedding
            query_vector = self.generate_mock_embedding(query)
            
            results = self.vector_client.search(
                search_text=None,  # Pure vector search
                vectors=[{
                    "value": query_vector,
                    "k_nearest_neighbors": 5,
                    "fields": "content_vector"
                }],
                top=5
            )
            return list(results)
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def search_hybrid(self, query):
        """Perform hybrid search (text + vector)"""
        try:
            # Generate query embedding
            query_vector = self.generate_mock_embedding(query)
            
            results = self.vector_client.search(
                search_text=query,  # Text search component
                vectors=[{
                    "value": query_vector,
                    "k_nearest_neighbors": 5,
                    "fields": "content_vector"
                }],  # Vector search component
                top=5
            )
            return list(results)
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            return []

    def display_results(self, results, search_type):
        """Display search results in a formatted way"""
        print(f"\n📋 {search_type.upper()} SEARCH RESULTS:")
        print("-" * 50)
        
        if not results:
            print("   ❌ No results found")
            return
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 📄 {result['title']}")
            print(f"   📂 Category: {result['category']}")
            print(f"   📝 Content: {result['content'][:100]}...")
            
            # Show relevance score if available
            if '@search.score' in result:
                print(f"   ⭐ Relevance Score: {result['@search.score']:.3f}")
            
            print()

    def compare_searches(self, query):
        """Compare all three search methods for a given query"""
        print(f"\n🔍 SEARCH COMPARISON FOR: '{query}'")
        print("=" * 70)
        
        # Traditional search
        traditional_results = self.search_traditional(query)
        self.display_results(traditional_results, "Traditional")
        
        # Vector search
        vector_results = self.search_vector(query)
        self.display_results(vector_results, "Vector")
        
        # Hybrid search
        hybrid_results = self.search_hybrid(query)
        self.display_results(hybrid_results, "Hybrid")
        
        # Analysis
        print("\n🎯 ANALYSIS:")
        print(f"   Traditional found: {len(traditional_results)} results")
        print(f"   Vector found: {len(vector_results)} results")
        print(f"   Hybrid found: {len(hybrid_results)} results")

    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all search types"""
        print("🚀 VECTOR VS TRADITIONAL SEARCH DEMONSTRATION")
        print("=" * 60)
        
        # Setup
        print("\n1️⃣ Setting up indexes...")
        self.create_traditional_index()
        self.create_vector_index()
        
        print("\n2️⃣ Uploading sample documents...")
        self.upload_documents()
        
        print("\n3️⃣ Waiting for indexing to complete...")
        time.sleep(3)
        
        # Test queries that demonstrate differences
        test_queries = [
            {
                "query": "car repair",
                "explanation": "Should find 'automotive maintenance' even without exact keywords"
            },
            {
                "query": "doctor guidelines",
                "explanation": "Should find 'medical professional' and 'physician' content"
            },
            {
                "query": "machine learning",
                "explanation": "Should find AI and neural network content"
            },
            {
                "query": "phone fix",
                "explanation": "Should find mobile device repair content"
            },
            {
                "query": "cooking tips",
                "explanation": "Should find culinary and recipe content"
            }
        ]
        
        print("\n4️⃣ Running search comparisons...")
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*20} TEST {i} {'='*20}")
            print(f"💡 Expected: {test_case['explanation']}")
            self.compare_searches(test_case['query'])
            
            if i < len(test_queries):
                input("\nPress Enter to continue to next test...")

    def cleanup_indexes(self):
        """Clean up created indexes"""
        try:
            self.index_client.delete_index(self.traditional_index)
            print(f"🗑️ Deleted {self.traditional_index}")
        except:
            pass
        
        try:
            self.index_client.delete_index(self.vector_index)
            print(f"🗑️ Deleted {self.vector_index}")
        except:
            pass


def main():
    """Main function to run the demo"""
    try:
        demo = VectorVsTraditionalDemo()
        
        print("Welcome to Vector vs Traditional Search Demo!")
        print("This demo will create two indexes and compare search results.")
        print("\nNote: This uses mock embeddings for demonstration.")
        print("In production, you would use real embedding models like OpenAI's text-embedding-ada-002")
        
        choice = input("\nDo you want to run the full demo? (y/n): ").lower().strip()
        
        if choice == 'y':
            demo.run_comprehensive_demo()
            
            cleanup = input("\nDo you want to clean up the created indexes? (y/n): ").lower().strip()
            if cleanup == 'y':
                demo.cleanup_indexes()
        else:
            print("Demo cancelled. You can run individual methods if needed.")
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("\nMake sure your Azure AI Search service is properly configured in the .env file.")


if __name__ == "__main__":
    main()
