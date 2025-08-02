"""
Vector Search Demo with Embeddings
Demonstrates custom vector search implementation using embedding models
"""

import os
import hashlib
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

# Optional: OpenAI for real embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ℹ️  OpenAI not installed - using mock embeddings for demo")

load_dotenv()

class VectorSearchDemo:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = "vector-search-demo"
        
        # OpenAI configuration (optional)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError("Please set Azure AI Search credentials in .env file")
        
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def generate_embedding(self, text, use_openai=False):
        """Generate embedding vector for text"""
        if use_openai and OPENAI_AVAILABLE and self.openai_api_key:
            return self._openai_embedding(text)
        else:
            return self._mock_embedding(text)
    
    def _openai_embedding(self, text):
        """Generate real OpenAI embedding"""
        try:
            openai.api_key = self.openai_api_key
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"⚠️  OpenAI embedding failed: {e}, using mock instead")
            return self._mock_embedding(text)
    
    def _mock_embedding(self, text, dimensions=384):
        """Generate mock embedding for demonstration"""
        # Create consistent hash-based embedding for demo purposes
        hash_obj = hashlib.md5(text.lower().encode())
        hash_bytes = hash_obj.digest()
        
        vector = []
        for i in range(dimensions):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            vector.append((byte_val - 128) / 128.0)
        
        return vector

    def create_vector_index(self):
        """Create index with vector search capabilities"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="tags", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=384,  # Using smaller dimension for demo
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
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            self.index_client.create_index(index)
            print("✅ Vector search index created successfully!")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("ℹ️  Vector index already exists")
            else:
                print(f"❌ Error creating vector index: {e}")

    def upload_documents_with_vectors(self):
        """Upload documents with generated vector embeddings"""
        documents = [
            {
                "id": "1",
                "title": "Artificial Intelligence in Healthcare",
                "content": "Machine learning algorithms are revolutionizing medical diagnosis. AI systems can analyze medical images, predict patient outcomes, and assist doctors in treatment planning.",
                "category": "Technology",
                "tags": "AI, healthcare, machine learning, medical diagnosis"
            },
            {
                "id": "2",
                "title": "Climate Change and Renewable Energy",
                "content": "Solar and wind power technologies are becoming more efficient and cost-effective. These sustainable energy sources are crucial for reducing carbon emissions.",
                "category": "Environment",
                "tags": "climate change, renewable energy, solar power, sustainability"
            },
            {
                "id": "3",
                "title": "Financial Investment Strategies",
                "content": "Diversified portfolios help manage investment risk. Long-term strategies often outperform short-term trading approaches in building wealth.",
                "category": "Finance",
                "tags": "investment, portfolio, wealth management, financial planning"
            },
            {
                "id": "4",
                "title": "Remote Work Technologies",
                "content": "Video conferencing and collaboration tools have transformed how teams work together. Cloud computing enables seamless remote access to business applications.",
                "category": "Technology",
                "tags": "remote work, video conferencing, cloud computing, collaboration"
            },
            {
                "id": "5",
                "title": "Healthy Cooking and Nutrition",
                "content": "Balanced diets with fresh vegetables and lean proteins promote wellness. Meal planning and preparation are key to maintaining healthy eating habits.",
                "category": "Health",
                "tags": "nutrition, healthy cooking, meal planning, wellness"
            }
        ]
        
        # Generate embeddings for each document
        print("Generating embeddings for documents...")
        for doc in documents:
            content_for_embedding = f"{doc['title']} {doc['content']} {doc['tags']}"
            doc['content_vector'] = self.generate_embedding(content_for_embedding)
        
        try:
            self.search_client.upload_documents(documents)
            print(f"✅ Uploaded {len(documents)} documents with vector embeddings")
        except Exception as e:
            print(f"❌ Error uploading documents: {e}")

    def vector_search(self, query):
        """Perform pure vector search"""
        try:
            # Generate embedding for query
            query_vector = self.generate_embedding(query)
            
            results = self.search_client.search(
                search_text=None,  # Pure vector search
                vectors=[{
                    "value": query_vector,
                    "k_nearest_neighbors": 5,
                    "fields": "content_vector"
                }],
                top=5
            )
            
            print(f"\n🔢 VECTOR SEARCH RESULTS FOR: '{query}'")
            print("=" * 60)
            
            result_count = 0
            for result in results:
                result_count += 1
                print(f"\n{result_count}. 📄 {result['title']}")
                print(f"   📂 Category: {result['category']}")
                print(f"   📝 Content: {result['content'][:100]}...")
                if '@search.score' in result:
                    print(f"   ⭐ Similarity Score: {result['@search.score']:.3f}")
            
            if result_count == 0:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"❌ Error in vector search: {e}")

    def hybrid_search(self, query):
        """Perform hybrid search (text + vector)"""
        try:
            # Generate embedding for query
            query_vector = self.generate_embedding(query)
            
            results = self.search_client.search(
                search_text=query,  # Text search component
                vectors=[{
                    "value": query_vector,
                    "k_nearest_neighbors": 5,
                    "fields": "content_vector"
                }],  # Vector search component
                top=5
            )
            
            print(f"\n🚀 HYBRID SEARCH RESULTS FOR: '{query}'")
            print("=" * 60)
            
            result_count = 0
            for result in results:
                result_count += 1
                print(f"\n{result_count}. 📄 {result['title']}")
                print(f"   📂 Category: {result['category']}")
                print(f"   📝 Content: {result['content'][:100]}...")
                if '@search.score' in result:
                    print(f"   ⭐ Combined Score: {result['@search.score']:.3f}")
            
            if result_count == 0:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")

    def traditional_search(self, query):
        """Compare with traditional text search"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="simple",
                top=5
            )
            
            print(f"\n📝 TRADITIONAL SEARCH RESULTS FOR: '{query}'")
            print("=" * 60)
            
            result_count = 0
            for result in results:
                result_count += 1
                print(f"\n{result_count}. 📄 {result['title']}")
                print(f"   📂 Category: {result['category']}")
                print(f"   📝 Content: {result['content'][:100]}...")
                if '@search.score' in result:
                    print(f"   ⭐ Text Score: {result['@search.score']:.3f}")
            
            if result_count == 0:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"❌ Error in traditional search: {e}")

    def run_vector_demo(self):
        """Run comprehensive vector search demonstration"""
        print("🔢 VECTOR SEARCH DEMONSTRATION")
        print("=" * 40)
        
        # Setup
        self.create_vector_index()
        self.upload_documents_with_vectors()
        
        # Wait for indexing
        import time
        print("\nWaiting for indexing...")
        time.sleep(3)
        
        # Test queries that show vector search benefits
        test_queries = [
            "medical AI diagnosis",  # Should find healthcare AI content
            "green energy solutions",  # Should find renewable energy content
            "money management tips",  # Should find investment content
            "working from home",  # Should find remote work content
            "healthy eating habits"  # Should find nutrition content
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"🔍 TESTING QUERY: '{query}'")
            print(f"{'='*80}")
            
            self.traditional_search(query)
            self.vector_search(query)
            self.hybrid_search(query)
            
            input("\nPress Enter to continue to next query...")

    def cleanup(self):
        """Clean up the demo index"""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"🗑️ Cleaned up index: {self.index_name}")
        except Exception as e:
            print(f"ℹ️  Cleanup note: {e}")


def main():
    """Main demonstration function"""
    try:
        demo = VectorSearchDemo()
        
        print("🎯 VECTOR SEARCH DEMO")
        print("This demo shows custom vector search implementation")
        print("Using mock embeddings for demonstration (set OPENAI_API_KEY for real embeddings)")
        
        demo.run_vector_demo()
        
        cleanup = input("\nClean up demo index? (y/n): ").lower().strip()
        if cleanup == 'y':
            demo.cleanup()
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\nMake sure you have valid Azure AI Search credentials in .env")


if __name__ == "__main__":
    main()
