"""
Simple Vector vs Traditional Search Example
This example demonstrates the key differences with real Azure AI Search
"""

import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class SimpleSearchComparison:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = "search-comparison-demo"
        
        if not self.endpoint or not self.key:
            raise ValueError("Please set Azure AI Search credentials in .env file")
        
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def create_index(self):
        """Create a search index for comparison demo"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="keywords", type=SearchFieldDataType.String),
        ]
        
        index = SearchIndex(name=self.index_name, fields=fields)
        
        try:
            self.index_client.create_index(index)
            print(f"✅ Index '{self.index_name}' created successfully!")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Index '{self.index_name}' already exists")
            else:
                print(f"❌ Error creating index: {e}")

    def upload_demo_documents(self):
        """Upload documents that clearly show traditional search limitations"""
        documents = [
            {
                "id": "1",
                "title": "Automotive Maintenance Guide",
                "content": "Vehicle servicing and car care. Regular maintenance includes engine checks, brake inspection, and fluid replacement. Professional mechanics recommend routine service.",
                "category": "Transportation",
                "keywords": "automotive maintenance vehicle car servicing repair"
            },
            {
                "id": "2", 
                "title": "Computer Hardware Troubleshooting",
                "content": "PC repair and system diagnostics. Hardware failures require motherboard testing, memory checks, and component replacement by technicians.",
                "category": "Technology",
                "keywords": "computer hardware troubleshooting PC repair diagnostics"
            },
            {
                "id": "3",
                "title": "Medical Professional Guidelines", 
                "content": "Healthcare provider protocols and physician guidelines. Doctors must follow clinical assessment procedures for patient diagnosis and treatment planning.",
                "category": "Healthcare",
                "keywords": "medical healthcare physician doctor guidelines clinical"
            },
            {
                "id": "4",
                "title": "Neural Networks and AI",
                "content": "Artificial intelligence and deep learning systems. Machine learning algorithms use neural networks to process data and learn patterns automatically.",
                "category": "Technology", 
                "keywords": "AI artificial intelligence neural networks machine learning deep learning"
            },
            {
                "id": "5",
                "title": "Mobile Device Service",
                "content": "Smartphone and tablet maintenance. Mobile phone repair includes screen replacement, battery service, and electronic component troubleshooting.",
                "category": "Technology",
                "keywords": "mobile smartphone tablet phone device repair service"
            },
            {
                "id": "6",
                "title": "Investment Portfolio Management",
                "content": "Financial planning and wealth building strategies. Investment advisors help with asset allocation, risk management, and retirement planning.",
                "category": "Finance",
                "keywords": "investment finance portfolio wealth financial planning"
            }
        ]
        
        try:
            self.search_client.upload_documents(documents)
            print(f"✅ Uploaded {len(documents)} demo documents")
        except Exception as e:
            print(f"❌ Error uploading documents: {e}")

    def traditional_search(self, query):
        """Perform traditional keyword-based search"""
        try:
            results = self.search_client.search(
                search_text=query,
                search_mode="all",  # Requires all words to match
                top=10
            )
            return list(results)
        except Exception as e:
            print(f"❌ Traditional search error: {e}")
            return []

    def fuzzy_search(self, query):
        """Perform search with fuzzy matching"""
        try:
            results = self.search_client.search(
                search_text=query,
                search_mode="any",  # Any word can match
                query_type="simple",
                top=10
            )
            return list(results)
        except Exception as e:
            print(f"❌ Fuzzy search error: {e}")
            return []

    def semantic_search(self, query):
        """Perform semantic search (if available)"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                top=10
            )
            return list(results)
        except Exception as e:
            print(f"ℹ️  Semantic search not available (requires configuration): {e}")
            return []

    def display_search_results(self, results, search_type):
        """Display search results in a clear format"""
        print(f"\n🔍 {search_type.upper()} SEARCH RESULTS:")
        print("-" * 45)
        
        if not results:
            print("   ❌ No results found")
            return
        
        for i, result in enumerate(results[:5], 1):  # Show top 5
            print(f"{i}. 📄 {result['title']}")
            print(f"   📂 {result['category']}")
            print(f"   📝 {result['content'][:80]}...")
            if '@search.score' in result:
                print(f"   ⭐ Score: {result['@search.score']:.2f}")
            print()

    def compare_search_approaches(self, query):
        """Compare different search approaches for a query"""
        print(f"\n{'='*60}")
        print(f"🔍 COMPARING SEARCH APPROACHES FOR: '{query}'")
        print(f"{'='*60}")
        
        # Traditional exact search
        traditional_results = self.traditional_search(query)
        self.display_search_results(traditional_results, "Traditional (Exact)")
        
        # Fuzzy search
        fuzzy_results = self.fuzzy_search(query)
        self.display_search_results(fuzzy_results, "Fuzzy (Any Match)")
        
        # Semantic search (if available)
        semantic_results = self.semantic_search(query)
        if semantic_results:
            self.display_search_results(semantic_results, "Semantic")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Traditional found: {len(traditional_results)} results")
        print(f"   Fuzzy found: {len(fuzzy_results)} results")
        print(f"   Semantic found: {len(semantic_results)} results")

    def run_comparison_demo(self):
        """Run a comprehensive comparison demo"""
        print("🚀 TRADITIONAL vs SEMANTIC SEARCH COMPARISON")
        print("=" * 55)
        
        # Setup
        print("\n1️⃣ Creating search index...")
        self.create_index()
        
        print("\n2️⃣ Uploading sample documents...")
        self.upload_demo_documents()
        
        # Wait for indexing
        import time
        print("\n3️⃣ Waiting for indexing...")
        time.sleep(2)
        
        # Test different types of queries
        test_queries = [
            {
                "query": "car repair",
                "note": "Exact words NOT in documents - they use 'automotive maintenance'"
            },
            {
                "query": "doctor guidelines", 
                "note": "Documents use 'physician' instead of 'doctor'"
            },
            {
                "query": "machine learning",
                "note": "Should find AI content even if exact phrase isn't used"
            },
            {
                "query": "phone fix",
                "note": "Documents talk about 'mobile device service'"
            },
            {
                "query": "investment advice",
                "note": "Documents mention 'portfolio management' and 'financial planning'"
            }
        ]
        
        print("\n4️⃣ Running search comparisons...")
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{'🧪 TEST ' + str(i):=^60}")
            print(f"💡 Note: {test['note']}")
            self.compare_search_approaches(test['query'])
            
            if i < len(test_queries):
                input("\n⏸️  Press Enter to continue...")

    def demonstrate_traditional_limitations(self):
        """Show specific limitations of traditional search"""
        print("\n❌ TRADITIONAL SEARCH LIMITATIONS")
        print("=" * 45)
        
        limitations = [
            {
                "issue": "Exact Keyword Dependency",
                "query": "car repair",
                "problem": "Misses 'automotive maintenance' documents",
                "solution": "Vector search understands semantic similarity"
            },
            {
                "issue": "Synonym Problems", 
                "query": "doctor",
                "problem": "Doesn't find 'physician' content",
                "solution": "Vector search handles synonyms naturally"
            },
            {
                "issue": "Context Ignorance",
                "query": "apple",
                "problem": "Can't distinguish fruit vs company",
                "solution": "Vector search considers context"
            },
            {
                "issue": "Language Variations",
                "query": "automobile",
                "problem": "Misses 'car', 'vehicle' content", 
                "solution": "Vector search works across language variations"
            }
        ]
        
        for limitation in limitations:
            print(f"\n🚫 {limitation['issue']}")
            print(f"   Query: '{limitation['query']}'")
            print(f"   Problem: {limitation['problem']}")
            print(f"   ✅ Solution: {limitation['solution']}")

    def cleanup(self):
        """Clean up the demo index"""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"🗑️ Cleaned up index: {self.index_name}")
        except Exception as e:
            print(f"ℹ️  Cleanup note: {e}")


def main():
    """Main function"""
    try:
        demo = SimpleSearchComparison()
        
        print("🎯 SEARCH COMPARISON DEMO")
        print("This demo shows the differences between traditional and semantic search")
        print("using real Azure AI Search capabilities.")
        
        # Run the demo
        demo.run_comparison_demo()
        
        # Show limitations
        demo.demonstrate_traditional_limitations()
        
        # Cleanup option
        cleanup = input("\n🗑️  Clean up demo index? (y/n): ").lower().strip()
        if cleanup == 'y':
            demo.cleanup()
        
        print("\n✨ Demo completed!")
        print("\n💡 Key Takeaway:")
        print("   Traditional search = Find exact words")
        print("   Vector search = Understand meaning")
        print("   Hybrid search = Best of both worlds")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\nMake sure your .env file has valid Azure AI Search credentials!")


if __name__ == "__main__":
    main()
