"""
Semantic Search Demo
Demonstrates Azure AI Search's built-in semantic search capabilities
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
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class SemanticSearchDemo:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = "semantic-search-demo"
        
        if not self.endpoint or not self.key:
            raise ValueError("Please set Azure AI Search credentials in .env file")
        
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def create_semantic_index(self):
        """Create index with semantic search configuration"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="tags", type=SearchFieldDataType.String),
        ]
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="tags")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            semantic_search=semantic_search
        )
        
        try:
            self.index_client.create_index(index)
            print("✅ Semantic search index created successfully!")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("ℹ️  Semantic index already exists")
            else:
                print(f"❌ Error creating semantic index: {e}")

    def upload_sample_documents(self):
        """Upload documents for semantic search testing"""
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
        
        try:
            self.search_client.upload_documents(documents)
            print(f"✅ Uploaded {len(documents)} documents for semantic search")
        except Exception as e:
            print(f"❌ Error uploading documents: {e}")

    def semantic_search(self, query):
        """Perform semantic search with answer extraction"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="default",
                query_caption="extractive",
                query_answer="extractive",
                top=5
            )
            
            print(f"\n🧠 SEMANTIC SEARCH RESULTS FOR: '{query}'")
            print("=" * 60)
            
            # Check for semantic answers
            for result in results:
                if '@search.answers' in result:
                    answers = result['@search.answers']
                    if answers:
                        print(f"\n💡 SEMANTIC ANSWER:")
                        for answer in answers:
                            print(f"   {answer['text']}")
                            print(f"   Confidence: {answer.get('score', 0):.2f}")
                        print("-" * 40)
                break
            
            # Display search results
            result_count = 0
            for result in results:
                result_count += 1
                print(f"\n{result_count}. 📄 {result['title']}")
                print(f"   📂 Category: {result['category']}")
                print(f"   📝 Content: {result['content'][:100]}...")
                
                # Show semantic captions if available
                if '@search.captions' in result:
                    captions = result['@search.captions']
                    if captions:
                        print(f"   🎯 Semantic Caption: {captions[0]['text']}")
                
                # Show relevance score
                if '@search.score' in result:
                    print(f"   ⭐ Relevance Score: {result['@search.score']:.3f}")
            
            if result_count == 0:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            if "not enabled" in str(e).lower():
                print("💡 Tip: Semantic search requires Standard tier or higher")

    def traditional_search(self, query):
        """Compare with traditional search"""
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
                    print(f"   ⭐ Score: {result['@search.score']:.3f}")
            
            if result_count == 0:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"❌ Error in traditional search: {e}")

    def run_comparison_demo(self):
        """Run semantic vs traditional search comparison"""
        print("🧠 SEMANTIC SEARCH DEMONSTRATION")
        print("=" * 40)
        
        # Setup
        self.create_semantic_index()
        self.upload_sample_documents()
        
        # Wait for indexing
        import time
        print("\nWaiting for indexing...")
        time.sleep(3)
        
        # Test queries that benefit from semantic understanding
        test_queries = [
            "How can AI help doctors?",
            "What are good ways to save money?", 
            "How to work from home effectively?",
            "What foods are healthy to eat?",
            "How to reduce environmental impact?"
        ]
        
        for query in test_queries:
            self.semantic_search(query)
            self.traditional_search(query)
            
            print("\n" + "="*80)
            input("Press Enter to continue to next query...")

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
        demo = SemanticSearchDemo()
        
        print("🎯 SEMANTIC SEARCH DEMO")
        print("This demo shows Azure AI Search's built-in semantic capabilities")
        print("Note: Requires Standard tier or higher for semantic search")
        
        demo.run_comparison_demo()
        
        cleanup = input("\nClean up demo index? (y/n): ").lower().strip()
        if cleanup == 'y':
            demo.cleanup()
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\nMake sure you have:")
        print("1. Valid Azure AI Search credentials in .env")
        print("2. Standard tier or higher for semantic search")


if __name__ == "__main__":
    main()
