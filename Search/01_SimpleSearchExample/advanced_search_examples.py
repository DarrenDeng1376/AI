"""
Advanced Azure AI Search Examples
This module contains more advanced search scenarios and configurations
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
    ComplexField,
    ScoringProfile,
    TextWeights,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class AdvancedSearchExample:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = "advanced-search-index"
        
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def create_advanced_index(self):
        """Create an advanced search index with scoring profiles"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="tags", type=SearchFieldDataType.String),  # Changed to single string
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="publishDate", type=SearchFieldDataType.DateTimeOffset, sortable=True, filterable=True),
            SimpleField(name="rating", type=SearchFieldDataType.Double, sortable=True, filterable=True, facetable=True),
            SimpleField(name="viewCount", type=SearchFieldDataType.Int32, sortable=True),
        ]
        
        # Create a scoring profile to boost certain fields
        scoring_profile = ScoringProfile(
            name="boost-title-and-tags",
            text_weights=TextWeights(weights={
                "title": 3.0,
                "tags": 2.0,
                "content": 1.0
            })
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            scoring_profiles=[scoring_profile]
        )
        
        try:
            result = self.index_client.create_index(index)
            print(f"Advanced index '{self.index_name}' created successfully!")
            return result
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Index '{self.index_name}' already exists")
                return True
            else:
                print(f"Error creating advanced index: {e}")
                return None

    def upload_sample_data(self):
        """Upload sample documents for advanced search demo"""        
        documents = [
            {
                "id": "1",
                "title": "Python Machine Learning Tutorial",
                "content": "Learn how to build machine learning models using Python, scikit-learn, and pandas. This comprehensive guide covers data preprocessing, model training, and evaluation.",
                "tags": "python, machine learning, tutorial, scikit-learn",
                "category": "Technology",
                "publishDate": "2024-01-15T00:00:00Z",
                "rating": 4.8,
                "viewCount": 12500
            },
            {
                "id": "2", 
                "title": "JavaScript Web Development Basics",
                "content": "Master the fundamentals of JavaScript for web development. Learn DOM manipulation, event handling, and modern ES6+ features.",
                "tags": "javascript, web development, DOM, ES6",
                "category": "Technology",
                "publishDate": "2024-02-10T00:00:00Z",
                "rating": 4.5,
                "viewCount": 8300
            },
            {
                "id": "3",
                "title": "Data Science with R Programming",
                "content": "Explore data science concepts using R. Cover statistical analysis, data visualization with ggplot2, and machine learning algorithms.",
                "tags": "R, data science, statistics, ggplot2",
                "category": "Technology",
                "publishDate": "2024-01-20T00:00:00Z",
                "rating": 4.3,
                "viewCount": 6700
            },
            {
                "id": "4",
                "title": "Cloud Computing with Azure",
                "content": "Introduction to Microsoft Azure cloud services. Learn about virtual machines, storage, networking, and serverless computing.",
                "tags": "azure, cloud computing, serverless, devops",
                "category": "Cloud",
                "publishDate": "2024-03-05T00:00:00Z",
                "rating": 4.6,
                "viewCount": 9200
            },
            {
                "id": "5",
                "title": "Mobile App Development with React Native",
                "content": "Build cross-platform mobile applications using React Native. Learn navigation, state management, and native module integration.",
                "tags": "react native, mobile development, cross-platform, javascript",
                "category": "Mobile",
                "publishDate": "2024-02-28T00:00:00Z",
                "rating": 4.4,
                "viewCount": 7800
            },
            {
                "id": "6",
                "title": "Database Design and SQL Optimization",
                "content": "Master database design principles and SQL query optimization techniques. Learn indexing, normalization, and performance tuning.",
                "tags": "SQL, database, optimization, indexing",
                "category": "Database",
                "publishDate": "2024-01-08T00:00:00Z",
                "rating": 4.7,
                "viewCount": 11200
            }
        ]
        
        try:
            result = self.search_client.upload_documents(documents)
            print(f"✅ Uploaded {len(documents)} sample documents")
            return result
        except Exception as e:
            print(f"❌ Error uploading sample data: {e}")
            return None

    def faceted_search(self, query="*", facets=None):
        """Perform a search with faceted navigation"""
        if facets is None:
            facets = ["category", "rating"]
        
        try:
            results = self.search_client.search(
                search_text=query,
                facets=facets,
                top=10
            )
            
            print(f"\nFaceted search results for: '{query}'")
            print("-" * 50)
            
            # Convert results to list to check if empty and get facets
            result_list = list(results)
            
            # Try to get facets from the search results
            try:
                # Re-run search to get facets (results iterator can only be consumed once)
                facet_results = self.search_client.search(
                    search_text=query,
                    facets=facets,
                    top=10
                ).get_facets()
                
                if facet_results:
                    for facet_name, facet_values in facet_results.items():
                        print(f"\n{facet_name.upper()} facets:")
                        if facet_values:
                            for facet in facet_values:
                                print(f"  {facet['value']}: {facet['count']} items")
                        else:
                            print(f"  No {facet_name} facets found")
                else:
                    print("\n❌ No facet results returned")
            except Exception as facet_error:
                print(f"\n❌ Error getting facets: {facet_error}")
            
            # Display search results
            print(f"\nSearch Results: ({len(result_list)} found)")
            if result_list:
                for result in result_list:
                    print(f"Title: {result.get('title', 'N/A')}")
                    print(f"Category: {result.get('category', 'N/A')}")
                    print(f"Rating: {result.get('rating', 'N/A')}")
                    print("-" * 30)
            else:
                print("No documents found for this query")
                
        except Exception as e:
            print(f"Error in faceted search: {e}")

    def autocomplete_example(self, partial_text):
        """Demonstrate autocomplete functionality"""
        try:
            # Note: This requires a suggester to be defined in the index
            # For this example, we'll show how to implement a basic version
            results = self.search_client.search(
                search_text=f"{partial_text}*",
                search_mode="any",
                top=5
            )
            
            suggestions = []
            for result in results:
                title = result.get('title', '')
                if title.lower().startswith(partial_text.lower()):
                    suggestions.append(title)
            
            print(f"\nAutocomplete suggestions for '{partial_text}':")
            for suggestion in suggestions[:5]:
                print(f"  - {suggestion}")
                
        except Exception as e:
            print(f"Error in autocomplete: {e}")

    def search_with_highlighting(self, query):
        """Search with result highlighting"""
        try:
            results = self.search_client.search(
                search_text=query,
                highlight_fields="title,content",
                highlight_pre_tag="<mark>",
                highlight_post_tag="</mark>",
                top=5
            )
            
            print(f"\nSearch with highlighting for: '{query}'")
            print("-" * 50)
            
            for result in results:
                print(f"Title: {result.get('title', 'N/A')}")
                
                # Display highlights if available
                if '@search.highlights' in result:
                    highlights = result['@search.highlights']
                    if 'title' in highlights:
                        print(f"Highlighted Title: {highlights['title'][0]}")
                    if 'content' in highlights:
                        print(f"Highlighted Content: {highlights['content'][0]}")
                
                print("-" * 30)
                
        except Exception as e:
            print(f"Error in highlighted search: {e}")


def run_advanced_examples():
    """Run advanced search examples"""
    try:
        advanced_search = AdvancedSearchExample()
        
        print("🔧 Creating advanced search index...")
        index_result = advanced_search.create_advanced_index()
        
        if index_result:
            print("\n📝 Uploading sample data...")
            upload_result = advanced_search.upload_sample_data()
            
            if upload_result:
                # Wait a moment for indexing to complete
                import time
                print("⏳ Waiting for documents to be indexed...")
                time.sleep(3)
                
                print("\n" + "="*60)
                print("🚀 ADVANCED AZURE AI SEARCH EXAMPLES")
                print("="*60)
                
                # Example searches with properly configured data
                advanced_search.faceted_search("*")  # Search all documents to show facets
                advanced_search.faceted_search("python")  # Search for python content
                advanced_search.autocomplete_example("python")
                advanced_search.search_with_highlighting("machine learning")
                
                print("\n" + "="*60)
                print("✅ Advanced search demo completed!")
                print("="*60)
            else:
                print("❌ Failed to upload sample data. Cannot proceed with demo.")
        else:
            print("❌ Failed to create index. Cannot proceed with demo.")
        
    except Exception as e:
        print(f"Error running advanced examples: {e}")


if __name__ == "__main__":
    run_advanced_examples()
