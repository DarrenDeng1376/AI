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
            SearchableField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="publishDate", type=SearchFieldDataType.DateTimeOffset, sortable=True, filterable=True),
            SimpleField(name="rating", type=SearchFieldDataType.Double, sortable=True),
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
            print(f"Error creating advanced index: {e}")
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
            
            # Display facets
            if hasattr(results, 'get_facets'):
                facet_results = results.get_facets()
                for facet_name, facet_values in facet_results.items():
                    print(f"\n{facet_name.upper()} facets:")
                    for facet in facet_values:
                        print(f"  {facet['value']}: {facet['count']} items")
            
            # Display search results
            print("\nSearch Results:")
            for result in results:
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"Category: {result.get('category', 'N/A')}")
                print(f"Rating: {result.get('rating', 'N/A')}")
                print("-" * 30)
                
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
                highlight_fields=["title", "content"],
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
        
        # Note: These examples require the advanced index to be created
        # and populated with appropriate data
        
        print("Advanced Azure AI Search Examples")
        print("=" * 50)
        
        # Example searches (these would work with properly configured data)
        advanced_search.faceted_search("technology")
        advanced_search.autocomplete_example("python")
        advanced_search.search_with_highlighting("machine learning")
        
    except Exception as e:
        print(f"Error running advanced examples: {e}")


if __name__ == "__main__":
    run_advanced_examples()
