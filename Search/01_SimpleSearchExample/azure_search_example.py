"""
Azure AI Search Simple Example
This example demonstrates basic operations with Azure AI Search:
1. Creating a search index
2. Uploading documents
3. Performing searches
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

# Load environment variables
load_dotenv()

class AzureSearchExample:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "sample-index")
        
        if not self.endpoint or not self.key:
            raise ValueError("Please set AZURE_SEARCH_SERVICE_ENDPOINT and AZURE_SEARCH_ADMIN_KEY in .env file")
        
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

    def create_index(self):
        """Create a simple search index for books"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="author", type=SearchFieldDataType.String),
            SearchableField(name="description", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="rating", type=SearchFieldDataType.Double, sortable=True, filterable=True),
        ]
        
        index = SearchIndex(name=self.index_name, fields=fields)
        
        try:
            result = self.index_client.create_index(index)
            print(f"Index '{self.index_name}' created successfully!")
            return result
        except Exception as e:
            print(f"Error creating index: {e}")
            return None

    def upload_sample_documents(self):
        """Upload sample book documents to the index"""
        documents = [
            {
                "id": "1",
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "description": "A classic American novel about the Jazz Age and the American Dream",
                "category": "Fiction",
                "rating": 4.2
            },
            {
                "id": "2",
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "description": "A novel about racial injustice and childhood in the American South",
                "category": "Fiction",
                "rating": 4.5
            },
            {
                "id": "3",
                "title": "1984",
                "author": "George Orwell",
                "description": "A dystopian novel about totalitarianism and surveillance",
                "category": "Science Fiction",
                "rating": 4.3
            },
            {
                "id": "4",
                "title": "Pride and Prejudice",
                "author": "Jane Austen",
                "description": "A romantic novel about manners and marriage in Georgian England",
                "category": "Romance",
                "rating": 4.1
            },
            {
                "id": "5",
                "title": "The Catcher in the Rye",
                "author": "J.D. Salinger",
                "description": "A coming-of-age story about teenage rebellion and alienation",
                "category": "Fiction",
                "rating": 3.8
            }
        ]
        
        try:
            result = self.search_client.upload_documents(documents)
            print(f"Uploaded {len(documents)} documents successfully!")
            return result
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return None

    def simple_search(self, query):
        """Perform a simple text search"""
        try:
            results = self.search_client.search(search_text=query)
            print(f"\nSearch results for: '{query}'")
            print("-" * 50)
            
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Author: {result['author']}")
                print(f"Category: {result['category']}")
                print(f"Rating: {result['rating']}")
                print(f"Description: {result['description']}")
                print("-" * 30)
                
        except Exception as e:
            print(f"Error searching: {e}")

    def filtered_search(self, query, category_filter=None, min_rating=None):
        """Perform a search with filters"""
        try:
            filter_expression = []
            
            if category_filter:
                filter_expression.append(f"category eq '{category_filter}'")
            
            if min_rating:
                filter_expression.append(f"rating ge {min_rating}")
            
            filter_str = " and ".join(filter_expression) if filter_expression else None
            
            results = self.search_client.search(
                search_text=query,
                filter=filter_str,
                order_by=["rating desc"]
            )
            
            print(f"\nFiltered search results for: '{query}'")
            if filter_str:
                print(f"Filters: {filter_str}")
            print("-" * 50)
            
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Author: {result['author']}")
                print(f"Category: {result['category']}")
                print(f"Rating: {result['rating']}")
                print("-" * 30)
                
        except Exception as e:
            print(f"Error in filtered search: {e}")

    def delete_index(self):
        """Delete the search index"""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully!")
        except Exception as e:
            print(f"Error deleting index: {e}")


def main():
    """Main function to demonstrate Azure AI Search functionality"""
    try:
        # Initialize the search example
        search_example = AzureSearchExample()
        
        # Create the index
        print("Creating search index...")
        search_example.create_index()
        
        # Upload sample documents
        print("\nUploading sample documents...")
        search_example.upload_sample_documents()
        
        # Wait a moment for indexing to complete
        import time
        print("\nWaiting for indexing to complete...")
        time.sleep(3)
        
        # Perform various searches
        search_example.simple_search("American")
        search_example.simple_search("love romance")
        search_example.filtered_search("novel", category_filter="Fiction", min_rating=4.0)
        
        # Optional: Clean up (uncomment if you want to delete the index)
        search_example.delete_index()
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
