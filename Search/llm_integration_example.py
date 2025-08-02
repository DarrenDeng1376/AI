"""
Azure AI Search with LLM Integration Examples
This example shows how to integrate Large Language Models with Azure AI Search
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
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

# For OpenAI integration (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Run: pip install openai")

load_dotenv()

class AISearchWithLLM:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = "ai-enhanced-search"
        
        # OpenAI configuration (optional)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
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

    def create_vector_search_index(self):
        """Create a search index with vector search capabilities"""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # For OpenAI text-embedding-ada-002
                vector_search_profile_name="my-vector-config"
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(
                name="my-vector-config",
                algorithm_configuration_name="my-hnsw"
            )],
            algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")]
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            result = self.index_client.create_index(index)
            print(f"Vector search index '{self.index_name}' created successfully!")
            return result
        except Exception as e:
            print(f"Error creating vector index: {e}")
            return None

    def get_text_embedding(self, text):
        """Generate embeddings using OpenAI (example)"""
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            print("OpenAI not configured. Returning mock embedding.")
            # Return a mock embedding vector (1536 dimensions)
            return [0.1] * 1536
        
        try:
            # Configure OpenAI client
            if self.openai_endpoint:
                # Azure OpenAI
                openai.api_type = "azure"
                openai.api_base = self.openai_endpoint
                openai.api_version = "2023-05-15"
                openai.api_key = self.openai_api_key
                engine = "text-embedding-ada-002"
            else:
                # Regular OpenAI
                openai.api_key = self.openai_api_key
                engine = "text-embedding-ada-002"
            
            response = openai.Embedding.create(
                input=text,
                engine=engine
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.1] * 1536  # Fallback mock embedding

    def upload_documents_with_embeddings(self):
        """Upload documents with vector embeddings"""
        documents = [
            {
                "id": "1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "category": "Technology"
            },
            {
                "id": "2", 
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "category": "Technology"
            },
            {
                "id": "3",
                "title": "Natural Language Processing",
                "content": "NLP enables computers to understand, interpret, and generate human language in a valuable way.",
                "category": "AI"
            }
        ]
        
        # Add embeddings to documents
        for doc in documents:
            content_for_embedding = f"{doc['title']} {doc['content']}"
            doc['content_vector'] = self.get_text_embedding(content_for_embedding)
        
        try:
            result = self.search_client.upload_documents(documents)
            print(f"Uploaded {len(documents)} documents with embeddings!")
            return result
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return None

    def hybrid_search(self, query):
        """Perform hybrid search (text + vector)"""
        try:
            # Generate embedding for the query
            query_vector = self.get_text_embedding(query)
            
            # Perform hybrid search
            results = self.search_client.search(
                search_text=query,  # Traditional text search
                vectors=[{
                    "value": query_vector,
                    "k_nearest_neighbors": 3,
                    "fields": "content_vector"
                }],  # Vector search
                top=5
            )
            
            print(f"\nHybrid search results for: '{query}'")
            print("-" * 50)
            
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Category: {result['category']}")
                print(f"Content: {result['content'][:100]}...")
                if '@search.score' in result:
                    print(f"Relevance Score: {result['@search.score']:.3f}")
                print("-" * 30)
                
        except Exception as e:
            print(f"Error in hybrid search: {e}")

    def rag_search(self, question):
        """Retrieval Augmented Generation (RAG) example"""
        try:
            # First, search for relevant documents
            results = self.search_client.search(
                search_text=question,
                top=3
            )
            
            # Collect context from search results
            context_parts = []
            for result in results:
                context_parts.append(f"Title: {result['title']}\nContent: {result['content']}")
            
            context = "\n\n".join(context_parts)
            
            print(f"\nRAG Search - Question: '{question}'")
            print("-" * 50)
            print("Retrieved Context:")
            print(context)
            
            # Here you would typically send the context + question to an LLM
            # For demonstration, we'll show the structure
            prompt = f"""
Context:
{context}

Question: {question}

Please answer the question based on the provided context.
"""
            
            print("\nGenerated Prompt for LLM:")
            print(prompt)
            
            # If you have OpenAI configured, you could generate an answer:
            # answer = self.generate_answer_with_llm(prompt)
            # print(f"\nAI Generated Answer: {answer}")
            
        except Exception as e:
            print(f"Error in RAG search: {e}")

    def semantic_search_example(self, query):
        """Example of semantic search capabilities"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="default",  # If configured
                query_caption="extractive",
                query_answer="extractive",
                top=5
            )
            
            print(f"\nSemantic search results for: '{query}'")
            print("-" * 50)
            
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Content: {result['content'][:100]}...")
                
                # Display semantic captions if available
                if '@search.captions' in result:
                    captions = result['@search.captions']
                    if captions:
                        print(f"Semantic Caption: {captions[0]['text']}")
                
                print("-" * 30)
                
        except Exception as e:
            print(f"Error in semantic search: {e}")


def demonstrate_ai_search_with_llm():
    """Demonstrate AI Search with LLM integration"""
    try:
        ai_search = AISearchWithLLM()
        
        print("Azure AI Search with LLM Integration Demo")
        print("=" * 60)
        
        # Create index with vector search
        print("\n1. Creating vector search index...")
        ai_search.create_vector_search_index()
        
        # Upload documents with embeddings
        print("\n2. Uploading documents with embeddings...")
        ai_search.upload_documents_with_embeddings()
        
        # Wait for indexing
        import time
        print("\nWaiting for indexing...")
        time.sleep(5)
        
        # Demonstrate different search types
        print("\n3. Hybrid Search (Text + Vector):")
        ai_search.hybrid_search("artificial intelligence algorithms")
        
        print("\n4. RAG Search Example:")
        ai_search.rag_search("What is deep learning?")
        
        print("\n5. Semantic Search Example:")
        ai_search.semantic_search_example("neural networks")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")


if __name__ == "__main__":
    demonstrate_ai_search_with_llm()
