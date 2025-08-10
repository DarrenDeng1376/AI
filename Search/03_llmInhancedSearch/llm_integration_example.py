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
from azure.identity import DefaultAzureCredential, ClientSecretCredential, AzureCliCredential

# For OpenAI integration
try:
    from openai import AzureOpenAI
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
        
        # Azure OpenAI configuration
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo")
        
        # Microsoft Entra ID configuration (optional)
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.use_entra_auth = os.getenv("USE_ENTRA_AUTH", "false").lower() == "true"
        
        # Initialize credentials
        self.credential = AzureKeyCredential(self.key)
        
        # Initialize Azure OpenAI client
        self.openai_client = self._initialize_openai_client()
        
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client with appropriate authentication"""
        if not OPENAI_AVAILABLE or not self.azure_openai_endpoint:
            return None
        
        if self.use_entra_auth:
            # Use token-based authentication for Azure OpenAI
            if self.tenant_id and self.client_id and self.client_secret:
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            else:
                credential = DefaultAzureCredential()
            
            # Store credential for token refresh
            self.azure_credential = credential
            
            # Get initial token for Azure Cognitive Services
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            
            return AzureOpenAI(
                api_version=self.azure_openai_api_version,
                azure_endpoint=self.azure_openai_endpoint,
                azure_ad_token=token.token
            )
        else:
            # Use API key authentication
            if not self.azure_openai_key:
                return None
            
            return AzureOpenAI(
                api_key=self.azure_openai_key,
                api_version=self.azure_openai_api_version,
                azure_endpoint=self.azure_openai_endpoint
            )
    
    def _refresh_openai_token_if_needed(self):
        """Refresh Azure OpenAI token if using Entra authentication"""
        if self.use_entra_auth and hasattr(self, 'azure_credential'):
            try:
                token = self.azure_credential.get_token("https://cognitiveservices.azure.com/.default")
                # Create a new client with the refreshed token
                self.openai_client = AzureOpenAI(
                    api_version=self.azure_openai_api_version,
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_ad_token=token.token
                )
            except Exception as e:
                print(f"Warning: Failed to refresh Azure OpenAI token: {e}")

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
        """Generate embeddings using Azure OpenAI"""
        if not self.openai_client:
            print("Azure OpenAI not configured. Returning mock embedding.")
            # Return a mock embedding vector (1536 dimensions)
            return [0.1] * 1536
        
        try:
            # Refresh token if using Entra authentication
            self._refresh_openai_token_if_needed()
            
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Try refreshing token once more if authentication error
            if "401" in str(e) or "authentication" in str(e).lower():
                try:
                    print("Attempting to refresh authentication token...")
                    self._refresh_openai_token_if_needed()
                    response = self.openai_client.embeddings.create(
                        input=text,
                        model=self.embedding_deployment_name
                    )
                    return response.data[0].embedding
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}")
            return [0.1] * 1536  # Fallback mock embedding

    def generate_answer_with_llm(self, prompt):
        """Generate an answer using Azure OpenAI chat model"""
        if not self.openai_client:
            return "Azure OpenAI not configured. Cannot generate answer."
        
        try:
            # Refresh token if using Entra authentication
            self._refresh_openai_token_if_needed()
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Try refreshing token once more if authentication error
            if "401" in str(e) or "authentication" in str(e).lower():
                try:
                    print("Attempting to refresh authentication token...")
                    self._refresh_openai_token_if_needed()
                    response = self.openai_client.chat.completions.create(
                        model=self.chat_deployment_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}")
            return "Sorry, I couldn't generate an answer due to an error."

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
            print(context[:500] + "..." if len(context) > 500 else context)
            
            # Generate prompt for the LLM
            prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so."""
            
            print("\nGenerating AI Answer...")
            
            # Generate answer using Azure OpenAI
            answer = self.generate_answer_with_llm(prompt)
            print(f"\nAI Generated Answer:")
            print(answer)
            
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
        
        # Check authentication method
        auth_method = "Microsoft Entra ID" if ai_search.use_entra_auth else "API Keys"
        print(f"🔐 Authentication Method: {auth_method}")
        
        # Check if Azure OpenAI is configured
        if not ai_search.openai_client:
            print("⚠️  Azure OpenAI not configured. Create a .env file with your Azure OpenAI credentials.")
            print("See .env.example for required environment variables.")
            print("The demo will run with mock embeddings only.\n")
        else:
            print("✅ Azure OpenAI configured successfully!\n")
        
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
        # ai_search.hybrid_search("artificial intelligence algorithms")
        
        print("\n4. RAG Search Example:")
        ai_search.rag_search("What is deep learning?")
        
        print("\n5. Semantic Search Example:")
        ai_search.semantic_search_example("neural networks")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        if "DefaultAzureCredential" in str(e):
            print("\n💡 Tip: Make sure you're logged in with Azure CLI (az login) or have proper credentials configured.")
        elif "ClientSecretCredential" in str(e):
            print("\n💡 Tip: Check your AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET values.")


if __name__ == "__main__":
    demonstrate_ai_search_with_llm()
