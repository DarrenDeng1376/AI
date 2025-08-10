# Azure AI Search with Azure OpenAI Integration

This example demonstrates how to integrate Azure AI Search with Azure OpenAI services for advanced search capabilities including vector search, hybrid search, and Retrieval Augmented Generation (RAG).

## Prerequisites

1. **Azure AI Search Service**: You need an Azure AI Search service with admin access
2. **Azure OpenAI Service**: You need an Azure OpenAI resource with deployed models
3. **Python 3.8+**

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Azure OpenAI Model Deployments

In your Azure OpenAI resource, you need to deploy these models:

- **Text Embedding Model**: `text-embedding-ada-002` or `text-embedding-3-small`
- **Chat Model**: `gpt-35-turbo` or `gpt-4`

Take note of your deployment names as you'll need them in the configuration.

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in your actual values:

```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:

```env
# Azure AI Search Configuration
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your-search-admin-key

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI Deployment Names (update these to match your actual deployments)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
```

### 4. Finding Your Azure Credentials

#### Azure AI Search:
- **Service Endpoint**: Found in Azure Portal > Your Search Service > Overview > URL
- **Admin Key**: Found in Azure Portal > Your Search Service > Settings > Keys

#### Azure OpenAI:
- **Endpoint**: Found in Azure Portal > Your OpenAI Resource > Overview > Endpoint
- **API Key**: Found in Azure Portal > Your OpenAI Resource > Resource Management > Keys and Endpoint
- **Deployment Names**: Found in Azure OpenAI Studio > Deployments

## Running the Example

```bash
python llm_integration_example.py
```

## Features Demonstrated

### 1. Vector Search Index Creation
- Creates a search index with vector search capabilities
- Configured for 1536-dimensional embeddings (text-embedding-ada-002)

### 2. Document Upload with Embeddings
- Uploads sample documents
- Generates and stores vector embeddings for each document

### 3. Hybrid Search
- Combines traditional text search with vector search
- Provides more relevant results by leveraging both approaches

### 4. RAG (Retrieval Augmented Generation)
- Searches for relevant documents
- Uses Azure OpenAI to generate contextual answers
- Demonstrates the full RAG pipeline

### 5. Semantic Search
- Uses Azure AI Search's semantic search capabilities
- Provides enhanced search results with captions and answers

## Code Structure

- `AISearchWithLLM`: Main class handling all search and LLM operations
- `get_text_embedding()`: Generates embeddings using Azure OpenAI
- `generate_answer_with_llm()`: Generates answers using Azure OpenAI chat models
- `hybrid_search()`: Demonstrates hybrid text + vector search
- `rag_search()`: Full RAG implementation with answer generation
- `semantic_search_example()`: Semantic search capabilities

## Troubleshooting

1. **Import errors**: Make sure you've installed all requirements
2. **Authentication errors**: Verify your API keys and endpoints
3. **Model not found**: Ensure your deployment names match the actual deployments in Azure OpenAI
4. **Quota exceeded**: Check your Azure OpenAI quota and usage limits

## Next Steps

- Customize the document schema for your use case
- Implement more advanced search filters
- Add conversation memory for multi-turn RAG
- Integrate with your own data sources
