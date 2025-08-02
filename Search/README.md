# Azure AI Search Simple Example

This project demonstrates how to use Azure AI Search to create, populate, and query a search index.

## Prerequisites

1. **Azure AI Search Service**: You need an Azure AI Search service. You can create one in the Azure portal.
2. **Python 3.7+**: Make sure you have Python installed.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   - Copy the `.env` file and update it with your Azure AI Search service details:
     - `AZURE_SEARCH_SERVICE_ENDPOINT`: Your search service URL (e.g., https://your-service.search.windows.net)
     - `AZURE_SEARCH_ADMIN_KEY`: Your admin key from the Azure portal
     - `AZURE_SEARCH_INDEX_NAME`: Name for your search index (optional, defaults to "sample-index")

## How to get Azure AI Search credentials

1. Go to the [Azure Portal](https://portal.azure.com)
2. Create or navigate to your Azure AI Search service
3. In the left menu, click on "Keys"
4. Copy the "Url" (this is your endpoint)
5. Copy one of the "Admin Keys" (primary or secondary)

## Running the Example

```bash
python azure_search_example.py
```

## What the example does

1. **Creates a search index** with fields for books (id, title, author, description, category, rating)
2. **Uploads sample documents** (5 classic books)
3. **Performs different types of searches**:
   - Simple text search
   - Filtered search by category and rating
   - Sorted results

## Key Features Demonstrated

- **Index Creation**: Defining searchable and filterable fields
- **Document Upload**: Adding documents to the search index
- **Text Search**: Full-text search across multiple fields
- **Filtering**: Restricting results based on specific criteria
- **Sorting**: Ordering results by rating or other fields

## Search Examples

The example performs these searches:
- Search for "American" (finds books with American themes)
- Search for "love romance" (finds romantic novels)
- Filtered search for "novel" in Fiction category with rating >= 4.0

## Additional Examples

### 🧠 Semantic vs Vector Search
See [`SEMANTIC_VS_VECTOR_SEARCH.md`](SEMANTIC_VS_VECTOR_SEARCH.md) for detailed comparison

**Demo Files:**
- [`semantic_search_demo.py`](semantic_search_demo.py) - Built-in semantic search
- [`vector_search_demo.py`](vector_search_demo.py) - Custom vector search with embeddings

### 🔍 Search Comparisons
- [`simple_search_comparison.py`](simple_search_comparison.py) - Traditional vs semantic search
- [`vector_vs_traditional_demo.py`](vector_vs_traditional_demo.py) - Full comparison with mock embeddings

### 📚 Learning Resources
- [`why_vector_search.py`](why_vector_search.py) - Understanding when vector search is needed
- [`vector_search_explained.py`](vector_search_explained.py) - Concept explanations
- [`LLM_INTEGRATION_GUIDE.md`](LLM_INTEGRATION_GUIDE.md) - LLM integration patterns

## Customization

You can modify the example to:
- Add more fields to the index schema
- Upload your own documents
- Implement different search scenarios
- Add faceted search capabilities
- Include suggestions and autocomplete

## Cleanup

Uncomment the `delete_index()` call in the main function if you want to clean up the index after testing.
