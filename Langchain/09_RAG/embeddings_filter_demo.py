"""
Demonstration of EmbeddingsFilter similarity thresholds
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_openai_embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_embeddings_filter():
    """Show how different similarity thresholds affect document filtering"""
    
    print("=== EmbeddingsFilter Demonstration ===")
    
    # Initialize embeddings with error handling
    try:
        print("🔧 Initializing Azure OpenAI embeddings...")
        embeddings = create_azure_openai_embeddings()
        print("✅ Azure OpenAI embeddings initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Azure OpenAI embeddings: {e}")
        print("\n💡 Make sure:")
        print("1. You have a .env file with Azure OpenAI configuration")
        print("2. You're authenticated with Azure CLI (az login)")
        print("3. Your Azure OpenAI resource is accessible")
        return
    
    # Create test documents with varying relevance to Azure OpenAI
    documents = [
        Document(
            page_content="Azure OpenAI Service provides REST API access to OpenAI's language models like GPT-4 with enterprise security.",
            metadata={"source": "azure_openai_docs", "relevance": "high"}
        ),
        Document(
            page_content="Python programming is commonly used with Azure OpenAI APIs for building AI applications and chatbots.",
            metadata={"source": "python_guide", "relevance": "high"}
        ),
        Document(
            page_content="Machine learning models require training data and computational resources for optimal performance.",
            metadata={"source": "ml_basics", "relevance": "medium"}
        ),
        Document(
            page_content="Database management systems store and retrieve data efficiently using structured query language.",
            metadata={"source": "database_guide", "relevance": "low"}
        ),
        Document(
            page_content="Web development frameworks help create responsive user interfaces for modern applications.",
            metadata={"source": "web_dev", "relevance": "low"}
        ),
        Document(
            page_content="Azure cloud services provide scalable infrastructure for enterprise applications and workloads.",
            metadata={"source": "azure_overview", "relevance": "medium"}
        )
    ]
    
    # Create vector store with error handling
    try:
        print("🔄 Creating vector store with Azure OpenAI embeddings...")
        vectorstore = Chroma.from_documents(
            documents, 
            embeddings,
            collection_name="filter_demo",
            persist_directory="./filter_demo_db"
        )
        print("✅ Vector store created successfully")
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return
    
    # Base retriever (gets all documents)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Test query
    test_query = "How do I use Azure OpenAI API in Python?"
    
    print(f"Query: '{test_query}'")
    print(f"Total documents available: {len(documents)}")
    print()
    
    # Test different similarity thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"--- Similarity Threshold: {threshold} ---")
        
        try:
            # Create embeddings filter with current threshold
            embeddings_filter = EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=threshold
            )
            
            # Create compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever
            )
            
            # Get filtered documents
            filtered_docs = compression_retriever.get_relevant_documents(test_query)
            
            print(f"Documents kept: {len(filtered_docs)}/{len(documents)}")
            
            if filtered_docs:
                print("Kept documents:")
                for i, doc in enumerate(filtered_docs, 1):
                    relevance = doc.metadata.get('relevance', 'unknown')
                    source = doc.metadata.get('source', 'unknown')
                    print(f"  {i}. [{relevance.upper()}] {source}")
                    print(f"     Content: {doc.page_content[:80]}...")
            else:
                print("  ❌ No documents passed the filter!")
        
        except Exception as e:
            print(f"  ❌ Error with threshold {threshold}: {e}")
        
        print()
    
    # Manual similarity calculation demonstration
    print("=== Manual Similarity Calculation ===")
    try:
        print("🔄 Computing query and document embeddings...")
        query_embedding = embeddings.embed_query(test_query)
        
        print(f"Query: '{test_query}'")
        print("Document similarities:")
        
        for i, doc in enumerate(documents, 1):
            doc_embedding = embeddings.embed_documents([doc.page_content])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            relevance = doc.metadata.get('relevance', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            status = "✅ KEPT" if similarity >= 0.7 else "❌ FILTERED"
            
            print(f"  {i}. {similarity:.3f} [{relevance.upper()}] {source} - {status}")
            print(f"     {doc.page_content[:60]}...")
            print()
    except Exception as e:
        print(f"❌ Error calculating similarities: {e}")

def explain_threshold_strategies():
    """Explain different threshold strategies"""
    
    print("=== Threshold Strategy Guide ===")
    
    strategies = {
        "Very Strict (0.8-0.9)": {
            "use_case": "High-precision requirements, expert systems",
            "pros": ["Highest quality results", "Minimal noise", "Expert-level accuracy"],
            "cons": ["May miss relevant context", "Risk of empty results", "Overly conservative"],
            "example": "Medical diagnosis, legal advice, financial analysis"
        },
        "Balanced (0.6-0.8)": {
            "use_case": "General-purpose RAG systems",
            "pros": ["Good quality/quantity balance", "Reliable performance", "Flexible"],
            "cons": ["Some irrelevant docs may pass", "Requires tuning"],
            "example": "Customer support, documentation Q&A, general chatbots"
        },
        "Lenient (0.4-0.6)": {
            "use_case": "Exploratory search, broad context needed",
            "pros": ["More comprehensive results", "Better for complex queries", "Less filtering"],
            "cons": ["More noise", "Longer context", "Higher costs"],
            "example": "Research assistance, creative writing, brainstorming"
        },
        "No Filter (0.0-0.4)": {
            "use_case": "Maximum context, let LLM decide",
            "pros": ["No information loss", "LLM handles filtering", "Comprehensive"],
            "cons": ["Expensive", "Potential noise", "May confuse LLM"],
            "example": "Complex analysis, when unsure about relevance"
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f"\n📋 {strategy_name}")
        print(f"Use Case: {details['use_case']}")
        print(f"✅ Pros: {', '.join(details['pros'])}")
        print(f"❌ Cons: {', '.join(details['cons'])}")
        print(f"💡 Example: {details['example']}")
        print("-" * 50)

if __name__ == "__main__":
    try:
        print("🚀 Starting EmbeddingsFilter Demonstration with Azure OpenAI")
        print("="*60)
        
        # Check if we have the required configuration
        required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing environment variables: {missing_vars}")
            print("\n💡 Please set these in your .env file:")
            for var in missing_vars:
                print(f"   {var}=your_value_here")
            print("\n📖 Example .env file:")
            print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002")
            print("AZURE_OPENAI_API_VERSION=2024-02-15-preview")
            exit(1)
        
        demonstrate_embeddings_filter()
        print("\n" + "="*60 + "\n")
        explain_threshold_strategies()
        
        print("\n🎯 Key Takeaways:")
        print("• similarity_threshold=0.7 is a good balanced choice")
        print("• Higher threshold = fewer, more relevant documents")
        print("• Lower threshold = more documents, some less relevant")
        print("• Always test with your specific data and queries")
        print("• Monitor both quality and quantity of results")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your .env file has Azure OpenAI configuration")
        print("2. Authenticate with Azure CLI: az login")
        print("3. Verify your Azure OpenAI resource is accessible")
        print("4. Install required packages: pip install chromadb langchain-openai azure-identity")
