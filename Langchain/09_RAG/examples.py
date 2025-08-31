"""
RAG (Retrieval-Augmented Generation) Examples

This module demonstrates how to build RAG systems with LangChain using Azure OpenAI:
1. Basic RAG implementation with Azure OpenAI
2. Document processing and chunking
3. Vector store operations
4. Advanced retrieval techniques
5. RAG system evaluation
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv

load_dotenv()

# Example 1: Basic RAG System with Azure OpenAI
def basic_rag_example():
    """
    Demonstrates a simple RAG system with Azure OpenAI and local documents
    """
    print("=== Example 1: Basic RAG System with Azure OpenAI ===")
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain_core.documents import Document
    
    # Initialize Azure OpenAI components
    print("🔧 Initializing Azure OpenAI components...")
    llm = create_azure_chat_openai(temperature=0)
    embeddings = create_azure_openai_embeddings()
    
    # Sample documents (in real scenario, load from files)
    documents = [
        Document(page_content="""
        LangChain is a framework for developing applications powered by language models. 
        It enables applications that are context-aware and can reason about information. 
        LangChain provides tools for prompt management, chains, memory, and agents.
        """, metadata={"source": "langchain_intro.txt", "category": "framework"}),
        
        Document(page_content="""
        Vector databases store high-dimensional vectors representing data embeddings. 
        They enable fast similarity search and are essential for RAG systems. 
        Popular vector databases include Chroma, Pinecone, and Weaviate.
        """, metadata={"source": "vector_db.txt", "category": "database"}),
        
        Document(page_content="""
        Embeddings are numerical representations of text that capture semantic meaning. 
        Similar texts have similar embeddings. Azure OpenAI's text-embedding-ada-002 is 
        commonly used for creating embeddings with enterprise security.
        """, metadata={"source": "embeddings.txt", "category": "ai_concepts"}),
        
        Document(page_content="""
        Azure OpenAI provides enterprise-grade AI services with robust security, compliance, 
        and scalability. It offers GPT models, embedding models, and fine-tuning capabilities 
        with Azure's security and privacy guarantees.
        """, metadata={"source": "azure_openai.txt", "category": "azure_services"})
    ]
    
    print(f"📚 Processing {len(documents)} documents...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Created {len(chunks)} text chunks")
    
    # Create embeddings and vector store
    print("🔄 Creating vector store with Azure OpenAI embeddings...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_rag",
        collection_name="rag_knowledge_base"
    )
    
    # Create retrieval chain
    print("🤖 Setting up RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True
    )
    
    # Test queries
    queries = [
        "What is LangChain and what does it provide?",
        "How do vector databases enable semantic search?",
        "What are the benefits of using Azure OpenAI for embeddings?",
        "Explain the relationship between embeddings and similarity search"
    ]
    
    print("\n🔍 Testing RAG system with sample queries:")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = qa_chain({"query": query})
            print(f"🎯 Answer: {result['result']}")
            print(f"\n📖 Sources used:")
            for j, doc in enumerate(result['source_documents'], 1):
                source = doc.metadata.get('source', 'Unknown')
                category = doc.metadata.get('category', 'Unknown')
                print(f"  {j}. {source} (Category: {category})")
                print(f"     Preview: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        if i < len(queries):
            print()
    
    print("\n" + "="*60 + "\n")


# Example 2: Advanced Document Processing
def document_processing_example():
    """
    Shows advanced document processing techniques with Azure OpenAI
    """
    print("=== Example 2: Advanced Document Processing ===")
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
    from langchain_core.documents import Document
    import tiktoken
    
    # Sample long document about AI and Azure
    long_text = """
    Artificial Intelligence (AI) has transformed numerous industries and continues to evolve rapidly. 
    Microsoft Azure provides comprehensive AI services through Azure Cognitive Services and Azure OpenAI.
    
    Machine Learning, a subset of AI, enables computers to learn and improve from experience without 
    being explicitly programmed. Azure Machine Learning provides a cloud-based environment for 
    training, deploying, and managing machine learning models at scale.
    
    Deep Learning, a subset of Machine Learning, uses neural networks with multiple layers to model 
    and understand complex patterns in data. Azure's GPU-powered virtual machines provide the 
    computational power needed for deep learning workloads.
    
    Natural Language Processing (NLP) is another crucial area of AI that focuses on the interaction 
    between computers and human language. Azure OpenAI Service provides access to advanced language 
    models like GPT-4, which excel at understanding and generating human-like text.
    
    Computer Vision enables machines to interpret and understand visual information from the world. 
    Azure Computer Vision API can analyze images and videos to extract meaningful information, 
    detect objects, read text, and identify faces.
    
    The future of AI holds great promise with emerging technologies like Generative AI, which can 
    create new content including text, images, and code. Azure OpenAI Service makes these capabilities 
    accessible through secure, enterprise-ready APIs with built-in responsible AI features.
    
    Azure's AI services are designed with enterprise security, compliance, and scalability in mind. 
    They integrate seamlessly with other Azure services and provide comprehensive monitoring, 
    logging, and analytics capabilities for production AI applications.
    """ * 2  # Repeat to make it longer
    
    document = Document(
        page_content=long_text, 
        metadata={"source": "azure_ai_guide.txt", "topic": "Azure AI Services"}
    )
    
    print(f"📄 Original document length: {len(long_text)} characters")
    print(f"📊 Word count: {len(long_text.split())} words")
    
    # Different splitting strategies
    splitters = {
        "Small Chunks (Conservative)": RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,    # 10% overlap
            separators=["\n\n", "\n", " ", ""]
        ),
        "Small Chunks (Balanced)": RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,    # 16.7% overlap
            separators=["\n\n", "\n", " ", ""]
        ),
        "Medium Chunks (Conservative)": RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,    # 10% overlap
            separators=["\n\n", "\n", " ", ""]
        ),
        "Medium Chunks (Balanced)": RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,   # 16.7% overlap - YOUR CURRENT SETTING
            separators=["\n\n", "\n", " ", ""]
        ),
        "Large Chunks (Balanced)": RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,   # 15% overlap
            separators=["\n\n", "\n", " ", ""]
        ),
        "Token-based": TokenTextSplitter(
            chunk_size=150,
            chunk_overlap=30     # 20% overlap
        )
    }
    
    print("\n🔧 Comparing different text splitting strategies:")
    print("=" * 60)
    
    chunk_analysis = {}
    
    for name, splitter in splitters.items():
        print(f"\n📋 {name} Splitter:")
        chunks = splitter.split_documents([document])
        
        # Calculate statistics
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        # Calculate overlap ratio and effectiveness metrics
        # Different splitters store overlap differently
        if hasattr(splitter, '_chunk_overlap'):
            chunk_overlap = splitter._chunk_overlap
            chunk_size = splitter._chunk_size
        elif hasattr(splitter, 'chunk_overlap'):
            chunk_overlap = splitter.chunk_overlap
            chunk_size = splitter.chunk_size
        else:
            chunk_overlap = 0
            chunk_size = getattr(splitter, 'chunk_size', getattr(splitter, '_chunk_size', 1000))
        
        overlap_ratio = (chunk_overlap / chunk_size * 100) if chunk_size > 0 else 0
        
        print(f"  🔧 Debug - Splitter type: {type(splitter).__name__}")
        print(f"  🔧 Debug - Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        chunk_analysis[name] = {
            "count": len(chunks),
            "avg_size": avg_size,
            "min_size": min_size,
            "max_size": max_size,
            "overlap_ratio": overlap_ratio,
            "storage_efficiency": 1 - (overlap_ratio / 100)  # Lower overlap = higher efficiency
        }
        
        print(f"  📊 Number of chunks: {len(chunks)}")
        print(f"  📏 Average chunk size: {avg_size:.0f} characters")
        print(f"  📐 Size range: {min_size} - {max_size} characters")
        print(f"  📈 Overlap ratio: {overlap_ratio:.1f}%")
        print(f"  💾 Storage efficiency: {chunk_analysis[name]['storage_efficiency']:.1f}")
        print(f"  📝 First chunk preview: {chunks[0].page_content[:100]}...")
        
        # Show chunk boundaries for first few chunks
        if len(chunks) >= 2:
            print(f"  🔗 Chunk overlap analysis:")
            chunk1_end = chunks[0].page_content[-50:].strip()
            chunk2_start = chunks[1].page_content[:50].strip()
            
            # Calculate actual overlap by finding common text
            common_words = set(chunk1_end.split()) & set(chunk2_start.split())
            overlap_effectiveness = len(common_words) / max(len(chunk1_end.split()), 1)
            
            print(f"     Chunk 1 ends: ...{chunk1_end}")
            print(f"     Chunk 2 starts: {chunk2_start}...")
            print(f"     Context preservation: {overlap_effectiveness:.1%}")
    
    # Recommend optimal strategy with detailed analysis
    print(f"\n💡 Chunk Size vs Overlap Analysis:")
    print("=" * 50)
    
    for name, stats in chunk_analysis.items():
        overlap_ratio = stats.get('overlap_ratio', 0)
        storage_eff = stats.get('storage_efficiency', 1)
        
        # Categorize overlap strategy
        if overlap_ratio < 10:
            overlap_category = "Conservative"
        elif overlap_ratio <= 15:
            overlap_category = "Balanced ✅"
        elif overlap_ratio <= 20:
            overlap_category = "Aggressive"
        else:
            overlap_category = "Very Aggressive"
        
        # Determine use case
        if stats['avg_size'] < 400:
            use_case = "Quick queries, FAQ"
        elif stats['avg_size'] < 800:
            use_case = "General purpose, documentation"
        else:
            use_case = "Complex queries, research"
        
        print(f"\n• {name}:")
        print(f"  📏 Chunk Size: {stats['avg_size']:.0f} chars")
        print(f"  📈 Overlap: {overlap_ratio:.1f}% ({overlap_category})")
        print(f"  💾 Storage Efficiency: {storage_eff:.1%}")
        print(f"  🎯 Best for: {use_case}")
        print(f"  📊 Memory impact: {'Low' if stats['count'] < 8 else 'Medium' if stats['count'] < 15 else 'High'}")
    
    print(f"\n🏆 Overlap Best Practices Summary:")
    print("-" * 40)
    print("📋 CONSERVATIVE (5-10% overlap):")
    print("   • Use for: Simple Q&A, well-structured docs")
    print("   • Benefits: Lower storage, faster search")
    print("   • Risk: Context loss at boundaries")
    print()
    print("📋 BALANCED (10-20% overlap) ✅ RECOMMENDED:")
    print("   • Use for: Most general purposes")
    print("   • Benefits: Good context preservation")
    print("   • Your setting (16.7%) fits here perfectly!")
    print()
    print("📋 AGGRESSIVE (20%+ overlap):")
    print("   • Use for: Complex technical content")
    print("   • Benefits: Maximum context preservation")
    print("   • Cost: Higher storage and processing")
    
    print(f"\n🎯 Optimization Tips:")
    print("• Test with your actual data and query types")
    print("• Monitor retrieval quality vs storage costs")
    print("• Consider domain-specific separators")
    print("• Use token-based splitting for precise control")
    
    print("\n" + "="*60 + "\n")


# Example 3: Multiple Vector Store Types with Azure OpenAI
def vector_stores_example():
    """
    Compares different vector store implementations using Azure OpenAI embeddings
    """
    print("=== Example 3: Vector Store Comparison with Azure OpenAI ===")
    
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_core.documents import Document
    import time
    
    # Initialize Azure OpenAI embeddings
    print("🔧 Initializing Azure OpenAI embeddings...")
    embeddings = create_azure_openai_embeddings()
    
    # Sample documents about Azure and AI
    docs = [
        Document(
            page_content="Python is a versatile programming language widely used in Azure AI services and machine learning projects.",
            metadata={"category": "programming", "relevance": "high"}
        ),
        Document(
            page_content="Azure Machine Learning provides cloud-based tools for training, deploying, and managing ML models at enterprise scale.",
            metadata={"category": "azure_services", "relevance": "high"}
        ),
        Document(
            page_content="Natural language processing with Azure OpenAI enables advanced text analysis, generation, and understanding capabilities.",
            metadata={"category": "nlp", "relevance": "high"}
        ),
        Document(
            page_content="Vector databases enable semantic search by storing high-dimensional embeddings for fast similarity comparisons.",
            metadata={"category": "databases", "relevance": "medium"}
        ),
        Document(
            page_content="Azure Cognitive Services offers pre-built AI models for vision, speech, language, and decision-making tasks.",
            metadata={"category": "azure_services", "relevance": "high"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines information retrieval with language generation for more accurate AI responses.",
            metadata={"category": "ai_techniques", "relevance": "high"}
        )
    ]
    
    print(f"📚 Testing with {len(docs)} documents...")
    
    # Create different vector stores and measure performance
    vector_stores = {}
    creation_times = {}
    
    print("\n🔄 Creating vector stores...")
    
    # Chroma
    start_time = time.time()
    chroma_store = Chroma.from_documents(
        docs, 
        embeddings, 
        collection_name="azure_ai_docs",
        persist_directory="./chroma_comparison"
    )
    creation_times["Chroma"] = time.time() - start_time
    vector_stores["Chroma"] = chroma_store
    
    # FAISS
    start_time = time.time()
    faiss_store = FAISS.from_documents(docs, embeddings)
    creation_times["FAISS"] = time.time() - start_time
    vector_stores["FAISS"] = faiss_store
    
    # Test queries with performance measurement
    test_queries = [
        "Azure machine learning services",
        "Python programming for AI",
        "Natural language processing capabilities",
        "Vector database functionality"
    ]
    
    print(f"\n🔍 Performance comparison:")
    print("=" * 60)
    
    results_comparison = {}
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        print("-" * 40)
        
        query_results = {}
        
        for name, vs in vector_stores.items():
            # Measure search time
            start_time = time.time()
            results = vs.similarity_search_with_score(query, k=2)
            search_time = time.time() - start_time
            
            query_results[name] = {
                "search_time": search_time,
                "results": results
            }
            
            print(f"\n🏪 {name} Vector Store:")
            print(f"   ⏱️  Search time: {search_time*1000:.2f}ms")
            print(f"   📊 Results:")
            
            for i, (result, score) in enumerate(results, 1):
                category = result.metadata.get('category', 'unknown')
                relevance = result.metadata.get('relevance', 'unknown')
                print(f"     {i}. Score: {score:.4f} | Category: {category} | Relevance: {relevance}")
                print(f"        Content: {result.page_content[:80]}...")
        
        results_comparison[query] = query_results
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("=" * 40)
    
    for name in vector_stores.keys():
        avg_search_time = sum(
            results_comparison[q][name]["search_time"] 
            for q in test_queries
        ) / len(test_queries)
        
        print(f"\n🏪 {name}:")
        print(f"   🏗️  Creation time: {creation_times[name]*1000:.2f}ms")
        print(f"   🔍 Average search time: {avg_search_time*1000:.2f}ms")
        print(f"   💾 Storage: {'Persistent' if name == 'Chroma' else 'In-memory'}")
        print(f"   🎯 Best for: {'Production/Persistence' if name == 'Chroma' else 'Fast prototyping'}")
    
    # Demonstrate metadata filtering with Chroma
    print(f"\n🏷️  Metadata Filtering Example (Chroma only):")
    print("-" * 40)
    
    filtered_results = chroma_store.similarity_search(
        "Azure services",
        k=3,
        filter={"category": "azure_services"}
    )
    
    print(f"Query: 'Azure services' (filtered by category='azure_services')")
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. {result.page_content}")
        print(f"     Category: {result.metadata['category']}")
    
    print("\n" + "="*60 + "\n")


# Example 4: Advanced RAG with Modern LCEL Patterns
def advanced_rag_example():
    """
    Implements advanced RAG using modern LangChain Expression Language (LCEL)
    """
    print("=== Example 4: Advanced RAG with Modern LCEL ===")
    
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    
    # Initialize Azure OpenAI components
    print("🔧 Initializing Azure OpenAI components...")
    llm = create_azure_chat_openai(temperature=0.3)
    embeddings = create_azure_openai_embeddings()
    
    # Create more detailed documents about Azure AI
    documents = [
        Document(page_content="""
        Azure OpenAI Service provides REST API access to OpenAI's powerful language models including 
        GPT-4, GPT-3.5, and Embeddings model series. These models can be used for content generation, 
        summarization, semantic search, and natural language to code translation. Azure OpenAI 
        co-develops the APIs with OpenAI, ensuring compatibility and a smooth transition from OpenAI 
        to Azure OpenAI. The service also provides enterprise-grade security with private networking, 
        regional availability, and responsible AI content filtering.
        """, metadata={"source": "azure_openai_overview", "category": "services", "confidence": "high"}),
        
        Document(page_content="""
        Python is the most popular programming language for AI and machine learning development. 
        Azure supports Python through various services including Azure Machine Learning, Azure Functions, 
        and Azure App Service. Python developers can use familiar frameworks like TensorFlow, PyTorch, 
        and scikit-learn with Azure's cloud infrastructure. The Azure SDK for Python provides 
        comprehensive tools for managing Azure resources and services programmatically.
        """, metadata={"source": "python_azure_integration", "category": "development", "confidence": "high"}),
        
        Document(page_content="""
        Data science workflows in Azure involve multiple services working together. Azure Machine Learning 
        provides model development and deployment capabilities. Azure Synapse Analytics handles large-scale 
        data processing. Azure Data Factory orchestrates data movement and transformation. Power BI 
        creates visualizations and dashboards. These services integrate seamlessly to create end-to-end 
        data science solutions with governance, security, and compliance features built-in.
        """, metadata={"source": "azure_data_science", "category": "analytics", "confidence": "medium"}),
        
        Document(page_content="""
        Vector databases and semantic search are transforming how we interact with data. Traditional 
        keyword search looks for exact matches, while semantic search understands meaning and context. 
        Vector embeddings capture semantic relationships between words and concepts. Azure Cognitive 
        Search now includes vector search capabilities, enabling hybrid search that combines keyword 
        and semantic search for optimal results.
        """, metadata={"source": "vector_search_guide", "category": "search", "confidence": "high"})
    ]
    
    print(f"📚 Processing {len(documents)} detailed documents...")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings,
        collection_name="advanced_rag_demo",
        persist_directory="./advanced_rag_db"
    )
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Add compression with embeddings filter
    print("🔧 Setting up contextual compression...")
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.7  # Only include docs above this similarity
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )
    
    # Modern LCEL RAG Chain
    print("🚀 Creating modern LCEL RAG chain...")
    
    # Custom prompt template
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Azure AI consultant. Use the following context to provide 
        comprehensive and accurate answers about Azure AI services, development practices, and best practices.
        
        Context:
        {context}
        
        Guidelines:
        - Provide detailed, technical answers when possible
        - Mention specific Azure services when relevant
        - Include practical recommendations
        - If information is incomplete, acknowledge limitations
        """),
        ("human", "{question}")
    ])
    
    # Helper function to format documents with metadata
    def format_docs_with_metadata(docs):
        if not docs:
            return "No relevant context found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', f'Document_{i}')
            category = doc.metadata.get('category', 'General')
            confidence = doc.metadata.get('confidence', 'Unknown')
            
            formatted.append(f"""
            [Source {i}: {source} | Category: {category} | Confidence: {confidence}]
            {doc.page_content}
            """)
        return "\n".join(formatted)
    
    # Create the LCEL chain
    rag_chain = (
        {
            "context": compression_retriever | format_docs_with_metadata,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Test advanced queries
    advanced_queries = [
        "How can I integrate Azure OpenAI with Python applications for enterprise use?",
        "What's the difference between traditional keyword search and semantic search in Azure?",
        "What Azure services should I use for a complete data science workflow?",
        "How does Azure OpenAI ensure security and compliance for enterprise applications?"
    ]
    
    print("\n🤖 Testing Advanced RAG System:")
    print("=" * 60)
    
    for i, query in enumerate(advanced_queries, 1):
        print(f"\n📋 Advanced Query {i}:")
        print(f"❓ {query}")
        print("-" * 50)
        
        try:
            # Show retrieval process
            retrieved_docs = compression_retriever.get_relevant_documents(query)
            print(f"🔍 Retrieved {len(retrieved_docs)} relevant documents (after compression)")
            
            # Get RAG response
            response = rag_chain.invoke(query)
            print(f"\n🎯 Expert Response:")
            print(response)
            
            # Show source breakdown
            if retrieved_docs:
                print(f"\n📖 Sources analyzed:")
                for j, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source', f'Source_{j}')
                    category = doc.metadata.get('category', 'Unknown')
                    confidence = doc.metadata.get('confidence', 'Unknown')
                    print(f"  {j}. {source} (Category: {category}, Confidence: {confidence})")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        if i < len(advanced_queries):
            print("\n" + "="*60)
    
    # Demonstrate streaming capability
    print(f"\n🌊 Streaming Response Example:")
    print("-" * 40)
    print("Query: 'Explain Azure OpenAI benefits for developers'")
    print("Response: ", end="", flush=True)
    
    try:
        for chunk in rag_chain.stream("Explain Azure OpenAI benefits for developers"):
            if chunk:
                print(chunk, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error with streaming: {e}")
    
    print("\n💡 Advanced Features Demonstrated:")
    print("✅ Contextual compression (removes irrelevant docs)")
    print("✅ Embeddings-based filtering")
    print("✅ Modern LCEL syntax")
    print("✅ Rich metadata integration")
    print("✅ Streaming responses")
    print("✅ Custom prompt engineering")
    
    print("\n" + "="*60 + "\n")


# Example 5: RAG with Custom Prompts and Azure OpenAI
def custom_prompt_rag_example():
    """
    Shows how to customize RAG prompts for specific use cases with Azure OpenAI
    """
    print("=== Example 5: RAG with Custom Prompts ===")
    
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    # Initialize Azure OpenAI components
    print("🔧 Initializing Azure OpenAI components...")
    llm = create_azure_chat_openai(temperature=0.3)
    embeddings = create_azure_openai_embeddings()
    
    # Create documents about a fictional Azure-focused company
    documents = [
        Document(page_content="""
        AzureTech Solutions was founded in 2021 and specializes in Azure-powered AI software solutions. 
        The company has 200 employees across offices in Seattle, London, and Singapore. 
        AzureTech's mission is to democratize Azure AI technology for enterprises of all sizes, 
        with a focus on security, compliance, and scalability.
        """, metadata={"source": "company_overview", "category": "company_info", "last_updated": "2024-01"}),
        
        Document(page_content="""
        AzureTech offers four main products: 
        1. DataMind Azure (analytics platform using Azure Synapse and Power BI)
        2. ChatBot Pro Azure (customer service automation with Azure OpenAI)
        3. PredictFlow Azure (predictive analytics using Azure Machine Learning)
        4. SecureDoc Azure (document processing with Azure Cognitive Services)
        All products are built on Azure infrastructure with enterprise-grade security and compliance.
        """, metadata={"source": "product_catalog", "category": "products", "last_updated": "2024-02"}),
        
        Document(page_content="""
        AzureTech's customer support operates 24/7 with Azure-powered chatbots and human experts. 
        Response time is guaranteed within 1 hour for Enterprise customers and 4 hours for Standard customers.
        The company offers three tiers: Standard ($199/month), Professional ($599/month), 
        and Enterprise (custom pricing starting at $2000/month). All plans include Azure infrastructure, 
        free onboarding, and comprehensive training programs.
        """, metadata={"source": "support_pricing", "category": "support", "last_updated": "2024-03"}),
        
        Document(page_content="""
        AzureTech maintains SOC 2 Type II, ISO 27001, and Azure compliance certifications. 
        All data is processed within customer-specified Azure regions with end-to-end encryption. 
        The company offers GDPR compliance, HIPAA compliance for healthcare customers, and 
        custom data residency options. Regular security audits are conducted by third-party firms.
        """, metadata={"source": "security_compliance", "category": "security", "last_updated": "2024-02"})
    ]
    
    print(f"📚 Processing {len(documents)} company documents...")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings,
        collection_name="azuretech_knowledge_base",
        persist_directory="./azuretech_db"
    )
    
    # Create different RAG systems for different use cases
    
    # 1. Customer Support RAG
    print("\n🎧 Customer Support RAG System:")
    print("-" * 40)
    
    support_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and professional customer service representative for AzureTech Solutions, 
        an Azure-focused AI company. Use the following context to answer customer questions accurately and professionally.

        Context: {context}

        Guidelines:
        - Always be polite and professional
        - Provide specific product/pricing information when available
        - If information isn't in the context, politely say so and offer to connect them with a specialist
        - Mention relevant Azure services when appropriate
        - Focus on how our Azure-based solutions can help their business
        """),
        ("human", "Customer Question: {question}")
    ])
    
    def format_support_docs(docs):
        if not docs:
            return "No relevant information found in knowledge base."
        return "\n\n".join([f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}" for doc in docs])
    
    support_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    support_rag = (
        {
            "context": support_retriever | format_support_docs,
            "question": RunnablePassthrough()
        }
        | support_prompt
        | llm
        | StrOutputParser()
    )
    
    # 2. Sales RAG
    print("💼 Sales-focused RAG System:")
    print("-" * 40)
    
    sales_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable sales consultant for AzureTech Solutions. Use the context to help 
        potential customers understand our Azure-powered AI solutions and their business value.

        Context: {context}

        Guidelines:
        - Emphasize business benefits and ROI
        - Highlight Azure integration and enterprise features
        - Suggest appropriate products/tiers based on customer needs
        - Mention security and compliance when relevant
        - Always include a call-to-action
        """),
        ("human", "Prospect Question: {question}")
    ])
    
    sales_rag = (
        {
            "context": support_retriever | format_support_docs,
            "question": RunnablePassthrough()
        }
        | sales_prompt
        | llm
        | StrOutputParser()
    )
    
    # 3. Technical Documentation RAG
    print("📖 Technical Documentation RAG System:")
    print("-" * 40)
    
    tech_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical documentation assistant for AzureTech Solutions. Provide 
        detailed, accurate technical information based on the context.

        Context: {context}

        Guidelines:
        - Provide precise technical details
        - Include relevant Azure service names and features
        - Mention compliance and security specifications
        - Use technical language appropriate for developers/IT professionals
        - Include version/update information when available
        """),
        ("human", "Technical Question: {question}")
    ])
    
    tech_rag = (
        {
            "context": support_retriever | format_support_docs,
            "question": RunnablePassthrough()
        }
        | tech_prompt
        | llm
        | StrOutputParser()
    )
    
    # Test different RAG systems
    test_scenarios = [
        {
            "system": "Customer Support",
            "rag": support_rag,
            "questions": [
                "What products does AzureTech offer?",
                "How much does the Professional plan cost?",
                "What's your customer support response time?",
                "Do you offer refunds?"  # Not in knowledge base
            ]
        },
        {
            "system": "Sales",
            "rag": sales_rag,
            "questions": [
                "How can AzureTech help my enterprise with AI?",
                "What makes your Azure solutions different from competitors?",
                "What compliance certifications do you have?"
            ]
        },
        {
            "system": "Technical Documentation",
            "rag": tech_rag,
            "questions": [
                "What Azure services are used in your products?",
                "What security certifications does AzureTech maintain?",
                "Which Azure regions can process our data?"
            ]
        }
    ]
    
    print("\n🧪 Testing Different RAG Systems:")
    print("=" * 60)
    
    for scenario in test_scenarios:
        print(f"\n🎯 {scenario['system']} RAG System:")
        print("=" * 40)
        
        for i, question in enumerate(scenario['questions'], 1):
            print(f"\n📋 Question {i}: {question}")
            print("-" * 30)
            
            try:
                response = scenario['rag'].invoke(question)
                print(f"Response: {response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print(f"\n💡 Custom Prompt Benefits:")
    print("✅ Role-specific responses")
    print("✅ Consistent brand voice")
    print("✅ Appropriate technical level")
    print("✅ Targeted call-to-actions")
    print("✅ Context-aware limitations")
    
    # Demonstrate metadata filtering
    print(f"\n🏷️  Metadata Filtering Example:")
    print("-" * 40)
    
    # Get only product-related information
    product_filter_results = vectorstore.similarity_search(
        "what products are available",
        k=2,
        filter={"category": "products"}
    )
    
    print("Query: 'what products are available' (filtered by category='products')")
    for i, result in enumerate(product_filter_results, 1):
        print(f"  {i}. Source: {result.metadata.get('source', 'Unknown')}")
        print(f"     Content: {result.page_content[:150]}...")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("RAG (Retrieval-Augmented Generation) Examples with Azure OpenAI")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        # test_llm = create_azure_chat_openai(temperature=0)
        # test_embeddings = create_azure_openai_embeddings()
        
        # Quick connectivity test
        # test_response = test_llm.invoke("Say 'Azure OpenAI connection successful!'")
        # test_embedding = test_embeddings.embed_query("test")
        
        # print(f"✅ LLM Response: {test_response.content}")
        #print(f"✅ Embeddings: Generated {len(test_embedding)}-dimensional vector")
        print()
        
        # Run all examples
        print("🚀 Starting RAG Examples...")
        print()
        
        # basic_rag_example()
        # document_processing_example()
        vector_stores_example()
        advanced_rag_example()
        custom_prompt_rag_example()
        
        print("🎉 All RAG examples completed successfully!")
        print("\n💡 Next Steps:")
        print("• Explore production deployment patterns")
        print("• Implement evaluation metrics for RAG systems")
        print("• Add memory and conversation management")
        print("• Scale with Azure Cognitive Search")
        print("• Integrate with Azure AI Studio")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check Azure OpenAI endpoint configuration in .env file")
        print("2. Verify Azure authentication (az login)")
        print("3. Ensure correct deployment names for chat and embeddings")
        print("4. Install required packages: chromadb, langchain-openai")
        print("5. Check Azure OpenAI quota and regional availability")
