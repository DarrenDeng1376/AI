"""
Vector Stores and Embeddings - Comprehensive Examples

This module demonstrates modern vector store usage with LangChain:
1. Embedding generation with Azure OpenAI
2. Different vector store implementations
3. Similarity search and retrieval
4. RAG system integration
5. Performance optimization techniques
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import pickle

load_dotenv()

# Example 1: Basic Embedding Generation
def embedding_basics_example():
    """Demonstrate basic embedding generation with Azure OpenAI"""
    print("=== Example 1: Basic Embedding Generation ===")
    
    # Initialize Azure OpenAI embeddings using the configuration helper
    from azure_config import create_azure_openai_embeddings
    embeddings = create_azure_openai_embeddings()
    
    # Sample texts for embedding
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Vector databases store high-dimensional numerical representations",
        "Natural language processing helps computers understand human text"
    ]
    
    print("📝 Sample Texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n🔄 Generating embeddings...")
    
    # Generate embeddings
    text_embeddings = embeddings.embed_documents(sample_texts)
    query_embedding = embeddings.embed_query("What is machine learning?")
    
    print(f"✅ Generated embeddings:")
    print(f"- Document embeddings: {len(text_embeddings)} vectors of {len(text_embeddings[0])} dimensions")
    print(f"- Query embedding: {len(query_embedding)} dimensions")
    
    # Calculate similarity between query and documents
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    print(f"\n📊 Similarity to query 'What is machine learning?':")
    similarities = []
    for i, doc_embedding in enumerate(text_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity, sample_texts[i]))
        print(f"{i+1}. {similarity:.4f} - {sample_texts[i]}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\n🏆 Most similar document: {similarities[0][2]} (similarity: {similarities[0][1]:.4f})")
    
    print("\n" + "="*60 + "\n")

# Example 2: ChromaDB Vector Store
def chromadb_vector_store_example():
    """Demonstrate ChromaDB vector store usage"""
    print("=== Example 2: ChromaDB Vector Store ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Sample documents about AI and programming
    documents = [
        Document(
            page_content="Artificial Intelligence is the simulation of human intelligence in machines.",
            metadata={"source": "ai_basics.txt", "category": "AI", "date": "2024-01-15"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables computers to learn without explicit programming.",
            metadata={"source": "ml_intro.txt", "category": "AI", "date": "2024-01-16"}
        ),
        Document(
            page_content="Deep Learning uses neural networks with multiple layers to model complex patterns.",
            metadata={"source": "dl_guide.txt", "category": "AI", "date": "2024-01-17"}
        ),
        Document(
            page_content="Python is a versatile programming language widely used in data science and AI.",
            metadata={"source": "python_intro.txt", "category": "Programming", "date": "2024-01-18"}
        ),
        Document(
            page_content="Vector databases store and retrieve high-dimensional vectors efficiently.",
            metadata={"source": "vector_db.txt", "category": "Database", "date": "2024-01-19"}
        ),
        Document(
            page_content="Natural Language Processing helps computers understand and generate human language.",
            metadata={"source": "nlp_basics.txt", "category": "AI", "date": "2024-01-20"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines information retrieval with text generation.",
            metadata={"source": "rag_explained.txt", "category": "AI", "date": "2024-01-21"}
        ),
        Document(
            page_content="Embedding models convert text into numerical vectors for similarity comparison.",
            metadata={"source": "embeddings.txt", "category": "AI", "date": "2024-01-22"}
        )
    ]
    
    print(f"📚 Creating vector store with {len(documents)} documents...")
    
    # Create ChromaDB vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="ai_knowledge_base",
        persist_directory="./chroma_db"
    )
    
    print("✅ Vector store created successfully!")
    
    # Test similarity search
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What programming language is good for AI?",
        "How do vector databases work?"
    ]
    
    print(f"\n🔍 Testing similarity search:")
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=3)
        
        print("Top 3 results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.page_content}")
            print(f"     Source: {result.metadata.get('source', 'Unknown')}")
            print(f"     Category: {result.metadata.get('category', 'Unknown')}")
    
    # Test similarity search with scores
    print(f"\n📊 Similarity search with scores:")
    query = "machine learning algorithms"
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    print(f"Query: '{query}'")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")
    
    # Test metadata filtering
    print(f"\n🏷️  Filtered search (AI category only):")
    ai_results = vector_store.similarity_search(
        "learning algorithms",
        k=3,
        filter={"category": "AI"}
    )
    
    for i, result in enumerate(ai_results, 1):
        print(f"  {i}. {result.page_content}")
        print(f"     Category: {result.metadata['category']}")
    
    print("\n" + "="*60 + "\n")

# Example 3: FAISS Vector Store
def faiss_vector_store_example():
    """Demonstrate FAISS vector store for high-performance similarity search"""
    print("=== Example 3: FAISS Vector Store ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Create a larger dataset for performance testing
    print("📊 Creating larger dataset for performance testing...")
    
    # Generate synthetic documents about different topics
    topics_data = {
        "Machine Learning": [
            "Supervised learning uses labeled data to train predictive models",
            "Unsupervised learning finds patterns in data without labels",
            "Reinforcement learning trains agents through rewards and penalties",
            "Feature engineering improves model performance by selecting relevant variables",
            "Cross-validation helps prevent overfitting in machine learning models"
        ],
        "Deep Learning": [
            "Convolutional Neural Networks excel at image recognition tasks",
            "Recurrent Neural Networks process sequential data effectively",
            "Transformer architectures revolutionized natural language processing",
            "Gradient descent optimizes neural network parameters during training",
            "Backpropagation calculates gradients for neural network optimization"
        ],
        "Data Science": [
            "Data visualization helps communicate insights from complex datasets",
            "Statistical analysis provides foundation for data-driven decisions",
            "Data cleaning removes inconsistencies and errors from datasets",
            "Feature scaling normalizes data for better algorithm performance",
            "Hypothesis testing validates statistical assumptions and findings"
        ],
        "Programming": [
            "Object-oriented programming organizes code into reusable classes",
            "Functional programming emphasizes pure functions and immutability",
            "Version control systems track changes in code over time",
            "Unit testing ensures individual components work correctly",
            "Code documentation improves maintainability and collaboration"
        ]
    }
    
    # Create documents with metadata
    large_documents = []
    doc_id = 0
    
    for topic, contents in topics_data.items():
        for content in contents:
            doc_id += 1
            large_documents.append(Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "topic": topic,
                    "length": len(content),
                    "created": datetime.now().isoformat()
                }
            ))
    
    print(f"📚 Created {len(large_documents)} documents across {len(topics_data)} topics")
    
    # Create FAISS vector store
    print("🔄 Building FAISS index...")
    start_time = time.time()
    
    faiss_store = FAISS.from_documents(
        documents=large_documents,
        embedding=embeddings
    )
    
    build_time = time.time() - start_time
    print(f"✅ FAISS index built in {build_time:.2f} seconds")
    
    # Test different search scenarios
    test_scenarios = [
        {
            "name": "Machine Learning Query",
            "query": "How do supervised learning algorithms work?",
            "expected_topic": "Machine Learning"
        },
        {
            "name": "Deep Learning Query", 
            "query": "What are neural networks and how do they learn?",
            "expected_topic": "Deep Learning"
        },
        {
            "name": "Data Science Query",
            "query": "How to visualize and analyze large datasets?",
            "expected_topic": "Data Science"
        },
        {
            "name": "Programming Query",
            "query": "What are best practices for writing clean code?",
            "expected_topic": "Programming"
        }
    ]
    
    print(f"\n🔍 Testing search performance and accuracy:")
    
    total_search_time = 0
    
    for scenario in test_scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"Query: '{scenario['query']}'")
        
        # Measure search time
        start_time = time.time()
        results = faiss_store.similarity_search(scenario["query"], k=3)
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(f"⏱️  Search time: {search_time*1000:.2f}ms")
        print(f"🎯 Expected topic: {scenario['expected_topic']}")
        print("📄 Top results:")
        
        for i, result in enumerate(results, 1):
            topic = result.metadata.get('topic', 'Unknown')
            correct = "✅" if topic == scenario['expected_topic'] else "❌"
            print(f"  {i}. {correct} Topic: {topic}")
            print(f"     Content: {result.page_content}")
    
    avg_search_time = total_search_time / len(test_scenarios)
    print(f"\n📊 Performance Summary:")
    print(f"- Total documents: {len(large_documents)}")
    print(f"- Average search time: {avg_search_time*1000:.2f}ms")
    print(f"- Index build time: {build_time:.2f}s")
    
    # Save and load FAISS index
    print(f"\n💾 Testing save/load functionality:")
    faiss_store.save_local("faiss_index")
    print("✅ FAISS index saved to disk")
    
    # Load the index
    loaded_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    test_result = loaded_store.similarity_search("machine learning", k=1)
    print(f"✅ FAISS index loaded successfully, test search returned: {len(test_result)} results")
    
    print("\n" + "="*60 + "\n")

# Example 4: RAG Integration
def rag_integration_example():
    """Demonstrate RAG (Retrieval-Augmented Generation) integration"""
    print("=== Example 4: RAG Integration ===")
    
    # Initialize components using configuration helpers
    llm = create_azure_chat_openai(temperature=0.3)
    embeddings = create_azure_openai_embeddings()
    
    # Create knowledge base documents
    knowledge_base = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for connecting LLMs to external data sources and creating complex workflows.",
            metadata={"source": "langchain_docs", "type": "framework"}
        ),
        Document(
            page_content="Vector stores are databases optimized for storing and querying high-dimensional vectors. They enable efficient similarity search for RAG applications.",
            metadata={"source": "vector_guide", "type": "database"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) improves LLM responses by retrieving relevant information from external knowledge bases before generating answers.",
            metadata={"source": "rag_paper", "type": "technique"}
        ),
        Document(
            page_content="Azure OpenAI provides enterprise-grade AI services including GPT models, embedding models, and fine-tuning capabilities with security and compliance features.",
            metadata={"source": "azure_docs", "type": "service"}
        ),
        Document(
            page_content="Prompt engineering involves crafting effective prompts to guide LLM behavior and improve output quality through techniques like few-shot learning and chain-of-thought.",
            metadata={"source": "prompt_guide", "type": "technique"}
        ),
        Document(
            page_content="Text splitting breaks large documents into smaller chunks for better embedding and retrieval performance while maintaining semantic coherence.",
            metadata={"source": "preprocessing_guide", "type": "technique"}
        )
    ]
    
    print(f"📚 Creating RAG knowledge base with {len(knowledge_base)} documents...")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=knowledge_base,
        embedding=embeddings,
        collection_name="rag_knowledge_base"
    )
    
    # Create retrieval chain - LEGACY APPROACH
    retrieval_qa_legacy = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        verbose=True
    )
    
    # Modern LCEL Approach - RECOMMENDED
    print("🚀 Creating modern LCEL RAG chain...")
    
    # Define the RAG prompt template
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use the following context to answer the user's question accurately and comprehensively. 
        
Context:
{context}

If the answer cannot be found in the context, say "I don't have enough information in the provided context to answer that question."""),
        ("human", "{question}")
    ])
    
    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Modern LCEL chain
    modern_rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG systems initialized!")
    
    # Test RAG with different questions
    test_questions = [
        "What is LangChain and how does it work?",
        "How do vector stores help in AI applications?", 
        "What are the benefits of RAG for language models?",
        "What Azure AI services are available for developers?",
        "How can I improve my prompts for better LLM responses?"
    ]
    
    print(f"\n🤖 Testing RAG systems (Legacy vs Modern):")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"❓ Question {i}: {question}")
        print("="*60)
        
        # Legacy approach
        print(f"\n🔶 LEGACY RetrievalQA:")
        try:
            legacy_result = retrieval_qa_legacy.invoke({"query": question})
            print(f"Answer: {legacy_result['result']}")
            print(f"Sources: {len(legacy_result['source_documents'])} documents")
        except Exception as e:
            print(f"Error: {e}")
        
        # Modern LCEL approach
        print(f"\n🚀 MODERN LCEL Chain:")
        try:
            # Get retrieved documents for comparison
            retrieved_docs = retriever.get_relevant_documents(question)
            
            # Get modern response
            modern_result = modern_rag_chain.invoke(question)
            print(f"Answer: {modern_result}")
            print(f"Sources: {len(retrieved_docs)} documents retrieved")
            
            # Show sources used
            print(f"\n📖 Sources used:")
            for j, doc in enumerate(retrieved_docs, 1):
                print(f"  {j}. {doc.metadata.get('source', 'Unknown')} ({doc.metadata.get('type', 'Unknown')})")
                print(f"     Content: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"Error: {e}")
        
        if i < len(test_questions):  # Don't wait after last question
            print(f"\n⏸️  Press Enter to continue to next question...")
            # input()  # Uncomment to pause between questions
    
    # Demonstrate retrieval without generation
    print(f"\n🔍 Pure Retrieval Example:")
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    query = "vector databases"
    retrieved_docs = retriever.get_relevant_documents(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(retrieved_docs)} relevant documents:")
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")
    
    # Advanced Modern RAG Examples
    print(f"\n🎯 Advanced Modern RAG Patterns:")
    
    # 1. RAG with source citation
    print(f"\n📚 1. RAG with Source Citations:")
    
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use the following context to answer the user's question. 
        Always cite your sources using [Source: filename] format.
        
Context:
{context}

Provide a comprehensive answer and cite each source used."""),
        ("human", "{question}")
    ])
    
    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', f'Document_{i}')
            content = f"[Source: {source}]\n{doc.page_content}"
            formatted.append(content)
        return "\n\n".join(formatted)
    
    citation_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough()
        }
        | citation_prompt
        | llm
        | StrOutputParser()
    )
    
    citation_result = citation_chain.invoke("What is RAG and how does it work?")
    print(f"Answer with citations: {citation_result}")
    
    # 2. Multi-step RAG (with follow-up questions)
    print(f"\n🔄 2. Multi-step RAG Chain:")
    
    multistep_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Based on the context provided, answer the question comprehensively.
        If the question requires multiple aspects to be covered, structure your answer clearly.
        
Context:
{context}"""),
        ("human", "{question}")
    ])
    
    # Chain that can handle complex queries
    multistep_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | multistep_prompt
        | llm
        | StrOutputParser()
    )
    
    complex_question = "Compare LangChain and vector stores, and explain how they work together in RAG applications"
    multistep_result = multistep_chain.invoke(complex_question)
    print(f"Complex question: {complex_question}")
    print(f"Structured answer: {multistep_result}")
    
    # 3. RAG with conditional logic
    print(f"\n🧠 3. Conditional RAG (different responses based on context quality):")
    
    conditional_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use the following context to answer the user's question.

Context:
{context}

Instructions:
- If the context contains relevant information, provide a detailed answer
- If the context is somewhat related but not directly relevant, acknowledge this and provide what insight you can
- If the context is not relevant at all, clearly state that you don't have relevant information

Always be honest about the limitations of your knowledge based on the provided context."""),
        ("human", "{question}")
    ])
    
    conditional_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | conditional_prompt
        | llm
        | StrOutputParser()
    )
    
    # Test with a question that might not have good context
    edge_case_question = "What is the weather like today?"
    conditional_result = conditional_chain.invoke(edge_case_question)
    print(f"Edge case question: {edge_case_question}")
    print(f"Conditional response: {conditional_result}")
    
    print(f"\n💡 Key Advantages of Modern LCEL Approach:")
    print(f"✅ More flexible and customizable")
    print(f"✅ Better error handling and debugging")
    print(f"✅ Easier to modify prompts and logic")
    print(f"✅ Supports streaming and async operations")
    print(f"✅ Better integration with LangSmith tracing")
    print(f"✅ More explicit data flow")
    print(f"✅ Easier to add custom processing steps")
    
    print("\n" + "="*60 + "\n")

# Example 5: Advanced Vector Operations
def advanced_vector_operations_example():
    """Demonstrate advanced vector operations and optimization techniques"""
    print("=== Example 5: Advanced Vector Operations ===")
    
    # Initialize embeddings using the configuration helper
    embeddings = create_azure_openai_embeddings()
    
    # Create diverse document set
    diverse_documents = [
        "Climate change is causing global temperature increases and extreme weather patterns.",
        "Quantum computing uses quantum mechanical phenomena to process information in fundamentally new ways.",
        "Blockchain technology provides decentralized and secure digital transaction recording.",
        "Renewable energy sources like solar and wind are becoming more cost-effective than fossil fuels.",
        "Artificial intelligence is transforming industries from healthcare to transportation.",
        "Space exploration continues with missions to Mars and the development of space tourism.",
        "Biotechnology advances are enabling personalized medicine and gene therapy treatments.",
        "Cybersecurity threats are evolving rapidly as digital infrastructure expands globally.",
        "Sustainable agriculture practices help reduce environmental impact while maintaining food security.",
        "Virtual reality and augmented reality are creating new forms of digital interaction and entertainment."
    ]
    
    # Create documents with enhanced metadata
    enhanced_docs = []
    categories = ["Environment", "Technology", "Finance", "Environment", "Technology", 
                 "Science", "Healthcare", "Security", "Agriculture", "Entertainment"]
    
    for i, (content, category) in enumerate(zip(diverse_documents, categories)):
        enhanced_docs.append(Document(
            page_content=content,
            metadata={
                "id": i,
                "category": category,
                "word_count": len(content.split()),
                "char_count": len(content),
                "keywords": content.lower().split()[:3]  # First 3 words as simple keywords
            }
        ))
    
    print(f"📊 Creating enhanced vector store with {len(enhanced_docs)} documents...")
    
    # Create vector store
    vector_store = FAISS.from_documents(enhanced_docs, embeddings)
    
    print("✅ Vector store created with enhanced metadata!")
    
    # 1. Multi-modal search strategies
    print(f"\n🔍 Testing different search strategies:")
    
    query = "environmental sustainability"
    
    # Standard similarity search
    standard_results = vector_store.similarity_search(query, k=3)
    print(f"\n📋 Standard similarity search for '{query}':")
    for i, doc in enumerate(standard_results, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Category: {doc.metadata['category']}")
    
    # Similarity search with score threshold
    scored_results = vector_store.similarity_search_with_score(query, k=5)
    print(f"\n📊 Similarity search with scores:")
    for i, (doc, score) in enumerate(scored_results, 1):
        print(f"  {i}. Score: {score:.4f}, Category: {doc.metadata['category']}")
        print(f"     Content: {doc.page_content[:80]}...")
    
    # 2. Batch operations
    print(f"\n⚡ Testing batch operations:")
    
    batch_queries = [
        "artificial intelligence applications",
        "renewable energy solutions", 
        "space exploration missions"
    ]
    
    print(f"Processing {len(batch_queries)} queries in batch...")
    start_time = time.time()
    
    batch_results = {}
    for query in batch_queries:
        batch_results[query] = vector_store.similarity_search(query, k=2)
    
    batch_time = time.time() - start_time
    print(f"✅ Batch processing completed in {batch_time:.3f} seconds")
    
    for query, results in batch_results.items():
        print(f"\n  Query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"    {i}. {doc.metadata['category']}: {doc.page_content[:60]}...")
    
    # 3. Vector clustering analysis
    print(f"\n🎯 Vector clustering analysis:")
    
    # Get all embeddings
    all_texts = [doc.page_content for doc in enhanced_docs]
    all_embeddings = embeddings.embed_documents(all_texts)
    
    # Simple clustering by category similarity
    category_centroids = {}
    for doc, embedding in zip(enhanced_docs, all_embeddings):
        category = doc.metadata['category']
        if category not in category_centroids:
            category_centroids[category] = []
        category_centroids[category].append(embedding)
    
    # Calculate average embeddings per category
    category_averages = {}
    for category, embeddings_list in category_centroids.items():
        category_averages[category] = np.mean(embeddings_list, axis=0)
    
    print(f"📊 Category analysis:")
    print(f"- Found {len(category_averages)} unique categories")
    for category, avg_embedding in category_averages.items():
        doc_count = len(category_centroids[category])
        print(f"  • {category}: {doc_count} documents")
    
    # Find most similar categories
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    print(f"\n🔗 Category similarity matrix:")
    categories_list = list(category_averages.keys())
    
    for i, cat1 in enumerate(categories_list):
        for j, cat2 in enumerate(categories_list):
            if i < j:  # Only show upper triangle
                similarity = cosine_similarity(
                    category_averages[cat1], 
                    category_averages[cat2]
                )
                print(f"  {cat1} ↔ {cat2}: {similarity:.4f}")
    
    # 4. Performance optimization demo
    print(f"\n⚡ Performance optimization techniques:")
    
    # Test different k values
    k_values = [1, 3, 5, 10]
    query = "technology innovation"
    
    print(f"Testing search performance with different k values:")
    for k in k_values:
        if k <= len(enhanced_docs):
            start_time = time.time()
            results = vector_store.similarity_search(query, k=k)
            search_time = time.time() - start_time
            print(f"  k={k}: {search_time*1000:.2f}ms, returned {len(results)} results")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Vector Stores and Embeddings - Comprehensive Examples")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_embeddings = create_azure_openai_embeddings()
        test_embedding = test_embeddings.embed_query("Hello world")
        print(f"✅ Connection successful: Generated {len(test_embedding)}-dimensional embedding")
        print()
        
        # Run all examples
        # embedding_basics_example()
        # chromadb_vector_store_example()
        # faiss_vector_store_example()
        rag_integration_example()
        advanced_vector_operations_example()
        
        print("🎉 All vector store examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI endpoint in .env file")
        print("2. Authenticated with Azure CLI (az login) or using managed identity")
        print("3. Valid Azure OpenAI embeddings deployment")
        print("4. Installed required packages: chromadb, faiss-cpu")
