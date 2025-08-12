"""
RAG (Retrieval-Augmented Generation) Examples

This module demonstrates how to build RAG systems with LangChain:
1. Basic RAG implementation
2. Document processing and chunking
3. Vector store operations
4. Advanced retrieval techniques
5. RAG system evaluation
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Example 1: Basic RAG System
def basic_rag_example():
    """
    Demonstrates a simple RAG system with local documents
    """
    print("=== Example 1: Basic RAG System ===")
    
    from langchain_openai import OpenAI, OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    
    # Sample documents (in real scenario, load from files)
    documents = [
        Document(page_content="""
        LangChain is a framework for developing applications powered by language models. 
        It enables applications that are context-aware and can reason about information. 
        LangChain provides tools for prompt management, chains, memory, and agents.
        """, metadata={"source": "langchain_intro.txt"}),
        
        Document(page_content="""
        Vector databases store high-dimensional vectors representing data embeddings. 
        They enable fast similarity search and are essential for RAG systems. 
        Popular vector databases include Chroma, Pinecone, and Weaviate.
        """, metadata={"source": "vector_db.txt"}),
        
        Document(page_content="""
        Embeddings are numerical representations of text that capture semantic meaning. 
        Similar texts have similar embeddings. OpenAI's text-embedding-ada-002 is 
        commonly used for creating embeddings.
        """, metadata={"source": "embeddings.txt"})
    ]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Create LLM and retrieval chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Test queries
    queries = [
        "What is LangChain?",
        "How do vector databases work?",
        "What are embeddings used for?"
    ]
    
    for query in queries:
        result = qa_chain({"query": query})
        print(f"Question: {query}")
        print(f"Answer: {result['result']}")
        print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")
        print("-" * 60)


# Example 2: Advanced Document Processing
def document_processing_example():
    """
    Shows advanced document processing techniques
    """
    print("=== Example 2: Advanced Document Processing ===")
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
    from langchain.schema import Document
    import tiktoken
    
    # Sample long document
    long_text = """
    Artificial Intelligence (AI) has transformed numerous industries and continues to evolve rapidly. 
    Machine Learning, a subset of AI, enables computers to learn and improve from experience without 
    being explicitly programmed. Deep Learning, a subset of Machine Learning, uses neural networks 
    with multiple layers to model and understand complex patterns in data.
    
    Natural Language Processing (NLP) is another crucial area of AI that focuses on the interaction 
    between computers and human language. It involves tasks like text analysis, language translation, 
    sentiment analysis, and question-answering systems.
    
    Computer Vision enables machines to interpret and understand visual information from the world. 
    This technology is used in applications like autonomous vehicles, medical imaging, and facial 
    recognition systems.
    
    The future of AI holds great promise with emerging technologies like Generative AI, which can 
    create new content, and Quantum Computing, which could exponentially increase computational power 
    for AI applications.
    """ * 3  # Repeat to make it longer
    
    document = Document(page_content=long_text)
    
    # Different splitting strategies
    splitters = {
        "Recursive Character": RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        ),
        "Token-based": TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
    }
    
    for name, splitter in splitters.items():
        chunks = splitter.split_documents([document])
        print(f"{name} Splitter:")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  First chunk (first 100 chars): {chunks[0].page_content[:100]}...")
        print(f"  Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) // len(chunks)} chars")
        print()


# Example 3: Multiple Vector Store Types
def vector_stores_example():
    """
    Compares different vector store implementations
    """
    print("=== Example 3: Vector Store Comparison ===")
    
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma, FAISS
    from langchain.schema import Document
    
    # Sample documents
    docs = [
        Document(page_content="Python is a versatile programming language."),
        Document(page_content="Machine learning algorithms learn from data."),
        Document(page_content="Natural language processing handles text data."),
        Document(page_content="Vector databases enable semantic search.")
    ]
    
    embeddings = OpenAIEmbeddings()
    
    # Create different vector stores
    vector_stores = {
        "Chroma": Chroma.from_documents(docs, embeddings),
        "FAISS": FAISS.from_documents(docs, embeddings)
    }
    
    query = "programming languages"
    
    for name, vs in vector_stores.items():
        results = vs.similarity_search(query, k=2)
        print(f"{name} Vector Store Results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.page_content}")
        print()


# Example 4: Advanced RAG with Re-ranking
def advanced_rag_example():
    """
    Implements advanced RAG with query expansion and re-ranking
    """
    print("=== Example 4: Advanced RAG with Re-ranking ===")
    
    from langchain_openai import OpenAI, OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.chains import RetrievalQA
    
    # Create more detailed documents
    documents = [
        Document(page_content="""
        Python is a high-level, interpreted programming language known for its simple syntax 
        and readability. It supports multiple programming paradigms including procedural, 
        object-oriented, and functional programming. Python is widely used in web development, 
        data analysis, artificial intelligence, and scientific computing.
        """),
        Document(page_content="""
        Machine learning is a subset of artificial intelligence that focuses on building 
        systems that can learn and improve from data without being explicitly programmed. 
        Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn. 
        Common algorithms include linear regression, decision trees, and neural networks.
        """),
        Document(page_content="""
        Data science combines statistics, computer science, and domain expertise to extract 
        insights from data. Data scientists use tools like pandas, numpy, and matplotlib 
        for data manipulation and visualization. The data science process includes data 
        collection, cleaning, analysis, and interpretation of results.
        """)
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Add contextual compression (re-ranking)
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Create QA chain with advanced retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True
    )
    
    # Test query
    query = "What programming tools are used in data analysis?"
    result = qa_chain({"query": query})
    
    print(f"Query: {query}")
    print(f"Answer: {result['result']}")
    print(f"Number of source documents: {len(result['source_documents'])}")
    print()


# Example 5: RAG with Custom Prompt
def custom_prompt_rag_example():
    """
    Shows how to customize the RAG prompt for specific use cases
    """
    print("=== Example 5: RAG with Custom Prompt ===")
    
    from langchain_openai import OpenAI, OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    
    # Create documents about a fictional company
    documents = [
        Document(page_content="""
        TechCorp Inc. was founded in 2020 and specializes in AI-powered software solutions. 
        The company has 150 employees and offices in New York, San Francisco, and London. 
        TechCorp's mission is to democratize AI technology for small and medium businesses.
        """),
        Document(page_content="""
        TechCorp offers three main products: DataMind (data analytics platform), 
        ChatBot Pro (customer service automation), and PredictFlow (predictive analytics tool). 
        All products use cutting-edge machine learning algorithms and are available as 
        cloud-based SaaS solutions.
        """),
        Document(page_content="""
        TechCorp's customer support operates 24/7 with response time guaranteed within 2 hours. 
        The company offers tiered pricing: Starter ($99/month), Professional ($299/month), 
        and Enterprise (custom pricing). All plans include free onboarding and training.
        """
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Custom prompt template
    custom_prompt = PromptTemplate(
        template="""You are a helpful customer service representative for TechCorp Inc. 
        Use the following context to answer customer questions accurately and professionally. 
        If you don't know the answer based on the context, politely say so and offer to connect 
        them with a specialist.

        Context: {context}

        Customer Question: {question}

        Response:""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain with custom prompt
    llm = OpenAI(temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )
    
    # Customer queries
    customer_questions = [
        "What products does TechCorp offer?",
        "How much does the Professional plan cost?",
        "Do you have offices outside the US?",
        "What's your refund policy?"  # Not in context
    ]
    
    for question in customer_questions:
        response = qa_chain.run(question)
        print(f"Customer: {question}")
        print(f"TechCorp Support: {response}")
        print("-" * 50)


if __name__ == "__main__":
    print("RAG (Retrieval-Augmented Generation) Examples")
    print("=" * 60)
    
    try:
        basic_rag_example()
        document_processing_example()
        vector_stores_example()
        # advanced_rag_example()  # Uncomment if you have the packages
        custom_prompt_rag_example()
        
        print("🎉 RAG examples completed!")
        print("Next: Explore advanced RAG techniques and production deployment")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all required packages and set API keys")
