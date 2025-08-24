"""
Document Processing and Text Splitting for Vector Stores

This module demonstrates how to properly prepare documents for vector storage:
1. Text splitting strategies for different document types
2. Metadata enhancement and extraction
3. Document preprocessing and cleaning
4. Chunking optimization for retrieval
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from typing import List, Dict, Any
import re
import json
from datetime import datetime

load_dotenv()

def text_splitting_strategies_example():
    """Demonstrate different text splitting strategies"""
    print("=== Text Splitting Strategies ===")
    
    # Sample long document
    long_document = """
# Machine Learning Overview

Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning
Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. The goal is to approximate the mapping function so well that when you have new input data (X) that you can predict the output variables (Y) for that data.

Examples of supervised learning include:
- Classification problems: Predicting categories (spam/not spam, disease/no disease)
- Regression problems: Predicting continuous values (house prices, stock prices)

### Unsupervised Learning
Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.

Examples include:
- Clustering: Finding groups in data (customer segmentation)
- Association: Finding rules that describe relationships (market basket analysis)

### Reinforcement Learning
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment to maximize some notion of cumulative reward. The agent learns through trial and error by receiving rewards or penalties for actions.

## Applications

Machine learning has numerous applications across industries:
- Healthcare: Medical diagnosis, drug discovery, personalized treatment
- Finance: Fraud detection, algorithmic trading, credit scoring
- Technology: Recommendation systems, computer vision, natural language processing
- Transportation: Autonomous vehicles, route optimization, predictive maintenance

## Conclusion

Machine learning continues to evolve and transform how we solve complex problems across various domains. Understanding its different approaches and applications is crucial for leveraging its potential effectively.
"""
    
    print(f"📄 Original document length: {len(long_document)} characters")
    print(f"Word count: {len(long_document.split())} words")
    
    # 1. Recursive Character Text Splitter (Recommended for most cases)
    print(f"\n🔧 1. Recursive Character Text Splitter:")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    recursive_chunks = recursive_splitter.split_text(long_document)
    print(f"   Created {len(recursive_chunks)} chunks")
    for i, chunk in enumerate(recursive_chunks[:3], 1):
        print(f"   Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
    
    # 2. Character Text Splitter (Simple splitting)
    print(f"\n🔧 2. Character Text Splitter:")
    char_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n\n"
    )
    
    char_chunks = char_splitter.split_text(long_document)
    print(f"   Created {len(char_chunks)} chunks")
    for i, chunk in enumerate(char_chunks[:2], 1):
        print(f"   Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
    
    # 3. Token Text Splitter (Based on tokens, not characters)
    print(f"\n🔧 3. Token Text Splitter:")
    token_splitter = TokenTextSplitter(
        chunk_size=100,  # Number of tokens
        chunk_overlap=10
    )
    
    token_chunks = token_splitter.split_text(long_document)
    print(f"   Created {len(token_chunks)} chunks")
    for i, chunk in enumerate(token_chunks[:2], 1):
        print(f"   Chunk {i}: {chunk[:100]}...")
    
    # 4. Markdown Header Text Splitter (Structure-aware)
    print(f"\n🔧 4. Markdown Header Text Splitter:")
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    
    markdown_chunks = markdown_splitter.split_text(long_document)
    print(f"   Created {len(markdown_chunks)} chunks with headers")
    for i, chunk in enumerate(markdown_chunks[:3], 1):
        print(f"   Chunk {i}: {chunk.page_content[:100]}...")
        print(f"   Metadata: {chunk.metadata}")
    
    print("\n" + "="*60 + "\n")

def document_preprocessing_example():
    """Demonstrate document preprocessing and cleaning"""
    print("=== Document Preprocessing and Cleaning ===")
    
    # Sample documents with various formatting issues
    raw_documents = [
        """
        
        
        Title:   Python Programming Guide    
        
        
        Python is a high-level programming language.          It's known for its simplicity and readability.
        
        
        Key features include:
        - Easy to learn syntax
        - Large standard library    
        - Cross-platform compatibility
        
        
        """,
        """
        ARTIFICIAL INTELLIGENCE - AN OVERVIEW
        
        
        AI refers to the simulation of human intelligence in machines...
        
        
        Types:
        1) Machine Learning  
        2) Deep Learning
        3) Natural Language Processing
        
        
        
        For more info, visit: https://example.com/ai-guide
        Email: info@example.com
        """,
        """
        ### Data Science Workflow
        
        Step 1: Data Collection
        Step 2: Data Cleaning  
        Step 3: Exploratory Data Analysis (EDA)
        Step 4: Model Building
        Step 5: Model Evaluation
        
        Note: This process is iterative and may require multiple cycles.
        """
    ]
    
    def clean_document(text: str) -> str:
        """Clean and normalize document text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove email addresses (for privacy)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove URLs (or replace with placeholder)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        return text
    
    def extract_metadata(text: str, doc_id: int) -> Dict[str, Any]:
        """Extract metadata from document text"""
        metadata = {
            "doc_id": doc_id,
            "char_count": len(text),
            "word_count": len(text.split()),
            "has_urls": bool(re.search(r'http[s]?://', text)),
            "has_emails": bool(re.search(r'\S+@\S+', text)),
            "has_code": bool(re.search(r'```|`.*`', text)),
            "has_lists": bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            "has_numbers": bool(re.search(r'\d+[.]\s', text)),
            "processed_at": datetime.now().isoformat()
        }
        
        # Extract title (first line or header)
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line:
                metadata["title"] = first_line[:100]  # Limit title length
        
        # Estimate reading time (average 200 words per minute)
        metadata["estimated_reading_time"] = max(1, metadata["word_count"] // 200)
        
        return metadata
    
    print(f"📄 Processing {len(raw_documents)} raw documents...")
    
    # Process documents
    processed_documents = []
    
    for i, raw_text in enumerate(raw_documents, 1):
        print(f"\n📋 Document {i}:")
        print(f"   Original length: {len(raw_text)} characters")
        
        # Clean the document
        cleaned_text = clean_document(raw_text)
        print(f"   Cleaned length: {len(cleaned_text)} characters")
        
        # Extract metadata
        metadata = extract_metadata(cleaned_text, i)
        print(f"   Metadata: {json.dumps({k: v for k, v in metadata.items() if k not in ['processed_at']}, indent=6)}")
        
        # Create document object
        doc = Document(
            page_content=cleaned_text,
            metadata=metadata
        )
        processed_documents.append(doc)
        
        print(f"   Processed content preview: {cleaned_text[:150]}...")
    
    # Split processed documents into chunks
    print(f"\n🔧 Splitting processed documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    
    all_chunks = []
    for doc in processed_documents:
        chunks = text_splitter.split_documents([doc])
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{doc.metadata['doc_id']}_chunk_{i+1}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_doc_id": doc.metadata['doc_id']
            })
        
        all_chunks.extend(chunks)
    
    print(f"✅ Created {len(all_chunks)} chunks from {len(processed_documents)} documents")
    
    # Show chunk details
    print(f"\n📊 Chunk Analysis:")
    chunk_sizes = [len(chunk.page_content) for chunk in all_chunks]
    print(f"   Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters")
    print(f"   Min chunk size: {min(chunk_sizes)} characters")
    print(f"   Max chunk size: {max(chunk_sizes)} characters")
    
    # Show sample chunks with metadata
    print(f"\n📄 Sample chunks with enhanced metadata:")
    for i, chunk in enumerate(all_chunks[:3], 1):
        print(f"\n   Chunk {i}:")
        print(f"   Content: {chunk.page_content[:100]}...")
        print(f"   Metadata: {json.dumps(chunk.metadata, indent=8)}")
    
    print("\n" + "="*60 + "\n")

def chunking_optimization_example():
    """Demonstrate chunking optimization for better retrieval"""
    print("=== Chunking Optimization for Retrieval ===")
    
    # Initialize embeddings using the Azure configuration helper
    from azure_config import create_azure_openai_embeddings
    embeddings = create_azure_openai_embeddings()
    
    # Sample technical document
    technical_document = """
    Deep Learning Neural Networks: A Comprehensive Guide

    Introduction
    Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain.

    Architecture Components
    
    Neurons and Layers
    Artificial neurons are the basic units of neural networks. Each neuron receives inputs, applies a transformation function, and produces an output. Layers are collections of neurons that process information at the same level of abstraction.
    
    Input Layer
    The input layer receives raw data and passes it to the hidden layers. The number of neurons in this layer corresponds to the number of features in the input data.
    
    Hidden Layers
    Hidden layers perform complex transformations on the input data. Deep networks have multiple hidden layers, allowing them to learn hierarchical representations of data.
    
    Output Layer
    The output layer produces the final predictions or classifications. The number of neurons depends on the specific task (binary classification, multi-class classification, regression).
    
    Activation Functions
    Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.
    
    ReLU (Rectified Linear Unit)
    ReLU is the most commonly used activation function. It outputs the input directly if positive, otherwise zero. It helps mitigate the vanishing gradient problem.
    
    Sigmoid
    Sigmoid functions output values between 0 and 1, making them useful for binary classification tasks. However, they can suffer from vanishing gradients.
    
    Tanh
    Tanh functions output values between -1 and 1, often providing better convergence than sigmoid functions.
    
    Training Process
    
    Forward Propagation
    Data flows from input to output layers, with each layer applying transformations and activation functions.
    
    Loss Calculation
    The network's predictions are compared to actual targets using a loss function (e.g., mean squared error, cross-entropy).
    
    Backpropagation
    Gradients are calculated and propagated backward through the network to update weights and minimize loss.
    
    Optimization
    Optimization algorithms like Adam, SGD, and RMSprop update network parameters to improve performance.
    
    Applications
    Deep learning has revolutionized numerous fields including computer vision, natural language processing, speech recognition, and autonomous systems.
    """
    
    # Test different chunking strategies
    chunking_strategies = [
        {
            "name": "Small Chunks",
            "chunk_size": 200,
            "chunk_overlap": 20
        },
        {
            "name": "Medium Chunks",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        {
            "name": "Large Chunks",
            "chunk_size": 1000,
            "chunk_overlap": 100
        }
    ]
    
    test_queries = [
        "What are activation functions in neural networks?",
        "How does backpropagation work in deep learning?",
        "What are the components of neural network architecture?"
    ]
    
    print(f"📊 Testing chunking strategies for retrieval quality...")
    
    strategy_results = {}
    
    for strategy in chunking_strategies:
        print(f"\n🔧 Testing {strategy['name']} (size: {strategy['chunk_size']}, overlap: {strategy['chunk_overlap']}):")
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy['chunk_size'],
            chunk_overlap=strategy['chunk_overlap'],
            length_function=len
        )
        
        # Split document
        chunks = splitter.split_text(technical_document)
        
        # Create documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "strategy": strategy['name'],
                    "chunk_size": len(chunk),
                    "source": "deep_learning_guide"
                }
            ))
        
        print(f"   Created {len(chunk_docs)} chunks")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            chunk_docs,
            embeddings,
            collection_name=f"chunks_{strategy['name'].lower().replace(' ', '_')}"
        )
        
        # Test retrieval quality
        strategy_scores = []
        
        for query in test_queries:
            results = vector_store.similarity_search_with_score(query, k=3)
            
            # Calculate average relevance score
            avg_score = sum(score for _, score in results) / len(results)
            strategy_scores.append(avg_score)
            
            print(f"     Query: '{query[:30]}...': Avg score: {avg_score:.4f}")
        
        overall_score = sum(strategy_scores) / len(strategy_scores)
        strategy_results[strategy['name']] = {
            "num_chunks": len(chunk_docs),
            "avg_chunk_size": sum(len(doc.page_content) for doc in chunk_docs) / len(chunk_docs),
            "overall_score": overall_score,
            "individual_scores": strategy_scores
        }
    
    # Compare strategies
    print(f"\n📈 Strategy Comparison:")
    print(f"{'Strategy':<15} {'Chunks':<8} {'Avg Size':<10} {'Score':<8} {'Quality'}")
    print("-" * 55)
    
    for strategy_name, results in strategy_results.items():
        quality = "🥇" if results['overall_score'] == min(r['overall_score'] for r in strategy_results.values()) else "🥈" if results['overall_score'] == sorted([r['overall_score'] for r in strategy_results.values()])[1] else "🥉"
        print(f"{strategy_name:<15} {results['num_chunks']:<8} {results['avg_chunk_size']:<10.0f} {results['overall_score']:<8.4f} {quality}")
    
    # Recommend optimal strategy
    best_strategy = min(strategy_results.items(), key=lambda x: x[1]['overall_score'])
    print(f"\n🏆 Recommended Strategy: {best_strategy[0]}")
    print(f"   Lowest average distance score: {best_strategy[1]['overall_score']:.4f}")
    print(f"   Balance of chunk count and retrieval quality")
    
    # Show optimal chunk example
    print(f"\n📄 Example optimal chunks:")
    optimal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Based on medium chunks typically performing well
        chunk_overlap=50,
        length_function=len
    )
    
    optimal_chunks = optimal_splitter.split_text(technical_document)
    
    for i, chunk in enumerate(optimal_chunks[:2], 1):
        print(f"\n   Optimal Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk[:200]}...")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Document Processing and Text Splitting for Vector Stores")
    print("=" * 60)
    
    try:
        # Run examples
        
        text_splitting_strategies_example()
        document_preprocessing_example()
        chunking_optimization_example()
        
        print("🎉 Document processing examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings")
        print("2. Installed required packages: chromadb")
