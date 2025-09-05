"""
Example script demonstrating the Intelligent Document Processor
This script shows how to use the document processing pipeline programmatically
"""
import asyncio
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from config import validate_configuration
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.qa_engine import QAEngine
from src.utils.azure_clients import validate_azure_configuration

async def main():
    """Main example function"""
    print("🤖 Intelligent Document Processor - Example Script")
    print("=" * 60)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    
    is_valid, issues = validate_configuration()
    azure_issues = validate_azure_configuration()
    
    if not is_valid or azure_issues:
        print("❌ Configuration issues found:")
        for issue in issues + azure_issues:
            print(f"   - {issue}")
        print("\nPlease check your .env file and Azure credentials.")
        return
    
    print("✅ Configuration is valid!")
    
    # Step 2: Initialize components
    print("\n2. Initializing components...")
    
    try:
        document_processor = DocumentProcessor()
        embedding_manager = EmbeddingManager("example_collection")
        qa_engine = QAEngine(embedding_manager)
        print("✅ All components initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Step 3: Example with sample text (since we don't have actual documents)
    print("\n3. Processing sample document...")
    
    # Create a sample document for demonstration
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a rapidly evolving field that aims to create intelligent machines 
    capable of performing tasks that typically require human intelligence. Machine Learning (ML) is a 
    subset of AI that focuses on algorithms that can learn and make decisions from data.
    
    Key Applications:
    - Natural Language Processing (NLP)
    - Computer Vision
    - Robotics
    - Autonomous Vehicles
    
    Current Trends:
    Large Language Models (LLMs) like GPT-4 have revolutionized natural language understanding and 
    generation. These models can perform complex reasoning, creative writing, and code generation.
    
    Future Outlook:
    The future of AI includes advancements in:
    1. Artificial General Intelligence (AGI)
    2. Quantum Machine Learning
    3. Neuromorphic Computing
    4. AI Ethics and Safety
    """
    
    # Create a sample document content object
    from src.document_processor import DocumentContent
    
    sample_document = DocumentContent(
        text=sample_text,
        tables=[],
        key_value_pairs=[],
        paragraphs=sample_text.split('\n\n'),
        pages=[{"page_number": 1}],
        confidence_scores={"overall": 0.95},
        metadata={
            "filename": "ai_overview.txt",
            "total_pages": 1,
            "has_tables": False,
            "character_count": len(sample_text),
            "processing_model": "sample"
        }
    )
    
    print("✅ Sample document created!")
    
    # Step 4: Create embeddings
    print("\n4. Creating embeddings...")
    
    document_id = "sample_doc_1"
    document_name = "AI Overview Sample"
    
    embedding_result = await embedding_manager.create_embeddings(
        sample_document, 
        document_id, 
        document_name
    )
    
    if embedding_result.success:
        print(f"✅ Created {embedding_result.embedding_count} embeddings in {embedding_result.processing_time:.2f}s")
    else:
        print(f"❌ Failed to create embeddings: {embedding_result.error_message}")
        return
    
    # Step 5: Test search functionality
    print("\n5. Testing search functionality...")
    
    search_queries = [
        "What is artificial intelligence?",
        "machine learning applications",
        "future of AI"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        
        search_results = await embedding_manager.search_similar_content(query, max_results=3)
        
        if search_results:
            print(f"   Found {len(search_results)} relevant chunks:")
            for i, result in enumerate(search_results):
                print(f"   {i+1}. Similarity: {result.similarity_score:.3f}")
                print(f"      Content: {result.content[:100]}...")
        else:
            print("   No relevant content found")
    
    # Step 6: Test Q&A functionality
    print("\n6. Testing Q&A functionality...")
    
    qa_questions = [
        "What is artificial intelligence?",
        "What are the main applications of AI?",
        "What does the future hold for AI?"
    ]
    
    for question in qa_questions:
        print(f"\n❓ Question: {question}")
        
        answer = await qa_engine.answer_question(question)
        
        print(f"💡 Answer: {answer.answer}")
        print(f"🎯 Confidence: {answer.confidence.value} ({answer.confidence_score:.1%})")
        print(f"⏱️ Processing time: {answer.processing_time:.2f}s")
        
        if answer.sources:
            print(f"📚 Sources: {len(answer.sources)} chunks used")
        
        if answer.follow_up_questions:
            print("💭 Follow-up questions:")
            for fq in answer.follow_up_questions:
                print(f"   - {fq}")
    
    # Step 7: Show statistics
    print("\n7. Document statistics...")
    
    stats = embedding_manager.get_document_stats()
    print(f"📊 Collection Statistics:")
    print(f"   - Total documents: {stats.get('total_documents', 0)}")
    print(f"   - Total chunks: {stats.get('total_chunks', 0)}")
    print(f"   - Total characters: {stats.get('total_characters', 0):,}")
    print(f"   - Total words: {stats.get('total_words', 0):,}")
    
    # Step 8: Cleanup
    print("\n8. Cleaning up...")
    
    # Clear the test collection
    embedding_manager.clear_all_documents()
    print("✅ Test collection cleared!")
    
    print("\n🎉 Example completed successfully!")
    print("\nTo use this with real documents:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Upload your documents through the web interface")
    print("3. Ask questions about your documents")

def run_health_check():
    """Run a quick health check of the system"""
    print("🔧 Running system health check...")
    
    # Check configuration
    is_valid, issues = validate_configuration()
    azure_issues = validate_azure_configuration()
    
    print(f"Configuration valid: {'✅' if is_valid else '❌'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print(f"Azure configuration valid: {'✅' if not azure_issues else '❌'}")
    if azure_issues:
        for issue in azure_issues:
            print(f"  - {issue}")
    
    # Test imports
    try:
        from src.document_processor import DocumentProcessor
        from src.embedding_manager import EmbeddingManager
        from src.qa_engine import QAEngine
        print("Module imports: ✅")
    except Exception as e:
        print(f"Module imports: ❌ ({e})")
    
    return is_valid and not azure_issues

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run full example")
    print("2. Run health check only")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        if run_health_check():
            print("\n" + "="*60)
            asyncio.run(main())
        else:
            print("❌ Health check failed. Please fix configuration issues first.")
    elif choice == "2":
        run_health_check()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
