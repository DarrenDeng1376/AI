"""
Test script to understand how RecursiveCharacterTextSplitter overlap actually works
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def test_overlap_behavior():
    # Simple test text
    test_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five. This is sentence six."
    
    print("Original text:")
    print(f"'{test_text}'")
    print(f"Length: {len(test_text)} characters")
    print()
    
    # Test with different overlap settings
    splitters = [
        ("No Overlap", RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)),
        ("10 char Overlap", RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)),
        ("20 char Overlap", RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)),
    ]
    
    for name, splitter in splitters:
        print(f"=== {name} ===")
        print(f"Chunk size: {splitter._chunk_size}, Overlap: {splitter._chunk_overlap}")
        
        # Create document and split
        doc = Document(page_content=test_text)
        chunks = splitter.split_documents([doc])
        
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: '{chunk.page_content}' (len: {len(chunk.page_content)})")
        
        # Check actual overlap between consecutive chunks
        if len(chunks) >= 2:
            chunk1 = chunks[0].page_content
            chunk2 = chunks[1].page_content
            
            # Find longest common suffix/prefix
            max_overlap = min(len(chunk1), len(chunk2))
            actual_overlap = 0
            
            for j in range(1, max_overlap + 1):
                if chunk1[-j:] == chunk2[:j]:
                    actual_overlap = j
            
            print(f"Actual overlap between chunk 1 and 2: {actual_overlap} characters")
            if actual_overlap > 0:
                print(f"Overlapping text: '{chunk1[-actual_overlap:]}'")
            else:
                print("No actual overlap found!")
        
        print()

def test_with_longer_text():
    # Test with paragraph-like text (NO paragraph breaks to force overlap)
    long_text = "Artificial Intelligence (AI) has transformed numerous industries and continues to evolve rapidly. Microsoft Azure provides comprehensive AI services through Azure Cognitive Services and Azure OpenAI. Machine Learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. Azure Machine Learning provides a cloud-based environment for training, deploying, and managing machine learning models at scale. Deep Learning uses neural networks with multiple layers to model complex patterns in data."
    
    print("=== Testing with longer text ===")
    print(f"Text length: {len(long_text)} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=[" ", ""]  # Only split on spaces/characters, not paragraphs
    )
    
    doc = Document(page_content=long_text)
    chunks = splitter.split_documents([doc])
    
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (len: {len(chunk.page_content)}):")
        print(f"'{chunk.page_content[:100]}...'")
    
    # Check overlaps between all consecutive chunks
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i].page_content
        chunk2 = chunks[i+1].page_content
        
        # Find actual overlap
        max_check = min(len(chunk1), len(chunk2), 100)  # Check up to 100 chars
        actual_overlap = 0
        
        for j in range(1, max_check + 1):
            if chunk1[-j:] == chunk2[:j]:
                actual_overlap = j
        
        print(f"Overlap between chunk {i+1} and {i+2}: {actual_overlap} characters")
        if actual_overlap > 0:
            overlap_text = chunk1[-actual_overlap:]
            print(f"  Overlapping text: '{overlap_text}'")

def test_overlap_scenarios():
    """Test different scenarios to understand when overlap works"""
    
    # Scenario 1: Text with natural paragraph breaks
    text_with_paragraphs = """First paragraph is here with some content that talks about AI and machine learning.

Second paragraph continues the discussion about Azure services and cloud computing capabilities.

Third paragraph concludes with information about enterprise features and security."""
    
    # Scenario 2: Continuous text without breaks
    continuous_text = "First sentence talks about AI. Second sentence discusses machine learning. Third sentence covers Azure services. Fourth sentence explains cloud computing. Fifth sentence mentions enterprise features. Sixth sentence discusses security aspects."
    
    scenarios = [
        ("With Paragraphs + Paragraph Separators", text_with_paragraphs, ["\n\n", "\n", " ", ""]),
        ("With Paragraphs + Space Separators Only", text_with_paragraphs, [" ", ""]),
        ("Continuous Text + Paragraph Separators", continuous_text, ["\n\n", "\n", " ", ""]),
        ("Continuous Text + Space Separators Only", continuous_text, [" ", ""])
    ]
    
    for name, text, separators in scenarios:
        print(f"\n=== {name} ===")
        print(f"Text: '{text[:100]}...'")
        print(f"Separators: {separators}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=30,
            separators=separators
        )
        
        doc = Document(page_content=text)
        chunks = splitter.split_documents([doc])
        
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: '{chunk.page_content}' (len: {len(chunk.page_content)})")
        
        # Check actual overlaps
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i].page_content
                chunk2 = chunks[i+1].page_content
                
                # Find actual overlap
                actual_overlap = 0
                max_check = min(len(chunk1), len(chunk2))
                
                for j in range(1, max_check + 1):
                    if chunk1[-j:] == chunk2[:j]:
                        actual_overlap = j
                
                print(f"  → Overlap {i+1}-{i+2}: {actual_overlap} chars")
                if actual_overlap > 0:
                    print(f"    Overlapping: '{chunk1[-actual_overlap:]}'")
        
        print("-" * 50)

if __name__ == "__main__":
    test_overlap_behavior()
    print("\n" + "="*60 + "\n")
    test_with_longer_text()
    print("\n" + "="*60 + "\n")
    test_overlap_scenarios()
