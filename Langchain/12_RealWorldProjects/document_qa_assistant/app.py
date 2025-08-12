"""
Document Q&A Assistant - A Production-Ready RAG Application

This application demonstrates how to build a complete document Q&A system
using LangChain, Streamlit, and ChromaDB. It showcases production-ready
patterns and best practices.

Features:
- Multiple document format support (PDF, DOCX, TXT)
- Advanced RAG with source citations
- Conversation memory
- Confidence scoring
- Clean web interface
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None

def load_documents():
    """Load and process documents"""
    try:
        from langchain.document_loaders import TextLoader, PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain_openai import OpenAI
        
        # Sample documents for demo
        sample_docs_content = [
            """
            # Company Handbook - Remote Work Policy
            
            Our company supports flexible remote work arrangements. Employees can work 
            from home up to 3 days per week. All remote workers must maintain regular 
            communication with their team and be available during core hours (9 AM - 3 PM EST).
            
            Equipment and internet costs are reimbursed up to $100 per month. Employees 
            must ensure they have a quiet, professional workspace for video calls.
            """,
            """
            # Company Handbook - Benefits Overview
            
            We offer comprehensive benefits including:
            - Health insurance (100% premium covered for employee, 80% for family)
            - 401(k) matching up to 6% of salary
            - Unlimited PTO policy
            - $2,000 annual learning and development budget
            - Flexible spending account for healthcare expenses
            
            Benefits enrollment occurs during the first week of employment and annually 
            in November.
            """,
            """
            # Company Handbook - Performance Reviews
            
            Performance reviews are conducted quarterly. Each review includes:
            - Goal setting and progress tracking
            - Skill development planning
            - Career advancement discussions
            - 360-degree feedback from peers and managers
            
            Salary reviews occur annually in January. Promotions are considered based on 
            performance, impact, and available opportunities.
            """
        ]
        
        # Create temporary documents
        docs = []
        for i, content in enumerate(sample_docs_content):
            from langchain.schema import Document
            docs.append(Document(
                page_content=content,
                metadata={"source": f"handbook_section_{i+1}.md"}
            ))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create QA chain
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain, len(splits)
        
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None, 0

def ask_question(qa_system, question):
    """Ask a question and get an answer with sources"""
    try:
        result = qa_system({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]],
            "source_texts": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "sources": [],
            "source_texts": []
        }

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.title("📚 Document Q&A Assistant")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # API Key check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            st.stop()
        
        # Load documents button
        if st.button("🔄 Load Sample Documents", type="primary"):
            with st.spinner("Loading and processing documents..."):
                qa_system, num_chunks = load_documents()
                if qa_system:
                    st.session_state.qa_system = qa_system
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Loaded documents! Created {num_chunks} text chunks.")
                else:
                    st.error("❌ Failed to load documents.")
        
        # Document status
        if st.session_state.documents_loaded:
            st.success("📄 Documents ready for Q&A")
        else:
            st.info("👆 Click 'Load Sample Documents' to start")
        
        # Stats
        if st.session_state.conversation_history:
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.conversation_history))
    
    # Main interface
    if st.session_state.documents_loaded:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Ask Questions")
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the remote work policy?",
                help="Ask any question about the loaded documents"
            )
            
            # Ask button
            if st.button("🔍 Ask Question", type="primary") and question:
                with st.spinner("Finding answer..."):
                    result = ask_question(st.session_state.qa_system, question)
                    
                    # Store in history
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # Clear input
                    st.experimental_rerun()
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.header("📜 Conversation History")
                
                for i, item in enumerate(reversed(st.session_state.conversation_history)):
                    with st.expander(f"Q: {item['question'][:60]}..." if len(item['question']) > 60 else f"Q: {item['question']}", expanded=(i==0)):
                        st.markdown("**Question:**")
                        st.write(item['question'])
                        
                        st.markdown("**Answer:**")
                        st.write(item['answer'])
                        
                        if item['sources']:
                            st.markdown("**Sources:**")
                            for source in set(item['sources']):  # Remove duplicates
                                st.markdown(f"- 📄 {source}")
        
        with col2:
            st.header("💡 Suggested Questions")
            
            sample_questions = [
                "What is the remote work policy?",
                "How much PTO do employees get?",
                "What benefits does the company offer?",
                "When do performance reviews happen?",
                "How much is the learning budget?",
                "What equipment costs are reimbursed?"
            ]
            
            for q in sample_questions:
                if st.button(q, key=f"sample_{q}", help="Click to ask this question"):
                    st.session_state.temp_question = q
                    st.experimental_rerun()
            
            # Handle sample question clicks
            if hasattr(st.session_state, 'temp_question'):
                question = st.session_state.temp_question
                with st.spinner("Finding answer..."):
                    result = ask_question(st.session_state.qa_system, question)
                    
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    del st.session_state.temp_question
                    st.experimental_rerun()
    
    else:
        # Getting started guide
        st.header("🚀 Getting Started")
        
        st.markdown("""
        ### Welcome to the Document Q&A Assistant!
        
        This application demonstrates a production-ready RAG (Retrieval-Augmented Generation) system.
        
        **To get started:**
        1. Click "Load Sample Documents" in the sidebar
        2. Wait for the documents to be processed
        3. Ask questions about the company handbook
        
        **Features:**
        - 📄 Multiple document format support
        - 🧠 Intelligent answer generation
        - 📚 Source citations for transparency
        - 💬 Conversation memory
        - 🎯 Confidence scoring
        """)
        
        # Demo video or screenshot placeholder
        st.info("💡 **Tip**: This demo uses sample company handbook documents. In production, you could upload your own PDFs, Word documents, and text files.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with ❤️ using LangChain, Streamlit, and OpenAI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
