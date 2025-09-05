"""
Intelligent Document Processor - Streamlit Web Application
A comprehensive document processing and Q&A system using Azure OpenAI and embeddings
"""
import streamlit as st
import asyncio
import time
import json
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from config import azure_config, app_config, validate_configuration, get_config_summary
from src.document_processor import DocumentProcessor, ProcessingResult
from src.embedding_manager import EmbeddingManager, EmbeddingResult, SearchResult
from src.qa_engine import QAEngine, Answer, AnswerConfidence
from src.utils.file_handlers import FileHandler, BatchFileProcessor, get_file_type_icon, format_file_size
from src.utils.azure_clients import AzureClientManager, validate_azure_configuration

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #fd7e14; font-weight: bold; }
    .confidence-very-low { color: #dc3545; font-weight: bold; }
    .source-citation {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-left: 3px solid #007bff;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class DocumentProcessorApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.initialize_session_state()
        self.load_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
        
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = f"session_{int(time.time())}"
        
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = False
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
    
    def load_components(self):
        """Load and initialize application components"""
        try:
            # Initialize components
            self.file_handler = FileHandler()
            self.document_processor = DocumentProcessor()
            self.embedding_manager = EmbeddingManager("main_collection")
            self.qa_engine = QAEngine(self.embedding_manager)
            self.client_manager = AzureClientManager()
            
            st.session_state.app_initialized = True
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize application components: {str(e)}")
            logger.error(f"Application initialization failed: {e}")
            st.session_state.app_initialized = False
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🤖 Intelligent Document Processor")
        st.markdown("**Powered by Azure OpenAI and Document Intelligence**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check configuration
        if not self.check_configuration():
            return
        
        # Sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Document Upload & Processing",
            "❓ Question & Answer",
            "📊 Document Analytics",
            "⚙️ System Status"
        ])
        
        with tab1:
            self.render_document_processing_tab()
        
        with tab2:
            self.render_qa_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_system_status_tab()
    
    def check_configuration(self) -> bool:
        """Check if the application is properly configured"""
        # Validate configuration
        is_valid, issues = validate_configuration()
        azure_issues = validate_azure_configuration()
        
        if not is_valid or azure_issues:
            st.error("⚠️ Configuration Issues Detected")
            
            if azure_issues:
                st.write("**Azure Configuration Issues:**")
                for issue in azure_issues:
                    st.write(f"- {issue}")
            
            if issues:
                st.write("**Application Configuration Issues:**")
                for issue in issues:
                    st.write(f"- {issue}")
            
            st.write("Please check your configuration and restart the application.")
            return False
        
        return True
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        with st.sidebar:
            st.header("🔧 Controls")
            
            # Quick stats
            if st.session_state.app_initialized:
                stats = self.embedding_manager.get_document_stats()
                
                st.subheader("📈 Quick Stats")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Documents", stats.get("total_documents", 0))
                
                with col2:
                    st.metric("Chunks", stats.get("total_chunks", 0))
            
            # Document management
            st.subheader("📚 Document Management")
            
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    self.clear_all_documents()
                    st.session_state.confirm_clear = False
                    st.success("All documents cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm clearing all documents")
            
            # Configuration summary
            st.subheader("⚙️ Configuration")
            config_summary = get_config_summary()
            
            for key, value in config_summary.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key.replace('_', ' ').title()}")
                else:
                    st.write(f"📋 {key.replace('_', ' ').title()}: {value}")
    
    def render_document_processing_tab(self):
        """Render the document processing tab"""
        st.header("📄 Document Upload & Processing")
        
        # File upload section
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose documents to process",
            type=app_config.allowed_file_types,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(app_config.allowed_file_types)}. Max size: {app_config.max_file_size_mb}MB per file."
        )
        
        if uploaded_files:
            # Show file information
            st.subheader("📋 Uploaded Files")
            
            file_data = []
            for file in uploaded_files:
                file_info = self.file_handler.validate_file(file.name, len(file.getvalue()))
                file_data.append({
                    "File": f"{get_file_type_icon(file_info.extension)} {file.name}",
                    "Size": format_file_size(file_info.size),
                    "Type": file_info.extension.upper(),
                    "Status": "✅ Valid" if file_info.is_valid else f"❌ {file_info.error_message}"
                })
            
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True)
            
            # Process button
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    self.process_documents(uploaded_files)
        
        # Processing results
        if st.session_state.processing_results:
            st.subheader("📊 Processing Results")
            self.display_processing_results()
    
    def process_documents(self, uploaded_files: List):
        """Process uploaded documents"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Process files
            batch_processor = BatchFileProcessor(self.file_handler)
            
            def update_progress(progress: float, message: str):
                progress_bar.progress(progress)
                status_text.text(message)
            
            # Step 1: Handle file uploads
            status_text.text("Preparing files...")
            file_results = batch_processor.process_file_batch(uploaded_files, update_progress)
            
            successful_files = [r for r in file_results["results"] if r.success]
            
            if not successful_files:
                st.error("No files could be processed successfully.")
                return
            
            # Step 2: Process documents with Azure Document Intelligence
            progress_bar.progress(0.3)
            status_text.text("Extracting document content...")
            
            processing_results = []
            
            for i, file_result in enumerate(successful_files):
                try:
                    # Read file content
                    success, content, error = self.file_handler.read_file_content(file_result.file_path)
                    
                    if not success:
                        st.error(f"Could not read file {file_result.file_info.filename}: {error}")
                        continue
                    
                    # Process with Document Intelligence
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    processing_result = loop.run_until_complete(
                        self.document_processor.process_document(
                            content, 
                            file_result.file_info.filename
                        )
                    )
                    
                    processing_results.append({
                        "file_result": file_result,
                        "processing_result": processing_result
                    })
                    
                    # Update progress
                    progress = 0.3 + (0.4 * (i + 1) / len(successful_files))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {file_result.file_info.filename}...")
                    
                except Exception as e:
                    st.error(f"Error processing {file_result.file_info.filename}: {str(e)}")
                    logger.error(f"Document processing error: {e}")
            
            # Step 3: Create embeddings
            progress_bar.progress(0.7)
            status_text.text("Creating embeddings...")
            
            embedding_results = []
            
            for i, result in enumerate(processing_results):
                if result["processing_result"].success:
                    try:
                        file_info = result["file_result"].file_info
                        doc_content = result["processing_result"].content
                        document_id = file_info.file_hash
                        
                        # Create embeddings
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        embedding_result = loop.run_until_complete(
                            self.embedding_manager.create_embeddings(
                                doc_content,
                                document_id,
                                file_info.filename
                            )
                        )
                        
                        embedding_results.append({
                            "file_info": file_info,
                            "embedding_result": embedding_result
                        })
                        
                        # Store in session state
                        st.session_state.documents[document_id] = {
                            "filename": file_info.filename,
                            "file_hash": file_info.file_hash,
                            "processing_result": result["processing_result"],
                            "embedding_result": embedding_result,
                            "upload_time": time.time()
                        }
                        
                        # Update progress
                        progress = 0.7 + (0.3 * (i + 1) / len(processing_results))
                        progress_bar.progress(progress)
                        status_text.text(f"Creating embeddings for {file_info.filename}...")
                        
                    except Exception as e:
                        st.error(f"Error creating embeddings for {file_info.filename}: {str(e)}")
                        logger.error(f"Embedding creation error: {e}")
            
            # Cleanup temporary files
            for result in processing_results:
                if result["file_result"].temp_file:
                    self.file_handler.cleanup_temp_file(result["file_result"].temp_file)
            
            # Update session state
            st.session_state.processing_results = embedding_results
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Show results
            successful_count = len([r for r in embedding_results if r["embedding_result"].success])
            st.success(f"Successfully processed {successful_count}/{len(uploaded_files)} documents!")
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Unexpected error during processing: {str(e)}")
            logger.error(f"Document processing pipeline error: {e}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def display_processing_results(self):
        """Display processing results"""
        results_data = []
        
        for result in st.session_state.processing_results:
            file_info = result["file_info"]
            embedding_result = result["embedding_result"]
            
            results_data.append({
                "Document": f"{get_file_type_icon(file_info.extension)} {file_info.filename}",
                "Status": "✅ Success" if embedding_result.success else f"❌ Failed",
                "Chunks": embedding_result.embedding_count if embedding_result.success else 0,
                "Processing Time": f"{embedding_result.processing_time:.2f}s",
                "Error": embedding_result.error_message or "-"
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
    
    def render_qa_tab(self):
        """Render the Q&A tab"""
        st.header("❓ Question & Answer")
        
        # Check if documents are available
        if not st.session_state.documents:
            st.info("Please upload and process documents first in the 'Document Upload & Processing' tab.")
            return
        
        # Document filter
        st.subheader("📚 Select Documents")
        
        document_options = ["All Documents"] + [
            doc_data["filename"] for doc_data in st.session_state.documents.values()
        ]
        
        selected_doc = st.selectbox(
            "Choose documents to search",
            options=document_options,
            help="Select specific documents or search all uploaded documents"
        )
        
        # Question input
        st.subheader("💬 Ask a Question")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the main topic of the documents? Who are the key people mentioned? What are the important dates?",
            help="Ask questions about the content of your uploaded documents"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_conversation_context = st.checkbox(
                "Use conversation context",
                value=True,
                help="Include previous questions and answers for better context"
            )
            
            show_sources = st.checkbox(
                "Show source citations",
                value=True,
                help="Display which document sections were used to answer the question"
            )
        
        # Submit question
        if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
            self.process_question(question, selected_doc, use_conversation_context, show_sources)
        
        # Display Q&A history
        if st.session_state.qa_history:
            st.subheader("📜 Question & Answer History")
            self.display_qa_history(show_sources)
    
    def process_question(
        self, 
        question: str, 
        selected_doc: str, 
        use_conversation_context: bool,
        show_sources: bool
    ):
        """Process a user question"""
        with st.spinner("🤔 Thinking..."):
            try:
                # Determine document filter
                document_filter = None
                if selected_doc != "All Documents":
                    # Find document ID by filename
                    for doc_id, doc_data in st.session_state.documents.items():
                        if doc_data["filename"] == selected_doc:
                            document_filter = doc_id
                            break
                
                # Get answer
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                answer = loop.run_until_complete(
                    self.qa_engine.answer_question(
                        question=question,
                        session_id=st.session_state.current_session_id,
                        document_filter=document_filter,
                        use_conversation_context=use_conversation_context
                    )
                )
                
                # Add to history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": time.time(),
                    "document_filter": selected_doc,
                    "show_sources": show_sources
                })
                
                # Display answer immediately
                self.display_answer(answer, show_sources)
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                logger.error(f"QA processing error: {e}")
    
    def display_answer(self, answer: Answer, show_sources: bool = True):
        """Display a single answer"""
        # Confidence indicator
        confidence_class = f"confidence-{answer.confidence.value.replace('_', '-')}"
        confidence_text = answer.confidence.value.replace('_', ' ').title()
        
        st.markdown(f"""
        <div class="status-card status-success">
            <strong>Answer</strong> 
            <span class="{confidence_class}">({confidence_text} Confidence: {answer.confidence_score:.1%})</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(answer.answer)
        
        # Show sources if requested and available
        if show_sources and answer.sources:
            with st.expander(f"📚 Sources ({len(answer.sources)} found)"):
                for i, source in enumerate(answer.sources):
                    st.markdown(f"""
                    <div class="source-citation">
                        <strong>Source {i+1}:</strong> {source.document_name}
                        {f" (Page {source.page_number})" if source.page_number else ""}
                        <br>
                        <strong>Relevance:</strong> {source.similarity_score:.1%}
                        <br>
                        <em>{source.content[:200]}...</em>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Follow-up questions
        if answer.follow_up_questions:
            st.subheader("💡 Suggested Follow-up Questions")
            for fq in answer.follow_up_questions:
                if st.button(fq, key=f"followup_{hash(fq)}"):
                    st.session_state.followup_question = fq
                    st.rerun()
        
        # Performance info
        with st.expander("⚡ Performance Details"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Processing Time", f"{answer.processing_time:.2f}s")
            
            with col2:
                if answer.token_usage:
                    st.metric("Tokens Used", answer.token_usage.get("total_tokens", 0))
            
            with col3:
                st.metric("Sources Found", len(answer.sources))
    
    def display_qa_history(self, show_sources: bool = True):
        """Display Q&A history"""
        for i, qa_item in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa_item['question'][:50]}..."):
                st.write(f"**Question:** {qa_item['question']}")
                st.write(f"**Document Scope:** {qa_item['document_filter']}")
                
                # Display answer
                self.display_answer(qa_item['answer'], qa_item.get('show_sources', show_sources))
    
    def render_analytics_tab(self):
        """Render the analytics tab"""
        st.header("📊 Document Analytics")
        
        if not st.session_state.documents:
            st.info("No documents available for analysis. Please upload documents first.")
            return
        
        # Document overview
        st.subheader("📋 Document Overview")
        
        # Create overview data
        overview_data = []
        for doc_id, doc_data in st.session_state.documents.items():
            processing_result = doc_data.get("processing_result")
            embedding_result = doc_data.get("embedding_result")
            
            if processing_result and processing_result.success:
                content = processing_result.content
                overview_data.append({
                    "Document": f"{get_file_type_icon(Path(doc_data['filename']).suffix.replace('.', ''))} {doc_data['filename']}",
                    "Pages": processing_result.page_count,
                    "Characters": content.metadata.get("character_count", 0),
                    "Tables": len(content.tables),
                    "Key-Value Pairs": len(content.key_value_pairs),
                    "Chunks": embedding_result.embedding_count if embedding_result and embedding_result.success else 0,
                    "Processing Time": f"{processing_result.processing_time:.2f}s"
                })
        
        if overview_data:
            df = pd.DataFrame(overview_data)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", len(overview_data))
            
            with col2:
                total_pages = sum(row["Pages"] for row in overview_data)
                st.metric("Total Pages", total_pages)
            
            with col3:
                total_chunks = sum(row["Chunks"] for row in overview_data)
                st.metric("Total Chunks", total_chunks)
            
            with col4:
                total_tables = sum(row["Tables"] for row in overview_data)
                st.metric("Total Tables", total_tables)
        
        # Document content analysis
        st.subheader("🔍 Content Analysis")
        
        selected_doc_for_analysis = st.selectbox(
            "Select document for detailed analysis",
            options=[doc_data["filename"] for doc_data in st.session_state.documents.values()],
            key="analysis_doc_selector"
        )
        
        if selected_doc_for_analysis:
            # Find the document
            doc_data = None
            for doc_id, data in st.session_state.documents.items():
                if data["filename"] == selected_doc_for_analysis:
                    doc_data = data
                    break
            
            if doc_data and doc_data.get("processing_result"):
                processing_result = doc_data["processing_result"]
                content = processing_result.content
                
                # Content details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Document Structure:**")
                    st.write(f"- Pages: {processing_result.page_count}")
                    st.write(f"- Paragraphs: {len(content.paragraphs)}")
                    st.write(f"- Tables: {len(content.tables)}")
                    st.write(f"- Key-Value Pairs: {len(content.key_value_pairs)}")
                
                with col2:
                    st.write("**Content Statistics:**")
                    st.write(f"- Total Characters: {len(content.text):,}")
                    st.write(f"- Word Count: ~{len(content.text.split()):,}")
                    st.write(f"- Processing Model: {content.metadata.get('processing_model', 'N/A')}")
                
                # Show tables if available
                if content.tables:
                    st.subheader("📊 Tables Found")
                    for i, table in enumerate(content.tables[:3]):  # Show first 3 tables
                        with st.expander(f"Table {i+1} ({table['row_count']}x{table['column_count']})"):
                            # Convert table to DataFrame for display
                            if table.get("cells"):
                                table_data = self.convert_table_to_dataframe(table)
                                if table_data is not None:
                                    st.dataframe(table_data)
                
                # Show key-value pairs if available
                if content.key_value_pairs:
                    st.subheader("🔑 Key-Value Pairs")
                    kv_data = []
                    for kv in content.key_value_pairs[:10]:  # Show first 10
                        kv_data.append({
                            "Key": kv["key"],
                            "Value": kv["value"],
                            "Confidence": f"{kv.get('confidence', 0):.1%}"
                        })
                    
                    if kv_data:
                        kv_df = pd.DataFrame(kv_data)
                        st.dataframe(kv_df, use_container_width=True)
    
    def convert_table_to_dataframe(self, table: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Convert table structure to pandas DataFrame"""
        try:
            cells = table.get("cells", [])
            if not cells:
                return None
            
            # Create matrix
            rows = table.get("row_count", 0)
            cols = table.get("column_count", 0)
            
            if rows == 0 or cols == 0:
                return None
            
            # Initialize matrix
            matrix = [["" for _ in range(cols)] for _ in range(rows)]
            
            # Fill matrix with cell content
            for cell in cells:
                row_idx = cell.get("row_index", 0)
                col_idx = cell.get("column_index", 0)
                content = cell.get("content", "")
                
                if 0 <= row_idx < rows and 0 <= col_idx < cols:
                    matrix[row_idx][col_idx] = content
            
            # Convert to DataFrame
            df = pd.DataFrame(matrix[1:], columns=matrix[0] if rows > 0 else None)
            return df
            
        except Exception as e:
            logger.error(f"Error converting table to DataFrame: {e}")
            return None
    
    def render_system_status_tab(self):
        """Render the system status tab"""
        st.header("⚙️ System Status")
        
        # Azure services status
        st.subheader("☁️ Azure Services")
        
        if st.button("🔄 Check Service Health", type="secondary"):
            with st.spinner("Checking Azure services..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    health_results = loop.run_until_complete(
                        self.client_manager.test_connections()
                    )
                    
                    for service, status in health_results.items():
                        status_class = "status-success" if status else "status-error"
                        status_text = "✅ Healthy" if status else "❌ Error"
                        
                        st.markdown(f"""
                        <div class="status-card {status_class}">
                            <strong>{service.replace('_', ' ').title()}:</strong> {status_text}
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error checking service health: {str(e)}")
        
        # Application statistics
        st.subheader("📊 Application Statistics")
        
        if st.session_state.app_initialized:
            stats = self.embedding_manager.get_document_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            
            with col2:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            
            with col3:
                st.metric("QA Sessions", len(st.session_state.qa_history))
            
            with col4:
                st.metric("Session ID", st.session_state.current_session_id[-8:])
        
        # Configuration details
        st.subheader("🔧 Configuration")
        
        config_summary = get_config_summary()
        
        config_df = pd.DataFrame([
            {"Setting": key.replace('_', ' ').title(), "Value": str(value)}
            for key, value in config_summary.items()
        ])
        
        st.dataframe(config_df, use_container_width=True)
        
        # Session management
        st.subheader("🔄 Session Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆕 New Session", type="secondary"):
                st.session_state.current_session_id = f"session_{int(time.time())}"
                st.session_state.qa_history = []
                st.success("New session started!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear QA History", type="secondary"):
                st.session_state.qa_history = []
                st.success("QA history cleared!")
                st.rerun()
    
    def clear_all_documents(self):
        """Clear all documents and reset the application state"""
        try:
            # Clear vector store
            self.embedding_manager.clear_all_documents()
            
            # Clear session state
            st.session_state.documents = {}
            st.session_state.processing_results = []
            st.session_state.qa_history = []
            
            # Cleanup temporary files
            self.file_handler.cleanup_all_temp_files()
            
            logger.info("All documents cleared successfully")
            
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")
            logger.error(f"Error clearing documents: {e}")

# Main application entry point
def main():
    """Main application entry point"""
    try:
        app = DocumentProcessorApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show error details in debug mode
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()
