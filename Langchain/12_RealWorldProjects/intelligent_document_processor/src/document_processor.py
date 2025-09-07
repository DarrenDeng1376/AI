"""
Azure Document Intelligence integration for document processing
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import io

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.formrecognizer.models import AnalyzeResult
from azure.core.exceptions import AzureError

from config import azure_config, app_config
from .utils.azure_clients import AzureClientManager

logger = logging.getLogger(__name__)

@dataclass
class DocumentContent:
    """Structured document content"""
    text: str
    tables: List[Dict[str, Any]]
    key_value_pairs: List[Dict[str, str]]
    paragraphs: List[str]
    pages: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Document processing result"""
    success: bool
    content: Optional[DocumentContent]
    error_message: Optional[str]
    processing_time: float
    document_type: str
    page_count: int

class DocumentProcessor:
    """Azure Document Intelligence wrapper for document processing"""
    
    def __init__(self):
        """Initialize the document processor"""
        try:
            self.client_manager = AzureClientManager()
            self.client = self.client_manager.get_document_intelligence_client()
            logger.info("Document Intelligence client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence client: {e}")
            raise
    
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str,
        model_id: str = "prebuilt-layout"
    ) -> ProcessingResult:
        """
        Process a document using Azure Document Intelligence
        
        Args:
            file_content: Raw file bytes
            filename: Original filename for context
            model_id: Model to use for processing
            
        Returns:
            ProcessingResult with extracted content
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing for: {filename}")
            
            # Determine file type
            file_extension = Path(filename).suffix.lower()
            content_type = self._get_content_type(file_extension)
            
            # Create analysis request
            file_stream = io.BytesIO(file_content)
            
            # Start analysis
            poller = self.client.begin_analyze_document(
                model_id=model_id,
                document=file_stream,
                content_type=content_type
            )
            
            # Wait for completion
            result: AnalyzeResult = poller.result()
            
            # Extract content
            document_content = self._extract_content(result, filename)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document processing completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                content=document_content,
                error_message=None,
                processing_time=processing_time,
                document_type=file_extension,
                page_count=len(result.pages) if result.pages else 0
            )
            
        except AzureError as e:
            error_msg = f"Azure Document Intelligence error: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                content=None,
                error_message=error_msg,
                processing_time=time.time() - start_time,
                document_type=file_extension,
                page_count=0
            )
        except Exception as e:
            error_msg = f"Unexpected error processing document: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                content=None,
                error_message=error_msg,
                processing_time=time.time() - start_time,
                document_type=file_extension,
                page_count=0
            )
    
    def _get_content_type(self, file_extension: str) -> str:
        """Get MIME content type for file extension"""
        content_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        return content_types.get(file_extension, 'application/octet-stream')
    
    def _extract_content(self, result: AnalyzeResult, filename: str) -> DocumentContent:
        """Extract structured content from analysis result"""
        
        # Extract main text content
        text_content = ""
        if result.content:
            text_content = result.content
        
        # Extract paragraphs
        paragraphs = []
        if result.paragraphs:
            paragraphs = [p.content for p in result.paragraphs if p.content]
        
        # Extract tables
        tables = []
        if result.tables:
            for table in result.tables:
                table_data = {
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cells': []
                }
                
                if table.cells:
                    for cell in table.cells:
                        cell_data = {
                            'content': cell.content,
                            'row_index': cell.row_index,
                            'column_index': cell.column_index,
                            'row_span': getattr(cell, 'row_span', 1),
                            'column_span': getattr(cell, 'column_span', 1),
                            'confidence': getattr(cell, 'confidence', 0.0)
                        }
                        table_data['cells'].append(cell_data)
                
                tables.append(table_data)
        
        # Extract key-value pairs
        key_value_pairs = []
        if result.key_value_pairs:
            for kv in result.key_value_pairs:
                if kv.key and kv.value:
                    key_value_pairs.append({
                        'key': kv.key.content if kv.key.content else "",
                        'value': kv.value.content if kv.value.content else "",
                        'confidence': getattr(kv, 'confidence', 0.0)
                    })
        
        # Extract page information
        pages = []
        if result.pages:
            for page in result.pages:
                page_data = {
                    'page_number': page.page_number,
                    'width': page.width,
                    'height': page.height,
                    'unit': page.unit,
                    'angle': getattr(page, 'angle', 0.0),
                    'lines': []
                }
                
                if page.lines:
                    for line in page.lines:
                        page_data['lines'].append({
                            'content': line.content,
                            'polygon': [{'x': p.x, 'y': p.y} for p in line.polygon] if line.polygon else []
                        })
                
                pages.append(page_data)
        
        # Calculate confidence scores
        confidence_scores = {
            'overall': 0.8,  # Default confidence
            'text_extraction': 0.9,
            'table_extraction': 0.8,
            'key_value_extraction': 0.7
        }
        
        # Document metadata
        metadata = {
            'filename': filename,
            'total_pages': len(pages),
            'has_tables': len(tables) > 0,
            'has_key_value_pairs': len(key_value_pairs) > 0,
            'paragraph_count': len(paragraphs),
            'character_count': len(text_content),
            'processing_model': 'prebuilt-layout'
        }
        
        return DocumentContent(
            text=text_content,
            tables=tables,
            key_value_pairs=key_value_pairs,
            paragraphs=paragraphs,
            pages=pages,
            confidence_scores=confidence_scores,
            metadata=metadata
        )
    
    async def process_multiple_documents(
        self, 
        documents: List[Tuple[bytes, str]]
    ) -> List[ProcessingResult]:
        """
        Process multiple documents concurrently
        
        Args:
            documents: List of (file_content, filename) tuples
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Processing {len(documents)} documents")
        
        # Process documents with concurrency limit
        semaphore = asyncio.Semaphore(app_config.max_concurrent_requests)
        
        async def process_with_semaphore(doc_data):
            async with semaphore:
                return await self.process_document(doc_data[0], doc_data[1])
        
        tasks = [process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing document {documents[i][1]}: {result}")
                processed_results.append(ProcessingResult(
                    success=False,
                    content=None,
                    error_message=str(result),
                    processing_time=0.0,
                    document_type="unknown",
                    page_count=0
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats"""
        return [
            'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp',
            'docx', 'xlsx', 'pptx'
        ]
    
    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate if file can be processed
        
        Returns:
            (is_valid, error_message)
        """
        # Check file extension
        file_extension = Path(filename).suffix.lower().replace('.', '')
        if file_extension not in self.get_supported_formats():
            return False, f"Unsupported file format: {file_extension}"
        
        # Check file size
        max_size_bytes = app_config.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"File too large: {file_size / (1024*1024):.1f}MB (max: {app_config.max_file_size_mb}MB)"
        
        return True, None

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_processor():
        processor = DocumentProcessor()
        
        # Test with a sample file (you would need to provide actual file content)
        sample_file = b"Sample document content"
        filename = "test.pdf"
        
        result = await processor.process_document(sample_file, filename)
        
        print(f"Processing successful: {result.success}")
        if result.success and result.content:
            print(f"Extracted text length: {len(result.content.text)}")
            print(f"Number of tables: {len(result.content.tables)}")
            print(f"Number of pages: {len(result.content.pages)}")
        else:
            print(f"Error: {result.error_message}")
    
    # Run test
    # asyncio.run(test_processor())
