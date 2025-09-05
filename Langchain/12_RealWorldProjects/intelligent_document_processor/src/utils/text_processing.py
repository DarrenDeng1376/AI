"""
Text processing utilities for document chunking and preprocessing
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import tiktoken

from config import app_config

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of text chunks"""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    MIXED = "mixed"

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    chunk_type: ChunkType
    page_number: Optional[int]
    chunk_index: int
    character_count: int
    word_count: int
    confidence_score: float
    source_section: str
    
    @property
    def token_count(self) -> int:
        """Estimate token count using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(self.content))
        except Exception:
            # Fallback estimation: ~4 characters per token
            return len(self.content) // 4

class TextChunker:
    """Handles intelligent text chunking for document processing"""
    
    def __init__(self):
        """Initialize the text chunker"""
        self.chunk_size = app_config.chunk_size
        self.chunk_overlap = app_config.chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken: {e}")
            self.tokenizer = None
    
    def chunk_document(
        self,
        text: str,
        paragraphs: List[str] = None,
        tables: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """
        Chunk document content intelligently
        
        Args:
            text: Full document text
            paragraphs: List of document paragraphs
            tables: List of table data
            metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_index = 0
        
        logger.info(f"Chunking document with {len(text)} characters")
        
        # Process paragraphs first (preferred chunking method)
        if paragraphs:
            chunks.extend(self._chunk_paragraphs(paragraphs, chunk_index))
            chunk_index += len(chunks)
        
        # Process tables separately
        if tables:
            table_chunks = self._chunk_tables(tables, chunk_index)
            chunks.extend(table_chunks)
            chunk_index += len(table_chunks)
        
        # If no structured content, fall back to simple text chunking
        if not chunks:
            chunks = self._chunk_text_simple(text, chunk_index)
        
        # Add any remaining text not captured in paragraphs
        elif paragraphs:
            # Check if we missed any significant content
            covered_text = " ".join([chunk.content for chunk in chunks])
            if len(covered_text) < len(text) * 0.8:  # Less than 80% covered
                remaining_chunks = self._chunk_text_simple(text, chunk_index)
                chunks.extend(remaining_chunks)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_paragraphs(self, paragraphs: List[str], start_index: int) -> List[TextChunk]:
        """Chunk document by paragraphs, combining when necessary"""
        chunks = []
        current_chunk = ""
        current_paragraphs = []
        chunk_index = start_index
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self._get_token_count(potential_chunk) <= self.chunk_size:
                # Add to current chunk
                current_chunk = potential_chunk
                current_paragraphs.append(i)
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        chunk_type=ChunkType.PARAGRAPH,
                        chunk_index=chunk_index,
                        source_section=f"paragraphs_{min(current_paragraphs)}-{max(current_paragraphs)}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with current paragraph
                if self._get_token_count(paragraph) <= self.chunk_size:
                    current_chunk = paragraph
                    current_paragraphs = [i]
                else:
                    # Paragraph is too long, split it
                    para_chunks = self._chunk_long_text(paragraph, chunk_index, ChunkType.PARAGRAPH)
                    chunks.extend(para_chunks)
                    chunk_index += len(para_chunks)
                    current_chunk = ""
                    current_paragraphs = []
        
        # Add final chunk if exists
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                chunk_type=ChunkType.PARAGRAPH,
                chunk_index=chunk_index,
                source_section=f"paragraphs_{min(current_paragraphs)}-{max(current_paragraphs)}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_tables(self, tables: List[Dict[str, Any]], start_index: int) -> List[TextChunk]:
        """Process tables into chunks"""
        chunks = []
        chunk_index = start_index
        
        for i, table in enumerate(tables):
            # Convert table to text representation
            table_text = self._table_to_text(table)
            
            if self._get_token_count(table_text) <= self.chunk_size:
                # Table fits in one chunk
                chunk = self._create_chunk(
                    content=table_text,
                    chunk_type=ChunkType.TABLE,
                    chunk_index=chunk_index,
                    source_section=f"table_{i}"
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Table is too large, split by rows
                table_chunks = self._chunk_large_table(table, chunk_index, i)
                chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
        
        return chunks
    
    def _chunk_text_simple(self, text: str, start_index: int) -> List[TextChunk]:
        """Simple text chunking with overlap"""
        chunks = []
        chunk_index = start_index
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split by sentences for better chunk boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._get_token_count(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        chunk_type=ChunkType.MIXED,
                        chunk_index=chunk_index,
                        source_section=f"text_chunk_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_sentences:
                    overlap_text = self._create_overlap(current_sentences)
                    current_chunk = overlap_text + " " + sentence
                    current_sentences = [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                chunk_type=ChunkType.MIXED,
                chunk_index=chunk_index,
                source_section=f"text_chunk_{chunk_index}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_long_text(
        self, 
        text: str, 
        start_index: int, 
        chunk_type: ChunkType
    ) -> List[TextChunk]:
        """Handle text that's longer than chunk size"""
        chunks = []
        chunk_index = start_index
        
        # Split by sentences first
        sentences = self._split_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._get_token_count(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        chunk_type=chunk_type,
                        chunk_index=chunk_index,
                        source_section=f"long_text_chunk_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # If single sentence is too long, split by words
                if self._get_token_count(sentence) > self.chunk_size:
                    word_chunks = self._chunk_by_words(sentence, chunk_index, chunk_type)
                    chunks.extend(word_chunks)
                    chunk_index += len(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                chunk_type=chunk_type,
                chunk_index=chunk_index,
                source_section=f"long_text_chunk_{chunk_index}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_words(
        self, 
        text: str, 
        start_index: int, 
        chunk_type: ChunkType
    ) -> List[TextChunk]:
        """Split text by words when sentences are too long"""
        chunks = []
        words = text.split()
        chunk_index = start_index
        
        current_chunk_words = []
        
        for word in words:
            current_chunk_words.append(word)
            current_text = " ".join(current_chunk_words)
            
            if self._get_token_count(current_text) > self.chunk_size:
                # Remove last word and save chunk
                if len(current_chunk_words) > 1:
                    current_chunk_words.pop()
                    chunk_text = " ".join(current_chunk_words)
                    
                    chunk = self._create_chunk(
                        content=chunk_text,
                        chunk_type=chunk_type,
                        chunk_index=chunk_index,
                        source_section=f"word_chunk_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_words = current_chunk_words[-self.chunk_overlap:]
                        current_chunk_words = overlap_words + [word]
                    else:
                        current_chunk_words = [word]
                else:
                    # Single word is too long, keep it anyway
                    chunk = self._create_chunk(
                        content=word,
                        chunk_type=chunk_type,
                        chunk_index=chunk_index,
                        source_section=f"word_chunk_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_words = []
        
        # Add final chunk
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk = self._create_chunk(
                content=chunk_text,
                chunk_type=chunk_type,
                chunk_index=chunk_index,
                source_section=f"word_chunk_{chunk_index}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_large_table(
        self, 
        table: Dict[str, Any], 
        start_index: int, 
        table_index: int
    ) -> List[TextChunk]:
        """Split large tables into chunks"""
        chunks = []
        chunk_index = start_index
        
        # Get table headers
        headers = self._extract_table_headers(table)
        header_text = " | ".join(headers) if headers else ""
        
        # Process table rows in batches
        cells = table.get("cells", [])
        rows = self._group_cells_by_row(cells)
        
        current_rows = []
        current_text = header_text
        
        for row in rows:
            row_text = " | ".join([cell.get("content", "") for cell in row])
            potential_text = current_text + "\n" + row_text if current_text else row_text
            
            if self._get_token_count(potential_text) <= self.chunk_size:
                current_text = potential_text
                current_rows.append(row)
            else:
                # Save current chunk
                if current_text:
                    chunk = self._create_chunk(
                        content=current_text,
                        chunk_type=ChunkType.TABLE,
                        chunk_index=chunk_index,
                        source_section=f"table_{table_index}_part_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with headers
                current_text = header_text + "\n" + row_text if header_text else row_text
                current_rows = [row]
        
        # Add final chunk
        if current_text:
            chunk = self._create_chunk(
                content=current_text,
                chunk_type=ChunkType.TABLE,
                chunk_index=chunk_index,
                source_section=f"table_{table_index}_part_{chunk_index}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        chunk_index: int,
        source_section: str,
        page_number: Optional[int] = None
    ) -> TextChunk:
        """Create a TextChunk object"""
        content = content.strip()
        
        return TextChunk(
            content=content,
            chunk_type=chunk_type,
            page_number=page_number,
            chunk_index=chunk_index,
            character_count=len(content),
            word_count=len(content.split()),
            confidence_score=0.9,  # Default confidence
            source_section=source_section
        )
    
    def _get_token_count(self, text: str) -> int:
        """Get accurate token count"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation
        return len(text) // 4
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (could be improved with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_overlap(self, sentences: List[str]) -> str:
        """Create overlap text from previous sentences"""
        if not sentences or self.chunk_overlap <= 0:
            return ""
        
        # Take last N sentences for overlap
        overlap_sentences = sentences[-self.chunk_overlap:]
        return " ".join(overlap_sentences)
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table structure to text representation"""
        try:
            cells = table.get("cells", [])
            if not cells:
                return ""
            
            # Group cells by row
            rows = self._group_cells_by_row(cells)
            
            # Convert to text
            text_rows = []
            for row in rows:
                row_text = " | ".join([cell.get("content", "") for cell in row])
                if row_text.strip():
                    text_rows.append(row_text)
            
            return "\n".join(text_rows)
            
        except Exception as e:
            logger.warning(f"Error converting table to text: {e}")
            return ""
    
    def _group_cells_by_row(self, cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group table cells by row"""
        rows = {}
        
        for cell in cells:
            row_index = cell.get("row_index", 0)
            if row_index not in rows:
                rows[row_index] = []
            rows[row_index].append(cell)
        
        # Sort by column index within each row
        for row_index in rows:
            rows[row_index].sort(key=lambda c: c.get("column_index", 0))
        
        # Return rows in order
        return [rows[i] for i in sorted(rows.keys())]
    
    def _extract_table_headers(self, table: Dict[str, Any]) -> List[str]:
        """Extract table headers if available"""
        cells = table.get("cells", [])
        if not cells:
            return []
        
        # Get first row as headers
        header_cells = [cell for cell in cells if cell.get("row_index") == 0]
        header_cells.sort(key=lambda c: c.get("column_index", 0))
        
        return [cell.get("content", "") for cell in header_cells]

# Example usage and testing
if __name__ == "__main__":
    chunker = TextChunker()
    
    # Test with sample text
    sample_text = """
    This is a sample document about artificial intelligence. 
    It contains multiple paragraphs and demonstrates text chunking.
    
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals.
    
    Machine learning is a subset of AI that focuses on algorithms that can 
    learn and make decisions from data.
    """
    
    paragraphs = [
        "This is a sample document about artificial intelligence. It contains multiple paragraphs and demonstrates text chunking.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
        "Machine learning is a subset of AI that focuses on algorithms that can learn and make decisions from data."
    ]
    
    chunks = chunker.chunk_document(sample_text, paragraphs)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Type: {chunk.chunk_type.value}")
        print(f"  Length: {chunk.character_count} chars, {chunk.word_count} words, ~{chunk.token_count} tokens")
        print(f"  Content: {chunk.content[:100]}...")