"""
Vector embedding management using Azure OpenAI and ChromaDB
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI

from config import azure_config, app_config, embedding_config
from .utils.text_processing import TextChunker
from .utils.azure_clients import AzureClientManager
from .document_processor import DocumentContent

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    success: bool
    chunk_ids: List[str]
    error_message: Optional[str]
    embedding_count: int
    processing_time: float

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    content: str
    similarity_score: float
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    metadata: Dict[str, Any]

class EmbeddingManager:
    """Manages document embeddings and vector search using Azure OpenAI and ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        """Initialize the embedding manager"""
        self.collection_name = collection_name
        self.text_chunker = TextChunker()
        
        # Initialize Azure OpenAI client through client manager
        try:
            self.client_manager = AzureClientManager()
            self.openai_client = self.client_manager.get_openai_client()
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
        
        # Initialize ChromaDB
        try:
            # Ensure vector store directory exists
            Path(app_config.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=app_config.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for intelligent search"}
            )
            
            logger.info(f"ChromaDB collection '{collection_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def create_embeddings(
        self, 
        document_content: DocumentContent, 
        document_id: str,
        document_name: str
    ) -> EmbeddingResult:
        """
        Create embeddings for document content
        
        Args:
            document_content: Processed document content
            document_id: Unique document identifier
            document_name: Human-readable document name
            
        Returns:
            EmbeddingResult with operation details
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Creating embeddings for document: {document_name}")
            
            # Chunk the document text
            chunks = self.text_chunker.chunk_document(
                text=document_content.text,
                paragraphs=document_content.paragraphs,
                tables=document_content.tables,
                metadata=document_content.metadata
            )
            
            logger.info(f"Created {len(chunks)} chunks for document")
            
            # Limit chunks if necessary
            if len(chunks) > app_config.max_chunks_per_document:
                logger.warning(f"Document has {len(chunks)} chunks, limiting to {app_config.max_chunks_per_document}")
                chunks = chunks[:app_config.max_chunks_per_document]
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self._generate_embeddings_batch(chunk_texts)
            
            # Prepare data for ChromaDB
            chunk_ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_{i}"
                chunk_ids.append(chunk_id)
                documents.append(chunk.content)
                
                metadata = {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": i,
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_number,
                    "character_count": len(chunk.content),
                    "word_count": len(chunk.content.split()),
                    "confidence_score": chunk.confidence_score,
                    "source_section": chunk.source_section,
                    "created_at": time.time()
                }
                
                # Add document-level metadata
                metadata.update({
                    "doc_total_pages": document_content.metadata.get("total_pages", 0),
                    "doc_has_tables": document_content.metadata.get("has_tables", False),
                    "doc_character_count": document_content.metadata.get("character_count", 0),
                    "doc_processing_model": document_content.metadata.get("processing_model", "unknown")
                })
                
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully created embeddings for {len(chunks)} chunks in {processing_time:.2f}s")
            
            return EmbeddingResult(
                success=True,
                chunk_ids=chunk_ids,
                error_message=None,
                embedding_count=len(embeddings),
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Failed to create embeddings: {str(e)}"
            logger.error(error_msg)
            return EmbeddingResult(
                success=False,
                chunk_ids=[],
                error_message=error_msg,
                embedding_count=0,
                processing_time=time.time() - start_time
            )
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            # Azure OpenAI has limits on batch size and tokens
            batch_size = 100  # Conservative batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Clean texts (remove excessive whitespace, etc.)
                cleaned_texts = [self._clean_text_for_embedding(text) for text in batch_texts]
                
                response = self.openai_client.embeddings.create(
                    input=cleaned_texts,
                    model=azure_config.openai_embedding_deployment
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text for embedding generation"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (Azure OpenAI has token limits)
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    async def search_similar_content(
        self, 
        query: str, 
        max_results: int = None,
        document_filter: Optional[str] = None,
        similarity_threshold: float = None
    ) -> List[SearchResult]:
        """
        Search for similar content using vector similarity
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            document_filter: Filter by specific document ID
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        try:
            if max_results is None:
                max_results = embedding_config.max_search_results
            
            if similarity_threshold is None:
                similarity_threshold = embedding_config.similarity_threshold
            
            logger.info(f"Searching for: {query}")
            
            # Generate embedding for query
            query_embedding = await self._generate_embeddings_batch([query])
            
            # Prepare filter
            where_filter = {}
            if document_filter:
                where_filter["document_id"] = document_filter
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=max_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results["ids"] and len(results["ids"]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    distance = results["distances"][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    # Filter by similarity threshold
                    if similarity_score < similarity_threshold:
                        continue
                    
                    metadata = results["metadatas"][0][i]
                    content = results["documents"][0][i]
                    
                    search_result = SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        similarity_score=similarity_score,
                        document_name=metadata.get("document_name", "Unknown"),
                        page_number=metadata.get("page_number"),
                        chunk_index=metadata.get("chunk_index", 0),
                        metadata=metadata
                    )
                    
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} relevant chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search content: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        max_results: int = None,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search
        """
        if not embedding_config.enable_hybrid_search:
            return await self.search_similar_content(query, max_results, document_filter)
        
        # Get semantic results
        semantic_results = await self.search_similar_content(
            query, max_results * 2, document_filter  # Get more for reranking
        )
        
        # Simple keyword search in the semantic results
        keyword_results = self._keyword_search(query, semantic_results)
        
        # Combine and rerank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, query
        )
        
        return combined_results[:max_results] if max_results else combined_results
    
    def _keyword_search(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """Simple keyword search within candidate results"""
        query_terms = set(query.lower().split())
        keyword_results = []
        
        for result in candidates:
            content_words = set(result.content.lower().split())
            matches = len(query_terms.intersection(content_words))
            
            if matches > 0:
                # Create a copy with keyword score
                keyword_result = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    similarity_score=matches / len(query_terms),  # Keyword match ratio
                    document_name=result.document_name,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    metadata=result.metadata
                )
                keyword_results.append(keyword_result)
        
        return keyword_results
    
    def _combine_search_results(
        self, 
        semantic_results: List[SearchResult], 
        keyword_results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Combine and rerank semantic and keyword search results"""
        # Create a mapping of chunk_id to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_map[result.chunk_id] = {
                'result': result,
                'semantic_score': result.similarity_score,
                'keyword_score': 0.0
            }
        
        # Add keyword scores
        for result in keyword_results:
            if result.chunk_id in result_map:
                result_map[result.chunk_id]['keyword_score'] = result.similarity_score
        
        # Calculate combined scores
        combined_results = []
        for chunk_id, data in result_map.items():
            semantic_score = data['semantic_score']
            keyword_score = data['keyword_score']
            
            # Weighted combination
            combined_score = (
                embedding_config.semantic_weight * semantic_score +
                embedding_config.keyword_weight * keyword_score
            )
            
            # Update the result with combined score
            result = data['result']
            result.similarity_score = combined_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return combined_results
    
    def get_document_stats(self, document_id: str = None) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            if document_id:
                # Stats for specific document
                results = self.collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
                
                if not results["ids"]:
                    return {"error": "Document not found"}
                
                metadatas = results["metadatas"]
                
                return {
                    "document_id": document_id,
                    "chunk_count": len(results["ids"]),
                    "total_characters": sum(m.get("character_count", 0) for m in metadatas),
                    "total_words": sum(m.get("word_count", 0) for m in metadatas),
                    "page_count": metadatas[0].get("doc_total_pages", 0) if metadatas else 0,
                    "has_tables": metadatas[0].get("doc_has_tables", False) if metadatas else False,
                    "created_at": min(m.get("created_at", 0) for m in metadatas) if metadatas else 0
                }
            else:
                # Overall stats
                collection_count = self.collection.count()
                
                if collection_count == 0:
                    return {"total_chunks": 0, "total_documents": 0}
                
                # Get all metadata to calculate stats
                all_results = self.collection.get(include=["metadatas"])
                metadatas = all_results["metadatas"]
                
                # Calculate unique documents
                unique_docs = set(m.get("document_id") for m in metadatas)
                
                return {
                    "total_chunks": collection_count,
                    "total_documents": len(unique_docs),
                    "total_characters": sum(m.get("character_count", 0) for m in metadatas),
                    "total_words": sum(m.get("word_count", 0) for m in metadatas),
                    "documents": list(unique_docs)
                }
                
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {"error": str(e)}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for intelligent search"}
            )
            logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from .document_processor import DocumentContent
    
    async def test_embedding_manager():
        # Initialize embedding manager
        manager = EmbeddingManager("test_collection")
        
        # Create sample document content
        sample_content = DocumentContent(
            text="This is a sample document about artificial intelligence and machine learning.",
            tables=[],
            key_value_pairs=[],
            paragraphs=["This is a sample document about artificial intelligence and machine learning."],
            pages=[{"page_number": 1}],
            confidence_scores={"overall": 0.9},
            metadata={"filename": "test.pdf", "total_pages": 1}
        )
        
        # Test embedding creation
        result = await manager.create_embeddings(sample_content, "test_doc_1", "test.pdf")
        print(f"Embedding creation successful: {result.success}")
        print(f"Created {result.embedding_count} embeddings")
        
        # Test search
        search_results = await manager.search_similar_content("artificial intelligence")
        print(f"Found {len(search_results)} search results")
        
        for result in search_results:
            print(f"  - Score: {result.similarity_score:.3f}, Content: {result.content[:100]}...")
        
        # Test stats
        stats = manager.get_document_stats()
        print(f"Collection stats: {stats}")
    
    # Run test
    # asyncio.run(test_embedding_manager())