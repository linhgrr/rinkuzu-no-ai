"""
Abstract interface for vector stores
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel


class VectorDocument(BaseModel):
    """Document with optional embedding vector"""
    id: str
    content: str
    embedding: Optional[List[float]] = None  # Optional - ChromaDB can auto-generate
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    """Search result with similarity score"""
    document: VectorDocument
    score: float


class VectorStore(ABC):
    """
    Abstract base class for vector databases
    
    Supports different vector stores (ChromaDB, Pinecone, etc.)
    """
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[VectorDocument],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Add documents to the vector store (embeddings auto-generated if not provided)
        
        Args:
            documents: List of documents (embeddings optional)
            collection_name: Optional collection/namespace name
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query_text: Query text (will be automatically embedded)
            collection_name: Collection to search in
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        pass
    
    @abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete documents by IDs
        
        Args:
            document_ids: List of document IDs to delete
            collection_name: Collection name
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(
        self, 
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        pass 