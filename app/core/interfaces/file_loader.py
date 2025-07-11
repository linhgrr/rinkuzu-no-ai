"""
Abstract interface for file loaders
"""
from abc import ABC, abstractmethod
from typing import List, BinaryIO, Optional, Dict, Any
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """Represents a chunk of processed document"""
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None


class FileLoader(ABC):
    """
    Abstract base class for file loaders
    
    Each file type (PDF, DOCX, images) should implement this interface
    """
    
    @abstractmethod
    async def load(self, file_stream: BinaryIO, filename: str) -> List[DocumentChunk]:
        """
        Load and process file content into chunks
        
        Args:
            file_stream: File binary stream
            filename: Original filename for metadata
            
        Returns:
            List of document chunks with content and metadata
        """
        pass
    
    @abstractmethod
    def supports_file_type(self, file_extension: str) -> bool:
        """
        Check if this loader supports the given file extension
        
        Args:
            file_extension: File extension (e.g., 'pdf', 'docx')
            
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions
        
        Returns:
            List of supported extensions
        """
        pass 