"""
File loader factory for automatic loader selection
"""
from typing import Dict, Type, Optional
from app.core.interfaces.file_loader import FileLoader
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .image_loader import ImageLoader
from .pptx_loader import PPTXLoader


class FileLoaderFactory:
    """
    Factory for creating appropriate file loaders based on file extension
    
    Automatically selects the right loader for different file types.
    Supports: PDF (PyMuPDF), DOCX, PPTX, Images (Gemini Vision)
    """
    
    def __init__(self, ai_service=None):
        """
        Initialize factory with available loaders
        
        Args:
            ai_service: AI service for image processing (optional)
        """
        self._loaders: Dict[str, Type[FileLoader]] = {}
        self.ai_service = ai_service
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default file loaders"""
        # PDF loader using PyMuPDF
        pdf_loader = PDFLoader
        for ext in PDFLoader().get_supported_extensions():
            self._loaders[ext.lower()] = pdf_loader
        
        # DOCX loader using python-docx
        docx_loader = DOCXLoader
        for ext in DOCXLoader().get_supported_extensions():
            self._loaders[ext.lower()] = docx_loader
        
        # PPTX loader using python-pptx
        pptx_loader = PPTXLoader
        for ext in PPTXLoader().get_supported_extensions():
            self._loaders[ext.lower()] = pptx_loader
        
        # Image loader using Gemini Vision
        image_loader = ImageLoader
        for ext in ImageLoader().get_supported_extensions():
            self._loaders[ext.lower()] = image_loader
    
    def register_loader(self, file_extension: str, loader_class: Type[FileLoader]):
        """
        Register a custom file loader
        
        Args:
            file_extension: File extension (e.g., 'txt', 'md')
            loader_class: FileLoader class to handle this extension
        """
        self._loaders[file_extension.lower()] = loader_class
    
    def get_loader(self, filename: str, **kwargs) -> FileLoader:
        """
        Get appropriate loader for a file
        
        Args:
            filename: Name of the file
            **kwargs: Additional arguments for loader initialization
            
        Returns:
            FileLoader instance for the file type
            
        Raises:
            ValueError: If no loader found for file type
        """
        file_extension = self._extract_extension(filename)
        
        if file_extension not in self._loaders:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {list(self._loaders.keys())}"
            )
        
        loader_class = self._loaders[file_extension]
        
        # Special handling for ImageLoader that needs AI service
        if loader_class == ImageLoader and self.ai_service:
            return loader_class(ai_service=self.ai_service)
        
        return loader_class(**kwargs)
    
    def supports_file(self, filename: str) -> bool:
        """
        Check if a file type is supported
        
        Args:
            filename: Name of the file
            
        Returns:
            True if file type is supported
        """
        file_extension = self._extract_extension(filename)
        return file_extension in self._loaders
    
    def get_supported_extensions(self) -> list[str]:
        """
        Get list of all supported file extensions
        
        Returns:
            List of supported extensions
        """
        return list(self._loaders.keys())
    
    def _extract_extension(self, filename: str) -> str:
        """
        Extract file extension from filename
        
        Args:
            filename: Name of the file
            
        Returns:
            File extension in lowercase
        """
        if '.' not in filename:
            return ""
        
        return filename.rsplit('.', 1)[-1].lower()


# Global factory instance (will be updated with AI service)
file_loader_factory = FileLoaderFactory()

def set_ai_service_for_factory(ai_service):
    """Set AI service for the global factory"""
    global file_loader_factory
    file_loader_factory.ai_service = ai_service


def get_loader(filename: str, **kwargs) -> FileLoader:
    """
    Convenience function to get a file loader
    
    Args:
        filename: Name of the file
        **kwargs: Additional arguments for loader
        
    Returns:
        FileLoader instance
    """
    return file_loader_factory.get_loader(filename, **kwargs)


def supports_file(filename: str) -> bool:
    """
    Convenience function to check if file is supported
    
    Args:
        filename: Name of the file
        
    Returns:
        True if supported
    """
    return file_loader_factory.supports_file(filename) 