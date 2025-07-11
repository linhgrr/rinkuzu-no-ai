"""
PDF file loader for RAG pipeline using PyMuPDF
"""
import io
from typing import List, BinaryIO, Dict, Any
import fitz  # PyMuPDF
from loguru import logger

from app.core.interfaces.file_loader import FileLoader, DocumentChunk


class PDFLoader(FileLoader):
    """
    PDF file loader using PyMuPDF for enhanced text extraction
    
    Features:
    - Advanced text extraction with layout preservation
    - Better handling of complex layouts and tables
    - Image and structure detection
    - Metadata preservation (page numbers, filename)
    - Chunking support for large documents
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize PDF loader
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = ["pdf"]
    
    async def load(self, file_stream: BinaryIO, filename: str) -> List[DocumentChunk]:
        """
        Load and process PDF file into chunks
        
        Args:
            file_stream: PDF file binary stream
            filename: Original filename for metadata
            
        Returns:
            List of document chunks with content and metadata
        """
        try:
            # Convert stream to bytes for PyMuPDF
            if hasattr(file_stream, 'read'):
                pdf_bytes = file_stream.read()
            else:
                pdf_bytes = file_stream
            
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_doc)
            
            logger.info(f"📄 Processing PDF with PyMuPDF: {filename} ({total_pages} pages)")
            
            all_chunks = []
            full_text = ""
            
            # Extract text from each page
            for page_num in range(total_pages):
                try:
                    page = pdf_doc[page_num]
                    page_text = page.get_text()
                    
                    # Get enhanced metadata
                    page_metadata = {
                        "filename": filename,
                        "page_number": page_num + 1,
                        "total_pages": total_pages,
                        "source_type": "pdf",
                        "chunk_type": "page",
                        "page_width": page.rect.width,
                        "page_height": page.rect.height,
                        "rotation": page.rotation
                    }
                    
                    # Detect images on page
                    try:
                        image_list = page.get_images()
                        page_metadata["image_count"] = len(image_list)
                    except:
                        page_metadata["image_count"] = 0
                    
                    # Detect potential tables/structures
                    try:
                        blocks = page.get_text("dict")
                        table_indicators = 0
                        for block in blocks.get("blocks", []):
                            if "lines" in block and len(block["lines"]) > 2:
                                table_indicators += 1
                        page_metadata["structure_complexity"] = table_indicators
                    except:
                        page_metadata["structure_complexity"] = 0
                    
                    if page_text.strip():
                        # Add page marker
                        page_content = f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        full_text += page_content
                        
                        # Create page-level chunk if it's small enough
                        if len(page_text) <= self.chunk_size:
                            chunk = DocumentChunk(
                                content=page_text.strip(),
                                metadata=page_metadata,
                                page_number=page_num + 1,
                                chunk_index=len(all_chunks)
                            )
                            all_chunks.append(chunk)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} of {filename}: {e}")
                    continue
            
            # Close the PDF document
            pdf_doc.close()
            
            # If we have large pages, create overlapping chunks
            if len(full_text) > self.chunk_size:
                text_chunks = self._create_text_chunks(full_text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        content=chunk_text.strip(),
                        metadata={
                            "filename": filename,
                            "total_pages": total_pages,
                            "source_type": "pdf",
                            "chunk_type": "text_chunk",
                            "chunk_size": len(chunk_text)
                        },
                        chunk_index=len(all_chunks) + i
                    )
                    all_chunks.append(chunk)
            
            logger.info(f"✅ PDF processed: {filename} -> {len(all_chunks)} chunks")
            return all_chunks
            
        except ImportError:
            logger.error("PyMuPDF library not installed. Install with: pip install PyMuPDF")
            raise ValueError("PDF processing library (PyMuPDF) not available")
        except Exception as e:
            logger.error(f"❌ Failed to process PDF {filename}: {e}")
            raise ValueError(f"Failed to process PDF file: {e}")
    
    def supports_file_type(self, file_extension: str) -> bool:
        """Check if this loader supports the given file extension"""
        return file_extension.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.supported_extensions.copy()
    
    def _create_text_chunks(self, text: str) -> List[str]:
        """
        Create overlapping text chunks from large text
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks with overlap
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space within chunk
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + self.chunk_size // 2:  # Don't break too early
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks 