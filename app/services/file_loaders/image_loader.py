"""
Image file loader using Gemini Vision API

Processes images using Google's Gemini multimodal capabilities to extract
text content, describe visual elements, and understand image context.
"""
import io
import base64
from typing import List, BinaryIO, Dict, Any, Optional
from PIL import Image
from loguru import logger

from app.core.interfaces.file_loader import FileLoader, DocumentChunk


class ImageLoader(FileLoader):
    """
    Image file loader using Gemini Vision for content extraction
    
    Features:
    - Text extraction from images (OCR)
    - Visual content description
    - Multi-language support
    - Educational content analysis
    - Chart/diagram interpretation
    """
    
    def __init__(self, ai_service=None):
        """
        Initialize image loader
        
        Args:
            ai_service: AI service instance for processing (will be injected)
        """
        self.ai_service = ai_service
        self.supported_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "webp"]
        
        # Vietnamese prompts for educational content analysis
        self.analysis_prompts = {
            "ocr": """
            Hãy trích xuất toàn bộ văn bản có trong hình ảnh này một cách chính xác nhất. 
            Bao gồm:
            - Tiêu đề, đề mục
            - Nội dung chính
            - Công thức toán học (nếu có)
            - Chú thích, ghi chú
            - Bất kỳ văn bản nào khác
            
            Giữ nguyên định dạng và cấu trúc của văn bản.
            """,
            
            "content_analysis": """
            Phân tích nội dung giáo dục trong hình ảnh này và mô tả:
            1. Chủ đề/môn học chính
            2. Các khái niệm quan trọng
            3. Biểu đồ, sơ đồ (nếu có)
            4. Công thức, phương trình (nếu có)
            5. Mối quan hệ giữa các yếu tố
            
            Trả lời bằng tiếng Việt.
            """,
            
            "diagram_analysis": """
            Nếu hình ảnh này chứa biểu đồ, sơ đồ, hoặc hình minh họa khoa học:
            1. Mô tả loại biểu đồ/sơ đồ
            2. Các thành phần chính
            3. Mối quan hệ được thể hiện
            4. Ý nghĩa giáo dục
            5. Dữ liệu hoặc thông tin số (nếu có)
            
            Trả lời bằng tiếng Việt.
            """
        }
    
    async def load(self, file_stream: BinaryIO, filename: str) -> List[DocumentChunk]:
        """
        Load and process image using Gemini Vision
        
        Args:
            file_stream: Image file stream
            filename: Name of the image file
            
        Returns:
            List of document chunks with extracted content
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            logger.info(f"🖼️ Processing image with Gemini Vision: {filename}")
            
            # Read and validate image
            if hasattr(file_stream, 'read'):
                image_bytes = file_stream.read()
            else:
                image_bytes = file_stream
            
            # Validate image format
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image_format = image.format
                image_size = image.size
                image_mode = image.mode
                
                logger.info(f"📸 Image info: {image_size[0]}x{image_size[1]} {image_format} {image_mode}")
            except Exception as e:
                raise ValueError(f"Invalid image format: {e}")
            
            # Check if AI service is available
            if not self.ai_service:
                logger.warning("AI service not available, creating basic image chunk")
                return self._create_basic_chunk(filename, image_bytes, image_format, image_size)
            
            chunks = []
            
            # 1. Extract text content (OCR)
            ocr_content = await self._extract_text_content(image_bytes, filename)
            if ocr_content:
                ocr_chunk = DocumentChunk(
                    content=ocr_content,
                    metadata={
                        "filename": filename,
                        "source_type": "image",
                        "chunk_type": "ocr_text",
                        "image_format": image_format,
                        "image_size": f"{image_size[0]}x{image_size[1]}",
                        "analysis_type": "text_extraction"
                    }
                )
                chunks.append(ocr_chunk)
            
            # 2. Analyze visual content
            content_analysis = await self._analyze_content(image_bytes, filename)
            if content_analysis:
                analysis_chunk = DocumentChunk(
                    content=content_analysis,
                    metadata={
                        "filename": filename,
                        "source_type": "image",
                        "chunk_type": "content_analysis",
                        "image_format": image_format,
                        "image_size": f"{image_size[0]}x{image_size[1]}",
                        "analysis_type": "visual_analysis"
                    }
                )
                chunks.append(analysis_chunk)
            
            # 3. Analyze diagrams/charts if present
            diagram_analysis = await self._analyze_diagrams(image_bytes, filename)
            if diagram_analysis:
                diagram_chunk = DocumentChunk(
                    content=diagram_analysis,
                    metadata={
                        "filename": filename,
                        "source_type": "image",
                        "chunk_type": "diagram_analysis",
                        "image_format": image_format,
                        "image_size": f"{image_size[0]}x{image_size[1]}",
                        "analysis_type": "diagram_analysis"
                    }
                )
                chunks.append(diagram_chunk)
            
            logger.info(f"✅ Image processed: {filename} -> {len(chunks)} content chunks")
            
            if not chunks:
                # Fallback: create basic chunk
                return self._create_basic_chunk(filename, image_bytes, image_format, image_size)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to process image {filename}: {e}")
            raise ValueError(f"Failed to process image file: {e}")
    
    async def _extract_text_content(self, image_bytes: bytes, filename: str) -> Optional[str]:
        """Extract text content using Gemini Vision OCR"""
        try:
            # Convert image to base64 for Gemini
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Use AI service to extract text
            response = await self.ai_service.process_with_image(
                prompt=self.analysis_prompts["ocr"],
                image_data=image_b64,
                image_format="auto"
            )
            
            if response.success and response.content.strip():
                logger.debug(f"📝 OCR extracted {len(response.content)} characters from {filename}")
                return response.content.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ OCR extraction failed for {filename}: {e}")
            return None
    
    async def _analyze_content(self, image_bytes: bytes, filename: str) -> Optional[str]:
        """Analyze visual content and educational elements"""
        try:
            # Convert image to base64 for Gemini
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Use AI service to analyze content
            response = await self.ai_service.process_with_image(
                prompt=self.analysis_prompts["content_analysis"],
                image_data=image_b64,
                image_format="auto"
            )
            
            if response.success and response.content.strip():
                logger.debug(f"🔍 Content analysis completed for {filename}")
                return response.content.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Content analysis failed for {filename}: {e}")
            return None
    
    async def _analyze_diagrams(self, image_bytes: bytes, filename: str) -> Optional[str]:
        """Analyze diagrams, charts, and scientific illustrations"""
        try:
            # Convert image to base64 for Gemini
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Use AI service to analyze diagrams
            response = await self.ai_service.process_with_image(
                prompt=self.analysis_prompts["diagram_analysis"],
                image_data=image_b64,
                image_format="auto"
            )
            
            if response.success and response.content.strip():
                # Only include if it contains meaningful diagram analysis
                content = response.content.strip()
                if any(keyword in content.lower() for keyword in 
                      ["biểu đồ", "sơ đồ", "chart", "diagram", "graph", "bảng", "công thức"]):
                    logger.debug(f"📊 Diagram analysis completed for {filename}")
                    return content
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Diagram analysis failed for {filename}: {e}")
            return None
    
    def _create_basic_chunk(self, filename: str, image_bytes: bytes, 
                          image_format: str, image_size: tuple) -> List[DocumentChunk]:
        """Create basic chunk when AI processing is not available"""
        basic_content = f"""
        Hình ảnh: {filename}
        Định dạng: {image_format}
        Kích thước: {image_size[0]}x{image_size[1]} pixels
        
        [Nội dung hình ảnh cần được phân tích bằng Gemini Vision]
        """
        
        chunk = DocumentChunk(
            content=basic_content.strip(),
            metadata={
                "filename": filename,
                "source_type": "image",
                "chunk_type": "basic_info",
                "image_format": image_format,
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "analysis_type": "metadata_only",
                "requires_ai_processing": True
            }
        )
        
        return [chunk]
    
    def supports_file_type(self, file_extension: str) -> bool:
        """Check if this loader supports the file type"""
        return file_extension.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.supported_extensions.copy()
    
    def set_ai_service(self, ai_service):
        """Set AI service for image processing"""
        self.ai_service = ai_service 