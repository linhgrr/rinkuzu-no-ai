"""
Gemini AI Service Implementation
"""
import base64
from typing import List, Optional, Union, Dict, Any
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from loguru import logger

from app.core.interfaces.ai_service import AIService, AIResponse, ChatMessage
from app.utils.key_rotation import KeyRotationManager
from app.core.config import settings


class GeminiService(AIService):
    """
    Google Gemini AI service implementation
    
    Features:
    - Multi-key rotation for load balancing
    - Automatic retry with different keys
    - Support for text and multimodal inputs
    - Safety settings configuration
    """
    
    def __init__(self, api_keys: Optional[List[str]] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini service
        
        Args:
            api_keys: List of Gemini API keys. If None, uses from settings
        """
        self.api_keys = api_keys.split(',')
        logger.info(f"API Keys: {self.api_keys}")

        if not self.api_keys:
            raise ValueError("No Gemini API keys provided")
        
        # Initialize key rotation manager
        self.key_manager = KeyRotationManager(
            keys=self.api_keys,
            max_errors_per_key=3,
            block_duration_minutes=5
        )
        
        # Model configuration
        self.model_name = model_name
        self.fallback_model = "gemini-2.0-flash"
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Rin-chan fallback system prompt – used when RAG cannot provide context
        self.fallback_system_prompt = """
Bạn là Rin-chan, một trợ lý AI thông minh và thân thiện. 
Bạn vừa báo rằng không tìm thấy thông tin trong tài liệu môn học.
Bây giờ hãy trả lời câu hỏi bằng kiến thức vốn có của bạn.

Quy tắc trả lời:
1. Luôn thừa nhận rằng thông tin này không có trong tài liệu môn học mà bạn biết
2. Trả lời bằng kiến thức chung một cách chi tiết và hữu ích
3. Giữ tone thân thiện, nhiệt tình như Rin-chan
4. Sử dụng emoji phù hợp để tạo cảm giác gần gũi

Ví dụ cách bắt đầu câu trả lời:
"Mặc dù Rin-chan không tìm thấy thông tin này trong tài liệu môn học này mà Rin-chan có, nhưng Rin-chan có thể giải thích dựa trên kiến thức chung..."
"""
        
        logger.info(f"Initialized GeminiService with {len(self.api_keys)} API keys")
    
    async def generate_text(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """
        Generate text response from prompt
        
        Args:
            prompt: The main prompt/question
            context: Optional context for the prompt
            model_name: Optional model override
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with generated text or error
        """
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt
        
        return await self._generate_with_retry(
            prompt=full_prompt,
            model_name=model_name or self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def chat(
        self, 
        messages: List[ChatMessage],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AIResponse:
        """
        Handle multi-turn chat conversation
        
        Args:
            messages: List of chat messages
            model_name: Optional model override
            temperature: Generation temperature
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with chat reply
        """
        # Convert messages to Gemini format
        chat_history = []
        current_prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                # Add system message as context
                current_prompt = f"System: {msg.content}\n\n" + current_prompt
            elif msg.role == "user":
                current_prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                current_prompt += f"Assistant: {msg.content}\n"
        
        # Add final user prompt
        current_prompt += "Assistant: "
        
        return await self._generate_with_retry(
            prompt=current_prompt,
            model_name=model_name or self.model_name,
            temperature=temperature,
            **kwargs
        )
    
    async def process_with_image(
        self,
        prompt: str,
        image_data: Union[bytes, str],
        model_name: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """
        Process prompt with image data
        
        Args:
            prompt: Text prompt
            image_data: Image as bytes or base64 string
            model_name: Optional model override
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with processed result
        """
        # Convert image to base64 if needed
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_b64 = image_data
        
        return await self._generate_with_retry(
            prompt=prompt,
            image_data=image_b64,
            model_name=model_name or self.model_name,
            **kwargs
        )
    
    def get_service_name(self) -> str:
        """Return the name of the AI service"""
        return "Google Gemini"
    
    def is_available(self) -> bool:
        """Check if the service is available/configured"""
        return len(self.api_keys) > 0
    
    async def _generate_with_retry(
        self,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        image_data: Optional[str] = None,
        content_parts: Optional[List[Any]] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """
        Generate response with automatic retry using different API keys
        
        Args:
            prompt: The prompt to send
            model_name: Model to use
            image_data: Optional base64 image data
            max_retries: Maximum retry attempts
            **kwargs: Additional model parameters
            
        Returns:
            AIResponse with result or error
        """
        max_retries = max_retries or min(len(self.api_keys), 3)
        last_error = None
        
        for attempt in range(max_retries):
            api_key = await self.key_manager.get_next_key()
            
            if not api_key:
                return AIResponse(
                    success=False,
                    error="All API keys are currently blocked. Please try again later."
                )
            
            try:
                # Configure API key
                configure(api_key=api_key)
                
                # Create model
                model = GenerativeModel(
                    model_name=model_name,
                    safety_settings=self.safety_settings
                )
                
                # Prepare content
                if content_parts is not None:
                    # Pre-formatted parts (can include text & images)
                    content = content_parts
                elif image_data:
                    # Single image with prompt
                    content = [
                        {"text": prompt or ""},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",  # Assume JPEG for now
                                "data": image_data
                            }
                        }
                    ]
                else:
                    # Text-only request
                    content = prompt or ""
                
                # Generate response
                response = await model.generate_content_async(
                    content,
                    generation_config={
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_output_tokens": kwargs.get("max_tokens"),
                        "top_p": kwargs.get("top_p", 0.95),
                        "top_k": kwargs.get("top_k", 40),
                    }
                )
                
                # Extract text from response
                result_text = response.text
                
                # Report success
                await self.key_manager.report_success(api_key)
                
                logger.info(f"✅ Gemini request successful with key {api_key[:8]}...")
                
                return AIResponse(
                    success=True,
                    content=result_text,
                    metadata={
                        "model": model_name,
                        "api_key_prefix": api_key[:8] + "...",
                        "attempt": attempt + 1
                    }
                )
                
            except Exception as error:
                last_error = error
                logger.warning(f"❌ Gemini request failed with key {api_key[:8]}...: {error}")
                
                # Report error to key manager
                await self.key_manager.report_error(api_key, error)
                
                # Check if we should retry with fallback model
                if "model" in str(error).lower() and model_name != self.fallback_model:
                    logger.info(f"🔄 Retrying with fallback model: {self.fallback_model}")
                    return await self._generate_with_retry(
                        prompt=prompt,
                        model_name=self.fallback_model,
                        image_data=image_data,
                        content_parts=content_parts,
                        max_retries=max_retries - attempt,
                        **kwargs
                    )
                
                # Continue to next key for other errors
                continue
        
        # All retries failed
        error_msg = f"Failed after {max_retries} attempts. Last error: {last_error}"
        logger.error(f"❌ Gemini service failed: {error_msg}")
        
        return AIResponse(
            success=False,
            error=error_msg,
            metadata={
                "attempts": max_retries,
                "last_error": str(last_error)
            }
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics
        
        Returns:
            Dictionary with service statistics
        """
        key_stats = await self.key_manager.get_key_statistics()
        
        return {
            "service_name": self.get_service_name(),
            "total_keys": len(self.api_keys),
            "available_keys": len([k for k, s in key_stats.items() if not s["is_blocked"]]),
            "key_statistics": key_stats,
            "model_name": self.model_name,
            "fallback_model": self.fallback_model
        } 

    # ------------------------------------------------------------------
    # Image helpers & fallback response

    def _base64_to_image_part(self, base64_data: str) -> Optional[Dict[str, Any]]:
        """Convert base64 string or URL to Gemini inline image part."""
        try:
            if base64_data.startswith("http://") or base64_data.startswith("https://"):
                import requests, mimetypes
                resp = requests.get(base64_data, timeout=10)
                if resp.status_code != 200:
                    raise ValueError(f"HTTP {resp.status_code}")
                mime_type = resp.headers.get("Content-Type") or mimetypes.guess_type(base64_data)[0] or "image/jpeg"
                image_bytes = resp.content
            else:
                # Remove data URL prefix if present
                if base64_data.startswith("data:image/"):
                    base64_data = base64_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                mime_type = "image/jpeg"

            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_bytes
                }
            }
        except Exception as e:
            logger.error(f"Failed to convert image data: {e}")
            return None

    async def generate_fallback_response(
        self,
        question: str,
        subject_id: str,
        question_image: Optional[str] = None,
        option_images: Optional[List[Optional[str]]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Fallback answer when no RAG context is found."""
        try:
            logger.info(f"🤖 Generating fallback response for subject {subject_id}")

            # Build prompt with personality & disclaimer
            prompt = f"{self.fallback_system_prompt}\n\nCâu hỏi thuộc môn: {subject_id}\n"

            # Append recent chat history (max 3 turns)
            if chat_history:
                prompt += "\nLịch sử trò chuyện:\n"
                for msg in chat_history[-3:]:
                    prompt += f"{msg.role}: {msg.content}\n"
                prompt += "\n"

            prompt += f"Câu hỏi hiện tại: {question}"

            content_parts: List[Any] = [prompt]

            # Attach images if provided
            if question_image:
                img_part = self._base64_to_image_part(question_image)
                if img_part:
                    content_parts.append(img_part)
                    content_parts.append("Hình ảnh trên là câu hỏi. Hãy phân tích và trả lời dựa trên nội dung hình ảnh.")

            if option_images:
                for idx, opt_img in enumerate(option_images):
                    if opt_img:
                        img_part = self._base64_to_image_part(opt_img)
                        if img_part:
                            content_parts.append(img_part)
                            content_parts.append(f"Đây là hình ảnh đáp án {chr(65 + idx)}.")

            # Use provided or fallback model
            selected_model = model_name or self.fallback_model

            ai_resp = await self._generate_with_retry(
                prompt="",  # not used when content_parts provided
                model_name=selected_model,
                content_parts=content_parts,
                **kwargs
            )

            # Enrich metadata
            ai_resp.metadata = ai_resp.metadata or {}
            ai_resp.metadata.update({
                "source": "gemini_fallback",
                "subject_id": subject_id,
                "has_images": bool(question_image or (option_images and any(option_images)))
            })

            return ai_resp

        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            return AIResponse(
                success=False,
                error=str(e),
                metadata={"source": "gemini_fallback", "subject_id": subject_id}
            )