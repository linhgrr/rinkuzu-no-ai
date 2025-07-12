"""
Gemini AI Service Implementation
"""
import base64
import mimetypes
from typing import List, Optional, Union, Dict, Any
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.client_options import ClientOptions
import httpx
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
            api_keys: List of Gemini API keys.
        """
        self.api_keys = api_keys        
        # Initialize key rotation manager
        self.key_manager = KeyRotationManager(
            keys=self.api_keys,
            max_errors_per_key=3,
            block_duration_minutes=5
        )
        
        # Model configuration
        self.model_name = model_name
        self.fallback_model = "gemini-2.5-flash"

        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
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
        images: Optional[List[Union[bytes, str]]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AIResponse:
        """Multi-turn chat with optional images.

        Args:
            messages: Conversation history (system / user / assistant).
            images: List of base64 strings, URLs, or raw bytes representing images to include.
            model_name: Override model.
            temperature: Sampling temperature.
        """
        # Build plain-text prompt from messages
        current_prompt = ""

        for msg in messages:
            if msg.role == "system":
                current_prompt = f"System: {msg.content}\n\n" + current_prompt
            elif msg.role == "user":
                current_prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                current_prompt += f"Assistant: {msg.content}\n"

        # Add assistant cue
        current_prompt += "Assistant: "

        # Prepare images if supplied
        content_parts: Optional[List[Any]] = None

        if images:
            content_parts = [current_prompt]

            for img in images:
                # Convert bytes → base64 if necessary
                if isinstance(img, (bytes, bytearray)):
                    img = base64.b64encode(img).decode("utf-8")

                img_part = await self._base64_to_image_part(img) if isinstance(img, str) else None
                if img_part:
                    content_parts.append(img_part)

        # Choose call path
        if content_parts is not None:
            return await self._generate_with_retry(
                prompt="", 
                model_name=model_name or self.model_name,
                content_parts=content_parts,
                temperature=temperature,
                **kwargs,
            )
        else:
            return await self._generate_with_retry(
                prompt=current_prompt,
                model_name=model_name or self.model_name,
                temperature=temperature,
                **kwargs,
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
                genai.configure(api_key=api_key)

                # Khởi tạo model mà không có client_options
                model = GenerativeModel(
                    model_name=model_name,
                    safety_settings=self.safety_settings,
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

    async def _base64_to_image_part(self, base64_data: str) -> Optional[Dict[str, Any]]:
        """Convert base64 string or URL to Gemini inline image part."""
        try:
            if base64_data.startswith("http://") or base64_data.startswith("https://"):
                # Use async HTTP client instead of blocking requests
                async with httpx.AsyncClient() as client:
                    resp = await client.get(base64_data, timeout=10)
                    resp.raise_for_status()  # Automatically raise error for non-2xx status codes
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