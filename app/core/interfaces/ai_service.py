"""
Abstract base interface for AI services
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str  # user, assistant, system
    content: str


class AIResponse(BaseModel):
    """Standard AI response format"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AIService(ABC):
    """
    Abstract base class for all AI services
    
    This interface ensures all AI services (Gemini, OpenAI, etc.) 
    have consistent methods and can be easily swapped
    """
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """
        Generate text response from prompt
        
        Args:
            prompt: The main prompt/question
            context: Optional context for the prompt
            **kwargs: Additional parameters specific to the service
            
        Returns:
            AIResponse with generated text or error
        """
        pass
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[ChatMessage],
        **kwargs
    ) -> AIResponse:
        """
        Handle multi-turn chat conversation
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with chat reply
        """
        pass
    
    @abstractmethod
    async def process_with_image(
        self,
        prompt: str,
        image_data: Union[bytes, str],
        **kwargs
    ) -> AIResponse:
        """
        Process prompt with image data
        
        Args:
            prompt: Text prompt
            image_data: Image as bytes or base64 string
            **kwargs: Additional parameters
            
        Returns:
            AIResponse with processed result
        """
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Return the name of the AI service"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available/configured"""
        pass 