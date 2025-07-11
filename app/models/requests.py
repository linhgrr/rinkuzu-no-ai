"""
Pydantic models for API requests
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadDocumentRequest(BaseModel):
    """Request model for document upload"""
    subject_id: str = Field(..., description="Subject/course identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AskQuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    subject_id: str = Field(..., description="Subject identifier for context")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation history")
    use_reranking: bool = Field(True, description="Whether to use reranking for better results")
    # Add image support
    question_image: Optional[str] = Field(None, description="Base64 encoded question image")
    option_images: Optional[List[Optional[str]]] = Field(None, description="Base64 encoded option images")
    force_fallback: bool = Field(False, description="Force fallback to Gemini even if context is found")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Máy tính làm việc như thế nào?",
                "subject_id": "computer_science_101",
                "chat_history": [
                    {"role": "user", "content": "CPU là gì?"},
                    {"role": "assistant", "content": "CPU là bộ xử lý trung tâm..."}
                ],
                "use_reranking": True,
                "question_image": None,
                "option_images": None,
                "force_fallback": False
            }
        }
    }


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class GenerateTextRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., min_length=1, max_length=5000, description="Text prompt")
    context: Optional[str] = Field(None, description="Optional context")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens to generate")


class ChatRequest(BaseModel):
    """Request model for chat conversations"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7
            }
        }
    } 