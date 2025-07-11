"""
General AI API routes for text generation and chat
"""
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.models.requests import GenerateTextRequest, ChatRequest
from app.models.responses import GenerateTextResponse, ChatResponse
from app.services.ai.gemini_service import GeminiService
from app.utils.dependencies import get_gemini_service, handle_service_errors
from app.core.interfaces.ai_service import ChatMessage

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post("/generate-text", response_model=GenerateTextResponse)
@handle_service_errors
async def generate_text(
    request: GenerateTextRequest,
    ai_service: GeminiService = Depends(get_gemini_service)
):
    """
    Generate text using AI model
    
    General text generation endpoint that can be used for various tasks:
    - Creative writing
    - Text completion
    - Content generation
    - Question answering without RAG
    
    - **prompt**: The input prompt for text generation
    - **context**: Optional context to provide additional information
    - **temperature**: Controls randomness (0.0 = deterministic, 2.0 = very random)
    - **max_tokens**: Maximum number of tokens to generate
    """
    logger.info(f"🤖 Text generation request: {request.prompt[:100]}...")
    
    # Generate text
    response = await ai_service.generate_text(
        prompt=request.prompt,
        context=request.context,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    if response.success:
        return GenerateTextResponse(
            success=True,
            content=response.content,
            metadata=response.metadata
        )
    else:
        return GenerateTextResponse(
            success=False,
            error=response.error,
            metadata=response.metadata
        )


@router.post("/chat", response_model=ChatResponse)
@handle_service_errors
async def chat(
    request: ChatRequest,
    ai_service: GeminiService = Depends(get_gemini_service)
):
    """
    Multi-turn chat conversation with AI
    
    Enables conversational interactions with the AI model:
    - Multi-turn conversations
    - Context awareness
    - Role-based interactions (system, user, assistant)
    
    - **messages**: List of conversation messages with roles
    - **temperature**: Controls response randomness
    
    The AI will respond as an assistant, maintaining conversation context.
    """
    logger.info(f"💬 Chat request with {len(request.messages)} messages")
    
    # Convert request messages to internal format
    chat_messages = [
        ChatMessage(role=msg.role, content=msg.content)
        for msg in request.messages
    ]
    
    # Generate chat response
    response = await ai_service.chat(
        messages=chat_messages,
        temperature=request.temperature
    )
    
    if response.success:
        return ChatResponse(
            success=True,
            content=response.content,
            metadata=response.metadata
        )
    else:
        return ChatResponse(
            success=False,
            error=response.error,
            metadata=response.metadata
        )


@router.get("/models", response_model=dict)
async def list_available_models(
    ai_service: GeminiService = Depends(get_gemini_service)
):
    """
    List available AI models and their capabilities
    
    Returns information about:
    - Available models
    - Model capabilities
    - Current configuration
    """
    try:
        stats = await ai_service.get_statistics()
        
        return {
            "success": True,
            "models": {
                "primary": stats.get("model_name"),
                "fallback": stats.get("fallback_model"),
                "service": stats.get("service_name")
            },
            "capabilities": {
                "text_generation": True,
                "chat": True,
                "multimodal": True,
                "languages": ["Vietnamese", "English", "Multiple"]
            },
            "configuration": {
                "total_keys": stats.get("total_keys"),
                "available_keys": stats.get("available_keys")
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.get("/statistics", response_model=dict)
async def get_ai_statistics(
    ai_service: GeminiService = Depends(get_gemini_service)
):
    """
    Get AI service statistics and usage information
    
    Returns detailed statistics about:
    - API key usage and rotation
    - Request success/failure rates
    - Model performance metrics
    """
    try:
        stats = await ai_service.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve AI statistics"
        ) 