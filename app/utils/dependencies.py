"""
FastAPI dependency injection utilities

This module provides dependency injection for all services,
ensuring singleton instances and proper initialization.
"""
import os
from functools import lru_cache
from typing import Dict, Any, List
from fastapi import HTTPException, status, UploadFile, Depends

from app.core.config import settings
from app.core.interfaces.ai_service import AIService
from app.services.ai.gemini_service import GeminiService
from app.services.ai.rag_tutor_service import RagTutorService
from app.services.vector.embedding_service import EmbeddingService
from app.services.vector.faiss_vector_store import FAISSVectorStore


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service"""
    return EmbeddingService(
        embedding_model_name=settings.embedding_model,
        reranker_model_name=settings.reranker_model
    )


@lru_cache()
def get_vector_store() -> FAISSVectorStore:
    """Get singleton FAISS vector store - much more stable than ChromaDB"""
    return FAISSVectorStore(
        persist_directory=settings.vector_db_path
    )


@lru_cache()
def get_gemini_service() -> GeminiService:
    """Get singleton Gemini AI service"""
    if not settings.gemini_keys or len(settings.gemini_keys) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable: No API keys configured"
        )
    
    api_keys = settings.gemini_keys.split(',') if settings.gemini_keys else []
    return GeminiService(
        api_keys=api_keys,
        model_name="gemini-2.0-flash",
    )


@lru_cache()
def get_rag_tutor_service() -> RagTutorService:
    """Get singleton RAG tutor service """
    return RagTutorService(
        ai_service=get_gemini_service(),
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_retrieval_docs=settings.max_retrieval_docs,
        reranker_top_k=settings.reranker_top_k,
        min_similarity=0.4 
    )


# ============================================================================
# VALIDATION DEPENDENCIES
# ============================================================================

async def validate_upload_file(file: UploadFile) -> UploadFile:
    """
    Validate uploaded file size and type
    
    Args:
        file: The uploaded file
        
    Returns:
        The validated file
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size / 1024 / 1024:.1f}MB"
        )
    
    # Check file type
    if file.filename:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_types)}"
            )
    
    return file


async def validate_subject_id(subject_id: str) -> str:
    """
    Validate and sanitize subject ID
    
    Args:
        subject_id: Subject identifier
        
    Returns:
        Sanitized subject ID
        
    Raises:
        HTTPException: If subject ID is invalid
    """
    # Basic validation
    if not subject_id or len(subject_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subject ID is required"
        )
    
    # Sanitize
    subject_id = subject_id.strip().lower()
    
    # Check format (alphanumeric + underscores/hyphens)
    import re
    if not re.match(r'^[a-z0-9_-]+$', subject_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subject ID must contain only letters, numbers, underscores, and hyphens"
        )
    
    # Length check
    if len(subject_id) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subject ID too long (max 50 characters)"
        )
    
    return subject_id


async def validate_api_key(api_key: str = None) -> bool:
    """
    Validate API key if authentication is enabled
    
    Args:
        api_key: Optional API key from request
        
    Returns:
        True if valid or authentication disabled
        
    Raises:
        HTTPException: If authentication fails
    """
    # Skip validation if no API key is configured
    if not settings.api_key:
        return True
    
    # Check if API key provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate API key
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True


# ============================================================================
# SERVICE HEALTH CHECKS
# ============================================================================

async def check_services_health() -> Dict[str, Dict[str, Any]]:
    """
    Check health of all services
    
    Returns:
        Dictionary with health status of all services
    """
    health_status = {}
    
    try:
        # Check embedding service
        embedding_service = get_embedding_service()
        health_status["embedding_service"] = {
            "status": "healthy",
            "model": embedding_service.embedding_model_name,
            "reranker": embedding_service.reranker_model_name
        }
    except Exception as e:
        health_status["embedding_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    try:
        # Check vector store
        vector_store = get_vector_store()
        collections = vector_store.list_collections()
        health_status["vector_store"] = {
            "status": "healthy",
            "collections": len(collections),
            "path": settings.vector_db_path
        }
    except Exception as e:
        health_status["vector_store"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    try:
        # Check AI service
        ai_service = get_gemini_service()
        stats = await ai_service.get_statistics()
        health_status["ai_service"] = {
            "status": "healthy",
            "name": stats.get("service_name"),
            "available_keys": stats.get("available_keys"),
            "total_keys": stats.get("total_keys")
        }
    except Exception as e:
        health_status["ai_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_status


# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_service_errors(func):
    """
    Decorator to handle common service errors
    
    Converts service exceptions to appropriate HTTP responses
    """
    import functools
    from loguru import logger
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except FileNotFoundError as e:
            logger.error(f"File not found in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Requested resource not found"
            )
        except PermissionError as e:
            logger.error(f"Permission error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        except ValueError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )
    
    return wrapper


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(path: str) -> str:
    """
    Ensure directory exists, create if needed
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def clear_service_cache():
    """
    Clear all cached service instances
    
    Useful for testing or configuration reloading
    """
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()
    get_gemini_service.cache_clear()
    get_rag_tutor_service.cache_clear() 