"""
Pydantic models for API responses
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")


class UploadDocumentResponse(BaseResponse):
    """Response model for document upload"""
    statistics: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed successfully",
                "statistics": {
                    "filename": "lecture_notes.pdf",
                    "subject_id": "computer_science_101",
                    "chunks_created": 15,
                    "chunks_indexed": 15,
                    "collection": "subject_computer_science_101"
                }
            }
        }


class AskQuestionResponse(BaseResponse):
    """Response model for question answering"""
    answer: Optional[str] = Field(None, description="AI-generated answer")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "answer": "Máy tính làm việc bằng cách xử lý thông tin qua CPU, bộ nhớ và các thành phần khác...",
                "metadata": {
                    "context_found": True,
                    "subject_id": "computer_science_101",
                    "search_results_count": 5,
                    "reranked_chunks": 3,
                    "context_sources": ["lecture_notes.pdf", "textbook.pdf"]
                }
            }
        }


class GenerateTextResponse(BaseResponse):
    """Response model for text generation"""
    content: Optional[str] = Field(None, description="Generated text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")


class ChatResponse(BaseResponse):
    """Response model for chat"""
    content: Optional[str] = Field(None, description="AI chat response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chat metadata")


class StatisticsResponse(BaseResponse):
    """Response model for statistics"""
    statistics: Optional[Dict[str, Any]] = Field(None, description="Service statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "statistics": {
                    "collection_name": "subject_computer_science_101",
                    "document_count": 150,
                    "metadata_keys": ["filename", "subject_id", "chunk_index"],
                    "embedding_model": "Alibaba-NLP/gte-multilingual-base",
                    "ai_service": "Google Gemini"
                }
            }
        }


class HealthResponse(BaseResponse):
    """Response model for health check"""
    services: Optional[Dict[str, Any]] = Field(None, description="Service health status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "services": {
                    "ai_service": {
                        "available": True,
                        "name": "Google Gemini"
                    },
                    "embedding_service": {
                        "cache_size": 50,
                        "embedding_model": "Alibaba-NLP/gte-multilingual-base"
                    },
                    "vector_store": {
                        "collections": ["subject_math_101", "subject_physics_101"]
                    }
                }
            }
        }


class ListResponse(BaseResponse):
    """Response model for list operations"""
    items: List[Any] = Field(default_factory=list, description="List of items")
    total: int = Field(0, description="Total number of items")


class SubjectListResponse(ListResponse):
    """Response model for listing subjects"""
    subjects: List[str] = Field(default_factory=list, description="List of subject IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "subjects": ["computer_science_101", "math_101", "physics_101"],
                "total": 3
            }
        } 