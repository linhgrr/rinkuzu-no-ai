"""
Tutor API routes for RAG-based question answering
"""
from typing import List
from fastapi import APIRouter, Depends, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.models.requests import AskQuestionRequest
from app.models.responses import (
    UploadDocumentResponse, 
    AskQuestionResponse, 
    StatisticsResponse,
    SubjectListResponse,
    BaseResponse
)
from app.services.ai.rag_tutor_service import RagTutorService
from app.utils.dependencies import (
    get_rag_tutor_service,
    validate_upload_file,
    validate_subject_id,
    handle_service_errors
)

router = APIRouter(prefix="/tutor", tags=["Tutor"])


@router.post("/upload-material", response_model=UploadDocumentResponse)
@handle_service_errors
async def upload_material(
    file: UploadFile = Depends(validate_upload_file),
    subject_id: str = Form(..., description="Subject/course identifier"),
    metadata: str = Form(None, description="Additional metadata as JSON string"),
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    Upload and index learning material for RAG-based tutoring
    
    This endpoint allows uploading PDF, DOCX, and other supported documents
    that will be processed, chunked, and indexed for retrieval-augmented generation.
    
    - **file**: Document file (PDF, DOCX, etc.)
    - **subject_id**: Identifier for the subject/course (e.g., "math_101", "physics_advanced")
    - **metadata**: Optional additional metadata as JSON string
    
    Returns processing statistics and indexing information.
    """
    logger.info(f"📚 Upload request: {file.filename} for subject {subject_id}")
    
    # Validate subject ID
    subject_id = await validate_subject_id(subject_id)
    
    # Parse metadata if provided
    additional_metadata = {}
    if metadata:
        try:
            import json
            additional_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON format"
            )
    
    # Read file content
    file_content = await file.read()
    
    # Process document
    result = await rag_service.upload_document(
        file_content=file_content,
        filename=file.filename,
        subject_id=subject_id,
        metadata=additional_metadata
    )
    
    if result["success"]:
        return UploadDocumentResponse(
            success=True,
            message=result["message"],
            statistics=result["statistics"]
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )


@router.post("/ask-question", response_model=AskQuestionResponse)
@handle_service_errors
async def ask_question(
    request: AskQuestionRequest,
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    Ask Rin-chan a question about uploaded learning materials
    
    This endpoint uses RAG (Retrieval-Augmented Generation) to answer questions
    based on the documents uploaded for the specified subject.
    
    - **question**: The question to ask (in Vietnamese or English)
    - **subject_id**: Subject identifier to limit search scope
    - **chat_history**: Optional previous conversation for context
    - **use_reranking**: Whether to use reranking for better results (default: true)
    
    Rin-chan will search relevant documents and provide an answer with sources.
    """
    logger.info(f"Question for {request.subject_id}: {request.question[:100]}...")
    
    # Validate subject ID
    subject_id = await validate_subject_id(request.subject_id)
    
    # Convert chat history to proper format
    from app.core.interfaces.ai_service import ChatMessage
    chat_history = None
    if request.chat_history:
        chat_history = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in request.chat_history
        ]
    
    # Ask the question
    response = await rag_service.ask_question(
        question=request.question,
        subject_id=subject_id,
        chat_history=chat_history,
        use_reranking=request.use_reranking,
        question_image=request.question_image,
        option_images=request.option_images,
        force_fallback=request.force_fallback
    )
    
    if response.success:
        return AskQuestionResponse(
            success=True,
            answer=response.content,
            metadata=response.metadata
        )
    else:
        return AskQuestionResponse(
            success=False,
            error=response.error,
            metadata=response.metadata
        )


@router.get("/subjects", response_model=SubjectListResponse)
@handle_service_errors
async def list_subjects(
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    List all available subjects with learning materials
    
    Returns a list of subject IDs that have uploaded documents.
    """
    try:
        # Get all collections from vector store
        collections = rag_service.vector_store.list_collections()
        
        # Extract subject IDs (remove "subject_" prefix)
        subjects = []
        for collection in collections:
            if collection.startswith("subject_"):
                subject_id = collection[8:]  # Remove "subject_" prefix
                subjects.append(subject_id)
        
        return SubjectListResponse(
            success=True,
            subjects=subjects,
            total=len(subjects)
        )
        
    except Exception as e:
        logger.error(f"Failed to list subjects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subjects list"
        )


@router.get("/subjects/{subject_id}/stats", response_model=StatisticsResponse)
@handle_service_errors
async def get_subject_statistics(
    subject_id: str,
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    Get statistics for a specific subject
    
    Returns information about:
    - Number of documents indexed
    - Total chunks stored
    - Metadata analysis
    - Model information
    """
    # Validate subject ID
    subject_id = await validate_subject_id(subject_id)
    
    # Get statistics
    stats = await rag_service.get_subject_statistics(subject_id)
    
    return StatisticsResponse(
        success=True,
        statistics=stats
    )


@router.delete("/subjects/{subject_id}", response_model=BaseResponse)
@handle_service_errors
async def delete_subject_documents(
    subject_id: str,
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    Delete all documents for a subject
    
    **Warning**: This will permanently delete all uploaded documents 
    and their indexed content for the specified subject.
    """
    # Validate subject ID
    subject_id = await validate_subject_id(subject_id)
    
    # Delete documents
    success = await rag_service.delete_subject_documents(subject_id)
    
    if success:
        return BaseResponse(
            success=True,
            message=f"All documents for subject '{subject_id}' have been deleted"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents for subject '{subject_id}'"
        )


@router.get("/health", response_model=dict)
async def health_check(
    rag_service: RagTutorService = Depends(get_rag_tutor_service)
):
    """
    Health check for the RAG tutor service
    
    Returns the status of all underlying services:
    - AI service (Gemini)
    - Embedding service (GTE models)
    - Vector store (ChromaDB)
    """
    try:
        health_status = await rag_service.get_service_health()
        return {
            "status": "healthy",
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        ) 