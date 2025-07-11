"""
API version 1 main router
"""
from fastapi import APIRouter
from app.api.v1.routes import tutor, ai

# Create main v1 router
router = APIRouter(prefix="/v1")

# Include all route modules
router.include_router(tutor.router)
router.include_router(ai.router)

# API v1 root endpoint
@router.get("/")
async def api_v1_root():
    """
    API v1 root endpoint
    """
    return {
        "message": "AI Service API v1",
        "version": "1.0.0",
        "endpoints": {
            "tutor": {
                "upload_material": "POST /v1/tutor/upload-material",
                "ask_question": "POST /v1/tutor/ask-question",
                "list_subjects": "GET /v1/tutor/subjects",
                "subject_stats": "GET /v1/tutor/subjects/{subject_id}/stats",
                "delete_subject": "DELETE /v1/tutor/subjects/{subject_id}",
                "health": "GET /v1/tutor/health"
            },
            "ai": {
                "generate_text": "POST /v1/ai/generate-text",
                "chat": "POST /v1/ai/chat",
                "models": "GET /v1/ai/models",
                "statistics": "GET /v1/ai/statistics"
            }
        },
        "docs": "/docs",
        "redoc": "/redoc"
    } 