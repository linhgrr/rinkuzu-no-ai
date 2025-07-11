"""
AI Service - Main FastAPI Application

A comprehensive AI microservice for educational applications featuring:
- RAG-based tutoring with Rin-chan
- PDF/DOCX document processing
- Multi-modal AI capabilities
- Vector search and reranking
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from loguru import logger
import uvicorn
import time 

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings
from app.api.v1 import router as v1_router


# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Handles startup and shutdown events for the FastAPI application
    """
    # Startup
    logger.info("🚀 Starting AI Service...")
    logger.info(f"🔧 Configuration: {settings.app_name} v{settings.app_version}")
    logger.info(f"🔧 Environment: Debug={settings.debug}")
    logger.info(f"🔧 Vector DB: {settings.vector_db_path}")
    
    # Initialize services
    try:
        from app.utils.dependencies import get_rag_tutor_service
        rag_service = get_rag_tutor_service()
        health = await rag_service.get_service_health()
        logger.info("✅ All services initialized successfully")
        logger.info(f"📊 Service health: {health['ai_service']['name']} available")
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Service...")


# Create FastAPI application
app = FastAPI(
    title="AI Service",
    description="""
## 🤖 AI Service - Educational Assistant Platform

A comprehensive AI microservice designed for educational applications, featuring **Rin-chan**, 
an intelligent RAG-based tutor system.

### 🌟 Key Features

- **📚 RAG Tutoring**: Upload learning materials and ask questions
- **🔍 Intelligent Search**: Vector search with reranking for accurate answers  
- **💬 Multi-turn Chat**: Contextual conversations with AI tutor
- **📄 Document Processing**: Support for PDF, DOCX, and other formats
- **🌐 Multi-language**: Vietnamese and English support
- **⚡ High Performance**: Optimized embedding and generation pipeline

### 🛠️ Technology Stack

- **AI Models**: Google Gemini 2.0 Flash
- **Embeddings**: GTE Multilingual Base & Reranker
- **Vector DB**: ChromaDB with persistent storage
- **Framework**: FastAPI with async/await
- **Architecture**: Clean OOP with dependency injection

### 📖 Usage Examples

1. **Upload Learning Material**:
   ```bash
   curl -X POST "/v1/tutor/upload-material" \\
     -F "file=@lecture_notes.pdf" \\
     -F "subject_id=computer_science_101"
   ```

2. **Ask Rin-chan a Question**:
   ```bash
   curl -X POST "/v1/tutor/ask-question" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "CPU hoạt động như thế nào?", "subject_id": "computer_science_101"}'
   ```

### 👨‍💻 For Developers

- **Clean Architecture**: Interfaces, services, and dependency injection
- **Plug-and-play**: Easy to swap AI models and vector stores
- **Comprehensive Logging**: Detailed logging with loguru
- **Type Safety**: Full Pydantic models and type hints
- **Documentation**: Auto-generated OpenAPI docs

Perfect for educational platforms, learning management systems, and AI-powered tutoring applications.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Add trusted host middleware for production
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.vercel.app"]
    )


# Custom OpenAPI schema with enhanced documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Service API",
        version=settings.app_version,
        description="Comprehensive AI microservice for educational applications",
        routes=app.routes,
    )
    
    # Add custom server information
    openapi_schema["servers"] = [
        {"url": "/", "description": "Current server"},
        {"url": "http://localhost:8000", "description": "Local development"},
    ]
    
    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "Tutor",
            "description": "RAG-based tutoring with Rin-chan. Upload documents and ask questions with intelligent context retrieval."
        },
        {
            "name": "AI",
            "description": "General AI capabilities including text generation and chat conversations."
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Include API routers
app.include_router(v1_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    AI Service root endpoint
    
    Welcome to the AI Service! This microservice provides comprehensive
    AI capabilities for educational applications.
    """
    return {
        "message": "🤖 Welcome to AI Service",
        "version": settings.app_version,
        "description": "Educational AI microservice with RAG-based tutoring",
        "features": [
            "📚 Document upload and indexing",
            "🤔 Intelligent question answering",
            "💬 Multi-turn conversations",
            "🔍 Vector search with reranking",
            "🌐 Multi-language support"
        ],
        "api_docs": "/docs",
        "endpoints": {
            "v1": "/v1"
        },
        "status": "🟢 Service running"
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Service health check
    
    Returns the overall health status of the AI service and its components.
    """
    try:
        from app.utils.dependencies import check_services_health
        health_status = await check_services_health()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.app_version,
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": settings.app_version
            }
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception for {request.url}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else "Internal server error",
            "path": str(request.url),
            "method": request.method
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all HTTP requests for monitoring and debugging
    """
    start_time = time.time()

    logger.info(f"📥 {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url} - {response.status_code} - {process_time:.2f}s")

    return response


# Run the application
if __name__ == "__main__":
    logger.info(f"🚀 Starting AI Service on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=True,
        log_level=settings.log_level.lower()
    ) 