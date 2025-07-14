"""
Application configuration management
"""
import os
from typing import List, Optional, Annotated
from pydantic import Field, BeforeValidator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


def parse_comma_separated_str(value: any) -> List[str]:
    if isinstance(value, str):
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        return value
    return []


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # App settings
    app_name: str = "AI Service"
    app_version: str = "1.0.0"
    debug: bool = Field(False, env="DEBUG")
    
    # Server settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    
    # AI API Keys - Support multiple keys for rotation
    gemini_keys: str = Field(..., env="GEMINI_KEYS")
    openai_keys: Optional[str] = Field(None, env="OPENAI_KEYS")
    
    # Vector Database
    vector_db_path: str = Field(default="./vector_db", env="VECTOR_DB_PATH")
    embedding_model: str = Field(default="Alibaba-NLP/gte-multilingual-base", env="EMBEDDING_MODEL")
    
    max_concurrent_embeddings: int = Field(default=3, env="MAX_CONCURRENT_EMBEDDINGS")
    embedding_timeout_seconds: int = Field(default=300, env="EMBEDDING_TIMEOUT_SECONDS")  # 5 minutes
    faiss_thread_pool_workers: int = Field(default=0, env="FAISS_THREAD_POOL_WORKERS")  # 0 = auto
    batch_size_documents: int = Field(default=10, env="BATCH_SIZE_DOCUMENTS")  # Process docs in batches
    
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_chunks_per_document: int = Field(default=50, env="MAX_CHUNKS_PER_DOCUMENT")
    
    # Models
    reranker_model: str = Field("Alibaba-NLP/gte-multilingual-reranker-base", env="RERANKER_MODEL")
    
    # RAG settings - Optimized for multiple choice questions
    max_retrieval_docs: int = Field(20, env="MAX_RETRIEVAL_DOCS")  # Increased for better coverage
    reranker_top_k: int = Field(8, env="RERANKER_TOP_K")  # Keep more relevant docs after reranking
    
    # File upload limits
    max_file_size: int = Field(50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_file_types: Annotated[List[str], BeforeValidator(parse_comma_separated_str)] = Field(
        default=["pdf", "docx", "pptx", "png", "jpg", "jpeg", "gif", "bmp", "webp"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Security
    api_key: Optional[str] = Field(None, env="API_KEY")
    cors_origins: Annotated[List[str], BeforeValidator(parse_comma_separated_str)] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    @property
    def gemini_key_list(self) -> List[str]:
        """Parse Gemini keys separated by comma, semicolon, or whitespace/newline"""
        import re
        return [k for k in re.split(r"[,;\s]+", self.gemini_keys.strip()) if k]
    
    @property
    def openai_key_list(self) -> List[str]:
        """Parse comma-separated OpenAI keys"""
        if not self.openai_keys:
            return []
        return [key.strip() for key in self.openai_keys.split(",") if key.strip()]
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings() 