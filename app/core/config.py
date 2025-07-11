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
    vector_db_path: str = Field("./vector_db", env="VECTOR_DB_PATH")
    
    # Models
    embedding_model: str = Field("Alibaba-NLP/gte-multilingual-base", env="EMBEDDING_MODEL")
    reranker_model: str = Field("Alibaba-NLP/gte-multilingual-reranker-base", env="RERANKER_MODEL")
    
    # RAG settings
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")
    max_retrieval_docs: int = Field(10, env="MAX_RETRIEVAL_DOCS")
    reranker_top_k: int = Field(5, env="RERANKER_TOP_K")
    
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