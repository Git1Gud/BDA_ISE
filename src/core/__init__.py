"""
Core RAG system components and configuration.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from logger import logger

@dataclass
class RAGConfig:
    """Configuration class for RAG system."""
    groq_api_key: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    gemini_api_key: Optional[str] = None
    provider: str = "google"
    model_name: str = "gemini-2.5-pro"
    model_path: str = "models/llama-2-7b-chat.Q4_K_M.gguf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    syllabus_collection: str = "syllabus"
    reference_collection: str = "references"
    syllabus_chunk_size: int = 500
    reference_chunk_size: int = 1000
    chunk_overlap: int = 100

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create config from environment variables."""
        import os
        # Optional fallback to global Config constants if present
        try:
            from config import config as app_config  # type: ignore
        except Exception:
            app_config = None

        def _get(name: str, default: Optional[str] = None):
            return os.getenv(name) or (getattr(app_config, name, None) if app_config else None) or default

        return cls(
            groq_api_key=_get("GROQ_API_KEY"),
            qdrant_api_key=_get("QDRANT_API_KEY"),
            qdrant_url=_get("QDRANT_URL"),
            gemini_api_key=_get("GEMINI_API_KEY"),
            provider=_get("PROVIDER", "google"),
            model_name=_get("MODEL_NAME", "gemini-2.5-pro"),
            model_path=_get("MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf"),
            embedding_model=_get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            syllabus_collection=_get("SYLLABUS_COLLECTION", "syllabus"),
            reference_collection=_get("REFERENCE_COLLECTION", "references"),
            syllabus_chunk_size=int(_get("SYLLABUS_CHUNK_SIZE", "500")),
            reference_chunk_size=int(_get("REFERENCE_CHUNK_SIZE", "1000")),
            chunk_overlap=int(_get("CHUNK_OVERLAP", "100")),
        )

class BaseRAGComponent:
    """Base class for RAG components with common functionality."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logger

    def _log_operation(self, operation: str, details: str = ""):
        """Log operation with consistent formatting."""
        self.logger.info(f"{operation}: {details}")

    def _log_error(self, operation: str, error: Exception):
        """Log error with consistent formatting."""
        self.logger.error(f"{operation} failed: {str(error)}")
