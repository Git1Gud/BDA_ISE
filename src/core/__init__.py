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
    model_name: str = "llama3-8b-8192"
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
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            gemini_api_key=os.getenv("GEMINI_API_KEY")
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
