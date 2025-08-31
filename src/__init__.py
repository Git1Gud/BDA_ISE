"""
Modular RAG System for Study Material Generation.

This package provides a modular implementation of a RAG (Retrieval-Augmented Generation)
system specifically designed for generating educational study materials.
"""

from . import core, search, processing, utils
from .core import RAGConfig, BaseRAGComponent
from .search import VectorStoreManager, SearchEngine
from .processing import ContentProcessor, MaterialGenerator, FileManager
from .utils import PDFProcessor, QueryParser, TopicFilter, EnvironmentManager
from .rag_system import StudyMaterialRAG

__version__ = "1.0.0"
__all__ = [
    # Main classes
    "StudyMaterialRAG",
    "RAGConfig",

    # Core components
    "BaseRAGComponent",

    # Search components
    "VectorStoreManager",
    "SearchEngine",

    # Processing components
    "ContentProcessor",
    "MaterialGenerator",
    "FileManager",

    # Utility components
    "PDFProcessor",
    "QueryParser",
    "TopicFilter",
    "EnvironmentManager",

    # Submodules
    "core",
    "search",
    "processing",
    "utils"
]
