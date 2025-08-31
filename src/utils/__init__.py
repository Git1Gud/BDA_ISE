"""
Utility functions and helpers for the RAG system.
"""

import os
from typing import Optional, Dict, Any
from logger import logger

class PDFProcessor:
    """Handles PDF file processing operations."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a string.
        """
        text = ""
        try:
            import PyPDF2
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
            logger.info(f"Successfully extracted text from PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Error reading PDF file {pdf_path}: {e}")
        return text

class QueryParser:
    """Handles query parsing and type detection."""

    @staticmethod
    def parse_query_type(query: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse the query to determine if it's a module, unit, or topic query.

        Returns:
            Tuple of (query_type, module_number, unit_number)
            query_type: 'module', 'unit', or 'topic'
        """
        query_lower = query.lower().strip()

        # Check for module query (e.g., "module 1", "module 2")
        if query_lower.startswith("module "):
            try:
                module_num = query_lower.split("module ")[1].strip()
                return "module", module_num, None
            except:
                pass

        # Check for unit query (e.g., "unit 1.1", "unit 2.3")
        if query_lower.startswith("unit "):
            try:
                unit_num = query_lower.split("unit ")[1].strip()
                return "unit", None, unit_num
            except:
                pass

        # Check for direct module/unit numbers (e.g., "1.2", "2")
        import re
        module_unit_match = re.match(r'^(\d+)(?:\.(\d+))?$', query.strip())
        if module_unit_match:
            module_num = module_unit_match.group(1)
            unit_num = module_unit_match.group(2)
            if unit_num:
                return "unit", module_num, f"{module_num}.{unit_num}"
            else:
                return "module", module_num, None

        # Default to topic query
        return "topic", None, None

class TopicFilter:
    """Handles topic filtering based on query parameters."""

    @staticmethod
    def filter_topics_by_query(topics: list, query_type: str,
                             module_number: Optional[str], unit_number: Optional[str]) -> list:
        """
        Filter topics based on the query type and parameters.
        """
        from prompts import Topic

        if query_type == "module":
            # Return all topics from the specified module
            return [t for t in topics if t.module_number == module_number]
        elif query_type == "unit":
            # Return topics matching the specific unit
            # Try exact match first, then try partial matches
            exact_matches = [t for t in topics if t.unit_number == unit_number]
            if exact_matches:
                return exact_matches

            # Try matching just the unit part (e.g., "1.2" should match unit "1.2")
            partial_matches = [t for t in topics if unit_number and
                             (t.unit_number and unit_number in t.unit_number)]
            if partial_matches:
                return partial_matches

            # If no exact matches, return all topics (fallback)
            logger.warning(f"No exact unit matches for {unit_number}, returning all topics")
            return topics
        else:
            # For topic queries, return all topics (let the search handle filtering)
            return topics

class EnvironmentManager:
    """Manages environment variables and configuration loading."""

    @staticmethod
    def load_environment():
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Environment variables loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

    @staticmethod
    def get_required_env_vars() -> Dict[str, Optional[str]]:
        """Get all required environment variables."""
        return {
            'GROQ_API_KEY': os.getenv("GROQ_API_KEY"),
            'QDRANT_API_KEY': os.getenv("QDRANT_API_KEY"),
            'QDRANT_URL': os.getenv("QDRANT_URL"),
            'GEMINI_API_KEY': os.getenv("GEMINI_API_KEY")
        }

    @staticmethod
    def validate_environment() -> bool:
        """Validate that all required environment variables are set."""
        required_vars = EnvironmentManager.get_required_env_vars()
        missing_vars = [key for key, value in required_vars.items() if not value]

        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False

        logger.info("All required environment variables are set")
        return True
