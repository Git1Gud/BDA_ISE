import os
from src import StudyMaterialRAG, RAGConfig
import sys
from material import generate_materials
from prompts import Topic
from logger import logger

def main():
    """Main application entry point."""
    logger.info("Starting Study Materials Generator")

    try:
        # Load environment variables first
        from src.utils import EnvironmentManager
        EnvironmentManager.load_environment()

        # Create RAG system with configuration
        config = RAGConfig.from_env()
        rag = StudyMaterialRAG(config)
        logger.info("RAG system initialized successfully")

        # Add syllabus content for testing
        syllabus = """
        1 Title Introduction to Distributed Systems
        1.1 Definition, Goals, Types of Distributed Computing Models, Issues in
        Distributed Systems.

        1.2 Hardware Concepts, Software Concepts, The Client-Server Model,
        Positioning Middleware, Models of Middleware, Services offered by
        Middleware.
        """

        logger.info("Adding syllabus content...")
        rag.add_syllabus_content(syllabus, {"course": "Distributed Systems", "source": "test"})
        logger.info("Syllabus content added successfully")

        # Test query
        query = "unit 1.2"
        logger.info(f"Processing query: {query}")

        # Perform hybrid search
        logger.info("Performing hybrid search...")
        results = rag.hybrid_search("reference", query, k_dense=8, k_sparse=24, k_final=10)
        logger.info(f"Hybrid search completed. Found {len(results)} results")

        for i, d in enumerate(results, 1):
            logger.debug(f"Result {i}: {d.page_content[:120].replace(chr(10),' ')}...")

        # Extract topics and generate materials
        logger.info("Extracting topics...")
        topics_model = rag.extract_topics(query)

        if topics_model and topics_model.topics:
            first_topic = topics_model.topics[0]
            logger.info(f"First topic: {first_topic.title}")
            logger.debug(f"Subtopics: {first_topic.subtopics}")

            try:
                logger.info("Generating study materials...")
                material = generate_materials(rag, query, teacher_id="T1", output_format='pdf')
                logger.info("Material generation completed successfully")
            except Exception as e:
                logger.error(f"Error generating materials: {e}")
                raise
        else:
            logger.warning("No topics found")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()