"""
Main RAG system that orchestrates all components.
"""

from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from .core import RAGConfig, BaseRAGComponent
from .search import VectorStoreManager, SearchEngine
from .processing import ContentProcessor, MaterialGenerator, FileManager
from .utils import PDFProcessor, QueryParser, TopicFilter, EnvironmentManager
from prompts import get_topic_extraction_prompt, get_study_material_prompt, Topic, Topics

class StudyMaterialRAG(BaseRAGComponent):
    """Main RAG system orchestrating all components."""

    def __init__(self, config: Optional[RAGConfig] = None):
        # Load environment if not provided
        if config is None:
            EnvironmentManager.load_environment()
            config = RAGConfig.from_env()

        super().__init__(config)

        # Initialize components
        self.vector_manager = VectorStoreManager(config)
        self.content_processor = ContentProcessor(config)
        self.material_generator = MaterialGenerator(config)
        self.file_manager = FileManager(config)

        # Initialize LLM and vector stores
        self._setup_llm()
        self._setup_vector_stores()
        self._setup_text_splitters()
        self._setup_prompts()

        # Initialize search engine after vector stores are set up
        self.search_engine = SearchEngine(config, self.vector_manager, {
            'syllabus': self.syllabus_store,
            'reference': self.reference_store
        })

        self._log_operation("RAG system initialized")

    def _setup_llm(self):
        """Initialize the language model."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp',
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=self.config.gemini_api_key,
            )
            self._log_operation("LLM initialized", "gemini-2.0-flash-exp")
        except Exception as e:
            self._log_error("LLM initialization", e)
            raise

    def _setup_vector_stores(self):
        """Setup Qdrant vector stores."""
        self.qdrant_client = None
        if self.config.qdrant_url:
            try:
                self.qdrant_client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)
                embedding_dim = len(self.vector_manager.embeddings.embed_query("dimension probe"))
                self._ensure_collection(self.config.syllabus_collection, embedding_dim)
                self._ensure_collection(self.config.reference_collection, embedding_dim)
            except Exception as e:
                self._log_error("Vector store setup", e)

        # Initialize vector stores
        try:
            if self.qdrant_client:
                self.syllabus_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.config.syllabus_collection,
                    embedding=self.vector_manager.embeddings,
                    timeout=60
                )
                self.reference_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.config.reference_collection,
                    embedding=self.vector_manager.embeddings
                )
            else:
                raise ValueError("No Qdrant client; will fallback to from_existing_collection")
        except Exception:
            # Fallback legacy initialization
            self.syllabus_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.config.syllabus_collection,
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                embedding=self.vector_manager.embeddings,
                timeout=60
            )
            self.reference_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.config.reference_collection,
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                embedding=self.vector_manager.embeddings
            )

    def _setup_text_splitters(self):
        """Initialize text splitters for content processing."""
        self.syllabus_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.syllabus_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )

        self.reference_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.reference_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )

    def _setup_prompts(self):
        """Initialize prompt templates."""
        self.topic_extraction_prompt = get_topic_extraction_prompt()
        self.study_material_prompt = get_study_material_prompt()

    def _ensure_collection(self, name: str, dim: int):
        """Create Qdrant collection with given name and embedding dim if it does not exist."""
        if not self.qdrant_client:
            return
        try:
            existing = {c.name for c in self.qdrant_client.get_collections().collections}
            if name in existing:
                return
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            self._log_operation("Created Qdrant collection", f"'{name}' (dim={dim})")
        except Exception as e:
            self._log_error("Collection creation", e)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        return PDFProcessor.extract_text_from_pdf(pdf_path)

    def hybrid_search(self, collection: str, query: str, k_dense: int = 5, k_sparse: int = 10, k_final: int = 5):
        """Perform hybrid search across collections."""
        return self.search_engine.hybrid_search(collection, query, k_dense, k_sparse, k_final)

    def extract_topics(self, course_query: str) -> Topics:
        """Extract topics from syllabus based on course query."""
        # Parse query type to determine search strategy
        query_type, module_number, unit_number = QueryParser.parse_query_type(course_query)

        # Get syllabus content based on query type
        syllabus_content = self._get_syllabus_content_for_query(query_type, module_number, unit_number, course_query)

        if not syllabus_content:
            self.logger.warning("No relevant syllabus content found.")
            return Topics(topics=[])

        # Extract topics using LLM
        return self._extract_topics_from_content(syllabus_content)

    def _get_syllabus_content_for_query(self, query_type: str, module_number: Optional[str],
                                       unit_number: Optional[str], course_query: str) -> str:
        """Get relevant syllabus content based on query type."""
        if query_type in ["module", "unit"]:
            # For module/unit queries, search for the entire syllabus content
            results = self.hybrid_search('syllabus', "syllabus module unit", k_dense=10, k_sparse=20, k_final=10)
        else:
            # For topic queries, use the original search logic
            results = self.hybrid_search('syllabus', course_query, k_dense=3, k_sparse=10, k_final=3)

        return "\n\n".join([doc.page_content for doc in results]) if results else ""

    def _extract_topics_from_content(self, syllabus_content: str) -> Topics:
        """Extract topics from syllabus content using LLM."""
        parser = PydanticOutputParser(pydantic_object=Topics)
        chain = self.topic_extraction_prompt | self.llm

        result = chain.invoke({"syllabus_content": syllabus_content})

        try:
            topics_container = parser.parse(result.content)
            self._log_operation("Topics extracted", f"{len(topics_container.topics)} topics found")
            return topics_container
        except Exception as e:
            self._log_error("Topic parsing", e)
            return Topics(topics=[])

    def generate_study_material(self, topic: Topic, query: str, teacher_id: str = "Unknown") -> str:
        """Generate study material for a specific topic."""
        # Prepare search query
        processed_query = self._prepare_search_query(topic, query)
        self.logger.debug(f"Processed query: {processed_query}")

        # Get reference materials
        reference_content = self._get_reference_content(processed_query)
        if not reference_content:
            self.logger.warning(f"No reference materials found for topic: {topic.title}")
            return self._create_empty_material_response(topic.title)

        # Generate material using LLM
        return self._generate_material_with_llm(topic, query, reference_content, teacher_id)

    def _prepare_search_query(self, topic: Topic, query: str) -> str:
        """Prepare optimized search query for reference materials."""
        subtopics = topic.subtopics if topic.subtopics else []
        return f"{query} {topic.title} {' '.join(subtopics)}"

    def _get_reference_content(self, processed_query: str) -> str:
        """Retrieve relevant reference content."""
        results = self.hybrid_search('reference', processed_query, k_dense=5, k_sparse=12, k_final=5)
        return "\n\n".join([doc.page_content for doc in results]) if results else ""

    def _create_empty_material_response(self, topic_title: str) -> str:
        """Create response when no reference materials are found."""
        return f"# {topic_title}\n\nNo reference materials available for this topic."

    def _generate_material_with_llm(self, topic: Topic, query: str, reference_content: str, teacher_id: str) -> str:
        """Generate study material using LLM with proper error handling."""
        chain = self.study_material_prompt | self.llm

        try:
            result = chain.invoke({
                "topic": query,
                "description": f"{topic.title} {' '.join(topic.subtopics or [])}",
                "reference_content": reference_content,
                "css_context": "Dark theme with overflow prevention - focus on mermaid diagrams",
                "teacher_id": teacher_id
            })
            return result.content
        except Exception as e:
            self._log_error("Material generation", e)
            return self._create_empty_material_response(topic.title)

    def create_full_course_materials(self, course_query: str, teacher_id: str) -> Dict[str, str]:
        """Create study materials based on syllabus and references."""
        # Parse the query type
        query_type, module_number, unit_number = QueryParser.parse_query_type(course_query)
        self._log_operation("Creating course materials", f"type={query_type}, module={module_number}, unit={unit_number}")

        # Extract all topics from syllabus
        topics_model = self.extract_topics(course_query)

        if not topics_model or not topics_model.topics:
            self.logger.warning("No topics could be extracted from the syllabus.")
            return {"error": "No topics could be extracted from the syllabus."}

        # Filter topics based on query type
        filtered_topics = TopicFilter.filter_topics_by_query(
            topics_model.topics, query_type, module_number, unit_number
        )

        if not filtered_topics:
            self.logger.warning(f"No topics found for query: {course_query}")
            return {"error": f"No topics found for query: {course_query}"}

        # Generate materials for filtered topics
        return self._generate_materials_for_topics(filtered_topics, course_query, teacher_id)

    def _generate_materials_for_topics(self, topics: List[Topic], course_query: str, teacher_id: str) -> Dict[str, str]:
        """Generate study materials for a list of topics."""
        materials = {}
        for topic in topics:
            try:
                material = self.generate_study_material(topic, course_query, teacher_id)
                materials[topic.title] = material
                self._log_operation("Generated material", f"topic: {topic.title}")
            except Exception as e:
                self._log_error("Material generation", e)
                materials[topic.title] = f"# {topic.title}\n\nError generating content: {str(e)}"

        return materials

    def add_syllabus_content(self, syllabus_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add syllabus content to the vector store and sparse index."""
        try:
            # Preprocess syllabus
            processed_syllabus = self._preprocess_syllabus(syllabus_text)

            # Extract and store topics
            self._extract_and_store_topics(processed_syllabus, metadata or {})

            self._log_operation("Syllabus content added", f"length: {len(processed_syllabus)}")
        except Exception as e:
            self._log_error("Adding syllabus content", e)

    def add_reference_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add reference content to the vector store and sparse index."""
        try:
            # Split content into chunks
            chunks = self.reference_splitter.split_text(content)

            # Add to vector store
            metadatas = [metadata or {}] * len(chunks)
            self.reference_store.add_texts(chunks, metadatas=metadatas)

            # Add to sparse corpus
            for i, chunk in enumerate(chunks):
                self.search_engine._reference_corpus.append({
                    'content': chunk,
                    'metadata': metadata or {}
                })

            # Rebuild sparse index
            self.search_engine._rebuild_sparse_index('reference')

            self._log_operation("Reference content added", f"chunks: {len(chunks)}")
        except Exception as e:
            self._log_error("Adding reference content", e)

    def _preprocess_syllabus(self, syllabus_text: str) -> str:
        """Preprocess syllabus content to make it more structured."""
        from langchain.prompts import PromptTemplate

        preprocess_prompt = PromptTemplate(
            template="""Improve the following syllabus content by making it more structured, clear, and concise.
            Ensure the content is well-organized and easy to understand.
            Only add things which look MODULES. DO NOT ADD COURSE OUTCOME,LAB RELATED THINGS. Also just gives Modules and units please.

            Original Syllabus:
            {syllabus_content}

            Improved Syllabus:""",
            input_variables=["syllabus_content"]
        )
        chain = preprocess_prompt | self.llm
        result = chain.invoke({"syllabus_content": syllabus_text})

        try:
            improved_syllabus = result.content
            self._log_operation("Syllabus preprocessing", "completed")
            return improved_syllabus
        except Exception as e:
            self._log_error("Syllabus preprocessing", e)
            return syllabus_text

    def _extract_and_store_topics(self, combined_syllabus: str, metadata: Dict[str, Any]) -> None:
        """Extract topics from syllabus and store them in the vector store."""
        from langchain.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=Topics)
        chain = self.topic_extraction_prompt | self.llm

        result = chain.invoke({"syllabus_content": combined_syllabus})

        try:
            topics_container = parser.parse(result.content)
            topics = topics_container.topics

            topic_texts = [
                f"Module: {topic.module_number or 'N/A'}\nUnit: {topic.unit_number or 'N/A'}\nTitle: {topic.title}\nDescription: {topic.description}\nSubtopics: {', '.join(topic.subtopics or [])}"
                for topic in topics
            ]

            # Add topics to vector store
            topic_metadatas = []
            for topic in topics:
                topic_metadata = metadata.copy()
                topic_metadata.update({
                    'module_number': topic.module_number,
                    'unit_number': topic.unit_number,
                    'title': topic.title,
                    'subtopics': topic.subtopics
                })
                topic_metadatas.append(topic_metadata)

            self.syllabus_store.add_texts(topic_texts, metadatas=topic_metadatas)

            # Add to sparse corpus
            for i, topic_text in enumerate(topic_texts):
                self.search_engine._syllabus_corpus.append({
                    'content': topic_text,
                    'metadata': topic_metadatas[i]
                })

            # Rebuild sparse index
            self.search_engine._rebuild_sparse_index('syllabus')

            self._log_operation("Topics extracted and stored", f"count: {len(topics)}")
        except Exception as e:
            self._log_error("Topic extraction and storage", e)
