from functools import lru_cache
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import os

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Other imports
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector, VectorParams, Distance
from logger import logger

# Local imports
from prompts import get_topic_extraction_prompt, get_study_material_prompt, Topic, Topics

# Configure logging
from logger import logger

load_dotenv()

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
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )

class StudyMaterialRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig.from_env()
        self._embedding_cache: Dict[str, List[float]] = {}
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all RAG components."""
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.config.gemini_api_key,
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)

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

        self._setup_vector_stores()

        self.topic_extraction_prompt = get_topic_extraction_prompt()
        self.study_material_prompt = get_study_material_prompt()

        # Local sparse corpus + TF-IDF (fallback hybrid if server fusion not available)
        self._syllabus_corpus: List[Dict[str, Any]] = []
        self._reference_corpus: List[Dict[str, Any]] = []
        self._syllabus_vectorizer: Optional[TfidfVectorizer] = None
        self._reference_vectorizer: Optional[TfidfVectorizer] = None
        self._syllabus_matrix = None
        self._reference_matrix = None

    def _setup_vector_stores(self) -> None:
        """Setup Qdrant vector stores."""
        self.qdrant_client = None
        if self.config.qdrant_url:
            try:
                self.qdrant_client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)
                embedding_dim = len(self.embeddings.embed_query("dimension probe"))
                self._ensure_collection(self.config.syllabus_collection, embedding_dim)
                self._ensure_collection(self.config.reference_collection, embedding_dim)
            except Exception as e:
                logger.warning(f"Unable to verify/create Qdrant collections: {e}")

        # Initialize vector stores
        try:
            if self.qdrant_client:
                self.syllabus_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.config.syllabus_collection,
                    embedding=self.embeddings,
                    timeout=60
                )
                self.reference_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.config.reference_collection,
                    embedding=self.embeddings
                )
            else:
                raise ValueError("No Qdrant client; will fallback to from_existing_collection")
        except Exception:
            # Fallback legacy initialization
            self.syllabus_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.config.syllabus_collection,
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                embedding=self.embeddings,
                timeout=60
            )
            self.reference_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.config.reference_collection,
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                embedding=self.embeddings
            )

    @lru_cache(maxsize=1000)
    def _cached_embed_query(self, text: str) -> str:
        """Cache embedding queries to avoid recomputation."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed_query_cached(self, text: str) -> List[float]:
        """Get cached embeddings or compute new ones."""
        cache_key = self._cached_embed_query(text)
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embeddings.embed_query(text)
        return self._embedding_cache[cache_key]

    # ---------------- Hybrid Retrieval (Dense + Sparse) -----------------
    def _rebuild_sparse_index(self, collection: str):
        corpus = self._syllabus_corpus if collection == 'syllabus' else self._reference_corpus
        if not corpus:
            return
        texts = [d['content'] for d in corpus]
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
        matrix = vectorizer.fit_transform(texts)
        if collection == 'syllabus':
            self._syllabus_vectorizer = vectorizer
            self._syllabus_matrix = matrix
        else:
            self._reference_vectorizer = vectorizer
            self._reference_matrix = matrix

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
            logger.info(f"Created Qdrant collection '{name}' (dim={dim}).")
        except Exception as e:
            logger.error(f"Failed to create or ensure collection '{name}': {e}")

    def _sparse_search(self, collection: str, query: str, k: int) -> List[Tuple[int, float]]:
        if collection == 'syllabus':
            if self._syllabus_vectorizer is None:
                return []
            vec = self._syllabus_vectorizer.transform([query])
            sims = (self._syllabus_matrix @ vec.T).toarray().ravel()
        else:
            if self._reference_vectorizer is None:
                return []
            vec = self._reference_vectorizer.transform([query])
            sims = (self._reference_matrix @ vec.T).toarray().ravel()
        if sims.size == 0:
            return []
        idxs = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idxs]

    def _dense_search(self, collection: str, query: str, k: int):
        store = self.syllabus_store if collection == 'syllabus' else self.reference_store
        try:
            # Use with scores to preserve order deterministically
            docs = store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Dense search failed ({collection}): {e}")
            return []

    def hybrid_search(self, collection: str, query: str, k_dense: int = 5, k_sparse: int = 10, k_final: int = 5) -> List[Any]:
        """
        Hybrid search with reciprocal rank fusion: 1/(1+rank_dense) + 1/(1+rank_sparse).

        Args:
            collection: Collection to search ('syllabus' or 'reference')
            query: Search query
            k_dense: Number of dense search results
            k_sparse: Number of sparse search results
            k_final: Final number of results to return

        Returns:
            List of search results
        """
        # Get dense search results
        dense_docs = self._dense_search(collection, query, k_dense)

        # Get sparse search results
        sparse_hits = self._sparse_search(collection, query, k_sparse)

        # Prepare corpus for sparse results
        corpus = self._syllabus_corpus if collection == 'syllabus' else self._reference_corpus

        # Combine results using reciprocal rank fusion
        combined: Dict[str, Dict[str, Any]] = {}

        # Add dense results
        for r, d in enumerate(dense_docs):
            combined.setdefault(d.page_content, {'doc': d})['dense_rank'] = r

        # Add sparse results
        for r, (idx, _score) in enumerate(sparse_hits):
            if idx < len(corpus):
                cont = corpus[idx]['content']
                if cont not in combined:
                    # Create a minimal doc-like wrapper
                    class _TempDoc:
                        page_content = cont
                        metadata = corpus[idx]['metadata']
                    combined[cont] = {'doc': _TempDoc(), 'sparse_rank': r}
                else:
                    combined[cont]['sparse_rank'] = r

        # Calculate final scores using reciprocal rank fusion
        scored = []
        for info in combined.values():
            score = 0.0
            if 'dense_rank' in info:
                score += 1.0 / (1 + info['dense_rank'])
            if 'sparse_rank' in info:
                score += 1.0 / (1 + info['sparse_rank'])
            scored.append((score, info['doc']))

        # Sort by score and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _s, d in scored[:k_final]]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a string.
        """
        text = ""
        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.error(f"Error reading PDF file {pdf_path}: {e}")
        return text

    def _preprocess_syllabus(self, syllabus_text: str) -> str:
        """Preprocess syllabus content to make it more structured."""
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
            logger.info("Syllabus text improved successfully.")
            return improved_syllabus
        except Exception as e:
            logger.error(f"Error improving syllabus text: {e}")
            return syllabus_text

    def _extract_and_store_topics(self, combined_syllabus: str, metadata: Dict[str, Any]) -> None:
        """Extract topics from syllabus and store them in the vector store."""
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
            topic_metadatas = [
                {
                    "topic_title": topic.title,
                    "module_number": topic.module_number,
                    "unit_number": topic.unit_number,
                    "teacher_id": metadata.get("teacher_id", "Unknown"),
                    "document_type": "extracted_topic"
                }
                for topic in topics
            ]

            self.syllabus_store.add_texts(topic_texts, metadatas=topic_metadatas)
            logger.info(f"Extracted topics added to the store: {[topic.title for topic in topics]}")

        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            logger.error(f"Raw output: {result.content}")

    def add_syllabus(self, syllabus_text: str, metadata: Dict[str, Any] = None) -> None:
        """
        Preprocess and add syllabus document to the syllabus collection and extract topics.

        Args:
            syllabus_text: Text content of the syllabus
            metadata: Metadata for the syllabus (course name, teacher ID, etc.)
        """
        if not metadata:
            metadata = {}

        if 'teacher_id' not in metadata:
            logger.warning("No teacher_id specified in metadata")

        if 'document_type' not in metadata:
            metadata['document_type'] = 'syllabus'

        # Preprocess syllabus
        improved_syllabus = self._preprocess_syllabus(syllabus_text)

        # Split and store chunks
        texts = self.syllabus_splitter.split_text(improved_syllabus)
        metadatas = [metadata.copy() for _ in texts]

        # Add syllabus chunks to local sparse corpus for hybrid
        for t, m in zip(texts, metadatas):
            self._syllabus_corpus.append({'content': t, 'metadata': m})
        self._rebuild_sparse_index('syllabus')

        # Extract and store topics
        combined_syllabus = "\n\n".join(texts)
        self._extract_and_store_topics(combined_syllabus, metadata)

    def add_reference_material(self, reference_text: str, metadata: Dict[str, Any] = None):
        """
        Add reference material to the reference collection.

        Args:
            reference_text: Text content of the reference material
            metadata: Metadata for the reference (source, author, teacher ID, etc.)
        """
        texts = self.reference_splitter.split_text(reference_text)

        if not metadata:
            metadata = {}

        if 'teacher_id' not in metadata:
            logger.warning("No teacher_id specified in metadata")

        if 'document_type' not in metadata:
            metadata['document_type'] = 'reference'

        metadatas = [metadata.copy() for _ in texts]

        self.reference_store.add_texts(texts, metadatas=metadatas)
        # Track reference chunks for sparse index
        for t, m in zip(texts, metadatas):
            self._reference_corpus.append({'content': t, 'metadata': m})
        self._rebuild_sparse_index('reference')
        logger.info("Reference material added successfully (hybrid indices updated).")

    def extract_topics(self, course_query: str) -> Topics:
        """
        Extract topics from syllabus based on course query.

        Args:
            course_query: Query to find relevant syllabus content

        Returns:
            Topics object containing list of Topic objects
        """
        # Parse query type to determine search strategy
        query_type, module_number, unit_number = self._parse_query_type(course_query)

        # Get syllabus content based on query type
        syllabus_content = self._get_syllabus_content_for_query(query_type, module_number, unit_number, course_query)

        if not syllabus_content:
            logger.warning("No relevant syllabus content found.")
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
            logger.info(f"Extracted topics: {len(topics_container.topics)} topics found")
            for i, topic in enumerate(topics_container.topics):
                logger.debug(f"Topic {i+1}: module={topic.module_number}, unit={topic.unit_number}, title={topic.title}")
            return topics_container
        except Exception as e:
            logger.error(f"Error parsing topics: {e}")
            logger.error(f"Raw output: {result.content}")
            return Topics(topics=[])

    def generate_study_material(self, topic: Topic, query: str, teacher_id: str = "Unknown") -> str:
        """
        Generate study material for a specific topic.

        Args:
            topic: Topic object with title, description, and subtopics
            query: The original query string
            teacher_id: ID of the teacher creating the material

        Returns:
            Marp-compatible markdown for the study material
        """
        # Prepare search query
        processed_query = self._prepare_search_query(topic, query)
        logger.debug(f"Processed query: {processed_query}")

        # Get reference materials
        reference_content = self._get_reference_content(processed_query)
        if not reference_content:
            logger.warning(f"No reference materials found for topic: {topic.title}")
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
            logger.error(f"Error generating study material for topic {topic.title}: {e}")
            return self._create_empty_material_response(topic.title)

    def _load_css_context(self) -> str:
        """
        Load the CSS context for the LLM to understand styling constraints.
        
        Returns:
            String containing key CSS styling information
        """
        css_file_path = os.path.join(os.path.dirname(__file__), "assets", "dark-theme.css")
        
        try:
            with open(css_file_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            
            # Extract key styling information for the LLM
            css_context = f"""
DARK THEME STYLING CONTEXT:

Color Scheme:
- Background: Dark gray (#1a1a1a)
- Text: White (#ffffff)
- Code blocks: Dark background (#2d2d2d) with light text (#f8f8f2)
- Links: Blue (#4dabf7)
- Accents: Blue (#007acc)

Layout Constraints:
- Slides are 16:9 aspect ratio (1280x720)
- Content must fit within slide boundaries
- Images max 100% width, 80% viewport height
- Text uses overflow-wrap: break-word for long words
- Maximum 8-12 bullet points per slide recommended

Visual Elements:
- Mermaid diagrams: Dark theme compatible
- Tables: Dark background with light text
- Blockquotes: Left border accent with dark background
- Code blocks: Rounded corners, proper padding

Content Guidelines:
- Use white text on dark background
- Ensure high contrast for readability
- Keep content concise to fit slide boundaries
- Use bullet points extensively
- Include at least 1 image every 3-4 slides
- Keep Mermaid diagrams under 800px width, 400px height

Technical Constraints:
- All content must be responsive
- No horizontal scrolling allowed
- Word wrapping enabled for long text
- Images automatically center and scale
"""
            return css_context
            
        except FileNotFoundError:
            logger.warning("CSS file not found, using default styling context")
            return """
DARK THEME STYLING CONTEXT:
- Dark background (#1a1a1a) with white text (#ffffff)
- Content must fit slide boundaries
- Use bullet points, keep text concise
- Include images and diagrams appropriately
- Ensure readability on dark theme
"""
        except Exception as e:
            logger.warning(f"Error loading CSS context: {e}")
            return "Dark theme with white text, ensure content fits slides."

    def _parse_query_type(self, query: str) -> Tuple[str, Optional[str], Optional[str]]:
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

    def _filter_topics_by_query(self, topics: List[Topic], query_type: str,
                               module_number: Optional[str], unit_number: Optional[str]) -> List[Topic]:
        """
        Filter topics based on the query type and parameters.
        """
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

    def create_full_course_materials(self, course_query: str, teacher_id: str) -> Dict[str, str]:
        """
        Create study materials based on syllabus and references.
        Supports module, unit, and topic level queries.

        Args:
            course_query: Query to find the relevant syllabus content
                         Can be: "module 1", "unit 1.2", "1.2", or topic name
            teacher_id: ID of the teacher creating the course

        Returns:
            Dictionary mapping the topic title to its study material content
        """
        # Parse the query type
        query_type, module_number, unit_number = self._parse_query_type(course_query)
        logger.info(f"Query type: {query_type}, Module: {module_number}, Unit: {unit_number}")

        # Extract all topics from syllabus
        topics_model = self.extract_topics(course_query)

        if not topics_model or not topics_model.topics:
            logger.warning("No topics could be extracted from the syllabus.")
            return {"error": "No topics could be extracted from the syllabus."}

        # Filter topics based on query type
        filtered_topics = self._filter_topics_by_query(
            topics_model.topics, query_type, module_number, unit_number
        )

        if not filtered_topics:
            logger.warning(f"No topics found for query: {course_query}")
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
                logger.info(f"Generated material for topic: {topic.title}")
            except Exception as e:
                logger.error(f"Failed to generate material for topic {topic.title}: {e}")
                materials[topic.title] = f"# {topic.title}\n\nError generating content: {str(e)}"

        return materials

# # Example usage
# if __name__ == "__main__":
#     study_system = StudyMaterialRAG()

#     # Add a syllabus from a PDF file
#     syllabus_pdf_path = "syllabus.pdf"  # Replace with the actual path to your syllabus PDF
#     syllabus_text = study_system.extract_text_from_pdf(syllabus_pdf_path)
#     study_system.add_syllabus(syllabus_text, {"course_id": "CS101", "teacher_id": "T123"})

#     # Add reference materials from a PDF file
#     reference_pdf_path = "reference.pdf"  # Replace with the actual path to your reference PDF
#     reference_text = study_system.extract_text_from_pdf(reference_pdf_path)
#     study_system.add_reference_material(reference_text, {"source": "Programming Fundamentals", "teacher_id": "T123"})

#     # Generate materials for a single topic
#     materials = study_system.create_full_course_materials("Virtualisation", "T123")
    
#     # Save the output to a .md file
#     output_file = "study_materials.md"
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write('''---
# marp: true
# theme: gaia
# paginate: true
# backgroundColor: "#1E1E2E"
# color: white
# \n''')
#         for topic, content in materials.items():
#             f.write('---\n')
#             f.write(f"### {topic}\n\n")
#             f.write(content)
#             f.write("\n\n")

#     print(f"Study materials saved to {output_file}")
#     pptx_path = os.path.join('./', "slides.pptx")
#     os.system(f"marp {output_file} --pptx -o {pptx_path}")

