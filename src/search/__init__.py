"""
Search and retrieval operations for the RAG system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import hashlib

from ..core import BaseRAGComponent, RAGConfig

class VectorStoreManager(BaseRAGComponent):
    """Manages vector store operations and embeddings."""

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._setup_embeddings()
        self._setup_vectorizers()

    def _setup_embeddings(self):
        """Initialize embedding models."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            self._log_operation("Embeddings initialized", self.config.embedding_model)
        except Exception as e:
            self._log_error("Embedding initialization", e)
            raise

    def _setup_vectorizers(self):
        """Initialize TF-IDF vectorizers for sparse search."""
        self._syllabus_vectorizer: Optional[TfidfVectorizer] = None
        self._reference_vectorizer: Optional[TfidfVectorizer] = None
        self._syllabus_matrix = None
        self._reference_matrix = None

    @lru_cache(maxsize=1000)
    def _cached_embed_query(self, text: str) -> str:
        """Cache embedding queries to avoid recomputation."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed_query_cached(self, text: str) -> List[float]:
        """Get cached embeddings or compute new ones."""
        cache_key = self._cached_embed_query(text)
        if not hasattr(self, '_embedding_cache'):
            self._embedding_cache: Dict[str, List[float]] = {}

        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embeddings.embed_query(text)

        return self._embedding_cache[cache_key]

class SearchEngine(BaseRAGComponent):
    """Handles search operations across different collections."""

    def __init__(self, config: RAGConfig, vector_manager: VectorStoreManager, vector_stores: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_manager = vector_manager
        self._vector_stores = vector_stores or {}
        self._syllabus_corpus: List[Dict[str, Any]] = []
        self._reference_corpus: List[Dict[str, Any]] = []

    def _rebuild_sparse_index(self, collection: str):
        """Rebuild sparse index for a collection."""
        corpus = self._syllabus_corpus if collection == 'syllabus' else self._reference_corpus
        if not corpus:
            return

        texts = [d['content'] for d in corpus]
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
        matrix = vectorizer.fit_transform(texts)

        if collection == 'syllabus':
            self.vector_manager._syllabus_vectorizer = vectorizer
            self.vector_manager._syllabus_matrix = matrix
        else:
            self.vector_manager._reference_vectorizer = vectorizer
            self.vector_manager._reference_matrix = matrix

    def _sparse_search(self, collection: str, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform sparse search using TF-IDF."""
        if collection == 'syllabus':
            if self.vector_manager._syllabus_vectorizer is None:
                return []
            vec = self.vector_manager._syllabus_vectorizer.transform([query])
            sims = (self.vector_manager._syllabus_matrix @ vec.T).toarray().ravel()
        else:
            if self.vector_manager._reference_vectorizer is None:
                return []
            vec = self.vector_manager._reference_vectorizer.transform([query])
            sims = (self.vector_manager._reference_matrix @ vec.T).toarray().ravel()

        if sims.size == 0:
            return []
        idxs = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idxs]

    def hybrid_search(self, collection: str, query: str, k_dense: int = 5, k_sparse: int = 10, k_final: int = 5) -> List[Any]:
        """
        Hybrid search with reciprocal rank fusion: 1/(1+rank_dense) + 1/(1+rank_sparse).
        """
        self._log_operation("Hybrid search", f"collection={collection}, query='{query[:50]}...'")

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

    def _dense_search(self, collection: str, query: str, k: int):
        """Perform dense search using vector similarity."""
        # Get the vector store from the RAG system
        if hasattr(self, '_vector_stores'):
            store = self._vector_stores.get(collection)
            if store:
                try:
                    # Use similarity search to get documents
                    docs = store.similarity_search(query, k=k)
                    return docs
                except Exception as e:
                    self._log_error("Dense search", f"Failed for {collection}: {e}")
                    return []
        return []
