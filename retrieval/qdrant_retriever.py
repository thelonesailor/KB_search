from haystack import component
from haystack.dataclasses import Document
from typing import List, Optional, Dict, Any
from document_stores.qdrant_document_store import QdrantDocumentStore
import logging

logger = logging.getLogger(__name__)


@component
class QdrantEmbeddingRetriever:
    def __init__(self, document_store: QdrantDocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], top_k: int = 10):
        """Retrieve documents using vector similarity"""
        try:
            documents = self.document_store.query_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k
            )
            return {"documents": documents}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"documents": []}


@component
class QdrantHybridRetriever:
    def __init__(self, document_store: QdrantDocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], query: str, top_k: int = 10):
        """Retrieve documents using hybrid search"""
        try:
            documents = self.document_store.hybrid_query(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k
            )
            return {"documents": documents}
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return {"documents": []}