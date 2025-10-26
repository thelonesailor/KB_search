from haystack import Document
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from typing import List, Optional, Dict, Any
from database.qdrant_manager import qdrant_manager
import uuid


class QdrantDocumentStore(DocumentStore):
    def __init__(self):
        self.qdrant_manager = qdrant_manager

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """Write documents to Qdrant"""
        chunks = []
        for doc in documents:
            chunk_data = {
                'content': doc.content,
                'metadata': doc.meta,
                'embedding': doc.embedding if hasattr(doc, 'embedding') and doc.embedding is not None else None,
                'id': doc.id or str(uuid.uuid4())
            }
            chunks.append(chunk_data)

        success = self.qdrant_manager.store_document_chunks(chunks)
        return len(chunks) if success else 0

    def query_by_embedding(self,
                           query_embedding: List[float],
                           filters: Optional[Dict[str, Any]] = None,
                           top_k: int = 10,
                           return_embedding: bool = False) -> List[Document]:
        """Query documents by embedding similarity"""
        results = self.qdrant_manager.search_similar_chunks(
            query_vector=query_embedding,
            limit=top_k,
            filters=filters
        )

        documents = []
        for result in results:
            doc = Document(
                content=result['content'],
                meta=result['metadata'],
                embedding=result.get('embedding') if return_embedding else None,
                id=result['id']
            )
            documents.append(doc)

        return documents

    def hybrid_query(self,
                     query_embedding: List[float],
                     query_text: str,
                     filters: Optional[Dict[str, Any]] = None,
                     top_k: int = 10) -> List[Document]:
        """Hybrid query using both vector and keyword search"""
        results = self.qdrant_manager.hybrid_search(
            query_vector=query_embedding,
            query_text=query_text,
            limit=top_k,
            filters=filters
        )

        documents = []
        for result in results:
            doc = Document(
                content=result['content'],
                meta=result['metadata'],
                id=result['id']
            )
            documents.append(doc)

        return documents

    def get_document_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get document count"""
        info = self.qdrant_manager.get_collection_info()
        return info.get('vectors_count', 0)

    def get_all_documents(self) -> List[Document]:
        """Get all documents (use with caution for large collections)"""
        try:
            all_points, _ = self.qdrant_manager.client.scroll(
                collection_name=self.qdrant_manager.collection_name,
                limit=10000
            )

            documents = []
            for point in all_points:
                doc = Document(
                    content=point.payload.get('content', ''),
                    meta=point.payload.get('metadata', {}),
                    id=point.id
                )
                documents.append(doc)

            return documents
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get all documents: {e}")
            return []

    def delete_documents(self, filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete documents matching filters"""
        if filters:
            self.qdrant_manager.delete_points_by_filter(filters)