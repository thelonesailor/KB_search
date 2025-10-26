# database/qdrant_manager.py
import qdrant_client
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid
import json
import logging
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    def __init__(self):
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        self.vector_dimension = settings.vector_dimension
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            if settings.qdrant_api_key:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
            else:
                self.client = QdrantClient(settings.qdrant_url)

            logger.info(f"Qdrant client initialized for {settings.qdrant_url}")
            self._ensure_collection()

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE
                    ),
                    # Optimize for hybrid search
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=50000
                    ),
                    # Enable payload indexing for metadata filtering
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def store_document_chunks(self, chunks: List[Dict]) -> bool:
        """Store document chunks in Qdrant"""
        try:
            points = []
            for chunk in chunks:
                point_id = str(uuid.uuid4())

                point = PointStruct(
                    id=point_id,
                    vector=chunk.get('embedding', []),
                    payload={
                        "content": chunk.get('content', ''),
                        "metadata": chunk.get('metadata', {}),
                        "chunk_id": chunk.get('id', point_id),
                        "source": chunk.get('metadata', {}).get('source', 'unknown'),
                        "document_type": chunk.get('metadata', {}).get('document_type', 'generic'),
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)

            # Batch upsert points
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )

            logger.info(f"Stored {len(points)} chunks in Qdrant")
            return operation_info.status == 'completed'

        except Exception as e:
            logger.error(f"Failed to store document chunks: {e}")
            return False

    def search_similar_chunks(self,
                              query_vector: List[float],
                              limit: int = 10,
                              score_threshold: float = 0.7,
                              filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        try:
            # Build search filters
            search_filters = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}" if key != "source" else "source",
                            match=MatchValue(value=value)
                        )
                    )
                search_filters = Filter(must=conditions)

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filters,
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'metadata': result.payload.get('metadata', {}),
                    'score': result.score,
                    'source': result.payload.get('source', 'unknown')
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def hybrid_search(self,
                      query_vector: List[float],
                      query_text: str,
                      limit: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Hybrid search combining vector and keyword search"""
        try:
            # First, do vector search
            vector_results = self.search_similar_chunks(
                query_vector=query_vector,
                limit=limit * 2,  # Get more results for re-ranking
                filters=filters
            )

            # Simple keyword matching in content
            keyword_results = []
            if query_text:
                # This is a simplified keyword search - in production, use proper text indexing
                all_points = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000  # Adjust based on collection size
                )[0]

                for point in all_points:
                    content = point.payload.get('content', '').lower()
                    if any(word in content for word in query_text.lower().split()):
                        keyword_results.append({
                            'id': point.id,
                            'content': point.payload.get('content', ''),
                            'metadata': point.payload.get('metadata', {}),
                            'score': 0.5,  # Default score for keyword matches
                            'source': point.payload.get('source', 'unknown')
                        })

            # Combine and deduplicate results
            all_results = {}
            for result in vector_results + keyword_results:
                if result['id'] not in all_results or result['score'] > all_results[result['id']]['score']:
                    all_results[result['id']] = result

            # Sort by score and return top results
            sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
            return sorted_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.search_similar_chunks(query_vector, limit, filters=filters)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and information"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            collection_stats = self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": getattr(collection_stats, 'points_count', 0),
                "status": collection_info.status,
                "dimension": self.vector_dimension
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> bool:
        """Check Qdrant health"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def delete_points_by_filter(self, filters: Dict) -> int:
        """Delete points matching filters"""
        try:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}" if key != "source" else "source",
                        match=MatchValue(value=value)
                    )
                )

            delete_filter = Filter(must=conditions)

            # First, count the points that will be deleted
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=delete_filter,
                limit=10000,  # Large limit to get all matching points
                with_payload=False,
                with_vectors=False
            )

            points_to_delete = scroll_result[0]
            count = len(points_to_delete)

            # Now perform the deletion
            if count > 0:
                result = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=delete_filter,
                    wait=True
                )
                logger.info(f"Deleted {count} points with filters: {filters}")
            else:
                logger.info(f"No points found matching filters: {filters}")

            return count

        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return 0


# Global Qdrant manager instance
qdrant_manager = QdrantManager()