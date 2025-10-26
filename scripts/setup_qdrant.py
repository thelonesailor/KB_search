# !/usr/bin/env python3
"""
Qdrant setup and verification script
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.qdrant_manager import qdrant_manager
from config import settings


def setup_qdrant():
    """Initialize Qdrant and verify setup"""
    print("Setting up Wand AI with Qdrant...")

    try:
        # Health check
        if qdrant_manager.health_check():
            print("Qdrant health check passed")
        else:
            print("Qdrant health check failed")
            return False

        # Get collection info
        info = qdrant_manager.get_collection_info()
        print(f"Qdrant collection info: {info}")

        # Test vector operations
        test_vector = [0.1] * settings.vector_dimension
        results = qdrant_manager.search_similar_chunks(test_vector, limit=1)
        print(f"Vector search test completed: {len(results)} results")

        # Test hybrid search
        hybrid_results = qdrant_manager.hybrid_search(
            query_vector=test_vector,
            query_text="test",
            limit=1
        )
        print(f"Hybrid search test completed: {len(hybrid_results)} results")

        print("\nQdrant setup completed successfully!")
        return True

    except Exception as e:
        print(f"Qdrant setup failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_qdrant()
    sys.exit(0 if success else 1)