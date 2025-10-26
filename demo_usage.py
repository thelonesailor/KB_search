"""
Demo script showing how to use the Advanced RAG system
"""
from agents.enrichment import EnrichmentOrchestrator
from retrieval.core import AdvancedRAGCore


def demo_advanced_rag():
    """Demonstrate the Advanced RAG system capabilities"""

    # Initialize system (simplified)
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    rag_core = AdvancedRAGCore(document_store)
    orchestrator = EnrichmentOrchestrator(rag_core)

    # Test queries that should trigger different enrichment paths

    test_queries = [
        # Straightforward factual query
        "What is our company's vacation policy?",

        # Ambiguous query that should trigger clarification
        "Tell me about Q3 performance",

        # Complex query that might need data enrichment
        "What were the sales figures for the European region last quarter and how do they compare to projections?",
    ]

    print("=== Wand AI Advanced RAG System Demo ===\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")

        result = orchestrator.process_query(query)

        print(f"Final Answer: {result.final_answer}")
        print(f"Confidence: {result.generation_output.confidence if result.generation_output else 'N/A'}")
        print(f"Sources: {result.generation_output.sources if result.generation_output else []}")
        print(f"Enrichment Triggered: {result.enriched_data is not None}")
        print(f"Clarification Triggered: {result.clarification_response is not None}")
        print(f"Execution Trace: {result.execution_trace}")

        print("-" * 80)


if __name__ == "__main__":
    demo_advanced_rag()
