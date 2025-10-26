from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack import Pipeline
# Note: SentenceTransformers downloads models from HuggingFace on first use
# The model is cached locally (~90MB) and won't re-download
from retrieval.qdrant_retriever import QdrantHybridRetriever
from document_stores.qdrant_document_store import QdrantDocumentStore
from clients.perplexity_client import perplexity_client
from config import settings
import time
import logging
import os

logger = logging.getLogger(__name__)


class QdrantRAGCore:
    def __init__(self, model: str = None):
        self.model = model or settings.chat_model
        self.client = perplexity_client.get_client()

        # Use Qdrant document store
        self.document_store = QdrantDocumentStore()

        # Build the RAG pipeline with correct components
        self.pipeline = Pipeline()

        # Set environment variables for Haystack's OpenAI components to use Perplexity (for generation only)
        os.environ["OPENAI_API_KEY"] = settings.perplexity_api_key
        os.environ["OPENAI_BASE_URL"] = settings.perplexity_base_url

        # Query embedding - use local sentence-transformers (no API key needed)
        self.pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(
            model=settings.embedding_model
        ))

        # Hybrid retriever
        self.pipeline.add_component("retriever", QdrantHybridRetriever(document_store=self.document_store))

        # Prompt builder
        prompt_template = """
        You are an enterprise knowledge assistant. Your responses MUST be strictly based on the provided context.

        CONTEXT:
        {% for document in documents %}
        Source: {{ document.meta.source }}, Chunk: {{ document.meta.get('chunk_index', 'N/A') }}
        Content: {{ document.content }}
        {% endfor %}

        USER QUERY: {{ query }}

        INSTRUCTIONS:
        1. Answer ONLY using information from the provided context
        2. If the context doesn't contain sufficient information, state what is missing
        3. Cite sources for every factual claim using format: [Source: {source_name}]
        4. Never fabricate or hallucinate information
        5. If uncertain, explicitly state the limitations

        RESPONSE:
        """

        self.pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))

        # Generator configured for Perplexity - let it discover API key from environment
        self.pipeline.add_component("generator", OpenAIGenerator(
            model=self.model,
            generation_kwargs={
                "temperature": 0.1,
                "max_tokens": 1000
            }
        ))

        # Connect components
        self.pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.prompt")

    def retrieve_and_generate(self, query: str, query_analysis=None):
        """Enhanced retrieval with Qdrant's capabilities"""
        start_time = time.time()

        enhanced_query = self._enhance_query(query, query_analysis)

        # Execute pipeline with hybrid search
        result = self.pipeline.run({
            "query_embedder": {"text": enhanced_query},
            "retriever": {"query": query, "top_k": 10},
            "prompt_builder": {"query": query}
        })

        execution_time = int((time.time() - start_time) * 1000)

        # Extract sources from retrieved documents
        sources = []
        if "retriever" in result and "documents" in result["retriever"]:
            for doc in result["retriever"]["documents"]:
                source = doc.meta.get("source", "unknown")
                if source and source != "unknown":
                    sources.append(source)

        # Remove duplicates while preserving order
        sources = list(dict.fromkeys(sources))

        generation_text = result["generator"]["replies"][0] if result["generator"][
            "replies"] else "No response generated."
        confidence = self._calculate_confidence(generation_text, sources)

        logger.info(f"Retrieved {len(sources)} sources: {sources}")

        return {
            "answer": generation_text,
            "sources": sources,
            "confidence": confidence,
            "execution_time_ms": execution_time
        }

    def _enhance_query(self, query: str, query_analysis) -> str:
        """Enhanced query processing"""
        if not query_analysis:
            return query

        enhanced = query

        # Handle both dict and object access patterns
        if isinstance(query_analysis, dict):
            sub_questions = query_analysis.get('sub_questions', [])
            required_data = query_analysis.get('required_data_elements', [])
        else:
            sub_questions = getattr(query_analysis, 'sub_questions', [])
            required_data = getattr(query_analysis, 'required_data_elements', [])

        if sub_questions:
            enhanced += " " + " ".join(sub_questions)

        if required_data:
            enhanced += " Relevant data: " + ", ".join(required_data)

        return enhanced

    def _calculate_confidence(self, generation: str, sources: list) -> float:
        """Confidence scoring"""
        if not generation or generation.strip() == "":
            return 0.0

        uncertainty_phrases = [
            "I don't know", "information is missing", "not in the context",
            "unable to find", "no information provided"
        ]

        uncertainty_score = 0.0
        for phrase in uncertainty_phrases:
            if phrase.lower() in generation.lower():
                uncertainty_score += 0.2

        source_confidence = min(len(sources) / 5.0, 1.0)
        length_confidence = min(len(generation.split()) / 50.0, 1.0)

        final_confidence = (0.4 * source_confidence +
                            0.4 * (1 - uncertainty_score) +
                            0.2 * length_confidence)

        return max(0.0, min(1.0, final_confidence))