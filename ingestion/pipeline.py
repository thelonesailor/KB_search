from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from document_stores.qdrant_document_store import QdrantDocumentStore
from typing import List
import tiktoken
import logging

logger = logging.getLogger(__name__)


class AdvancedChunkingStrategy:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_by_structure(self, content: str, doc_type: str) -> List[Document]:
        """Document-based chunking respecting structure"""
        chunks = []

        if doc_type == "markdown":
            lines = content.split('\n')
            current_chunk = []
            current_heading = ""

            for line in lines:
                if line.startswith('#') and len(line.strip()) > 1:
                    if current_chunk:
                        chunks.append(Document(
                            content='\n'.join(current_chunk),
                            meta={"heading": current_heading, "chunk_type": "section"}
                        ))
                    current_chunk = [line]
                    current_heading = line
                else:
                    current_chunk.append(line)

            if current_chunk:
                chunks.append(Document(
                    content='\n'.join(current_chunk),
                    meta={"heading": current_heading, "chunk_type": "section"}
                ))

        elif doc_type == "code":
            lines = content.split('\n')
            current_chunk = []
            in_function = False

            for line in lines:
                if line.strip().startswith(('def ', 'class ')):
                    if current_chunk and in_function:
                        chunks.append(Document(
                            content='\n'.join(current_chunk),
                            meta={"chunk_type": "code_block"}
                        ))
                    current_chunk = [line]
                    in_function = True
                elif in_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    chunks.append(Document(
                        content='\n'.join(current_chunk),
                        meta={"chunk_type": "code_block"}
                    ))
                    current_chunk = []
                    in_function = False
                else:
                    current_chunk.append(line)

        else:
            chunks = self._recursive_chunk(content)

        return chunks

    def _recursive_chunk(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
        """Recursive chunking with overlap"""
        tokens = self.encoding.encode(content)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(Document(
                content=chunk_text,
                meta={"chunk_type": "recursive", "overlap": overlap}
            ))

        return chunks


class QdrantIngestionPipeline:
    def __init__(self, embedding_model: str):
        self.chunking_strategy = AdvancedChunkingStrategy()
        self.embedding_model = embedding_model

        # Create Qdrant document store
        self.document_store = QdrantDocumentStore()

        # Create pipeline with Qdrant writer
        self.pipeline = Pipeline()
        self.pipeline.add_component("splitter",
                                    DocumentSplitter(split_by="sentence", split_length=200, split_overlap=20))
        # Use local sentence-transformers embedder (no API key needed)
        self.pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=self.embedding_model))
        self.pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))

        self.pipeline.connect("splitter.documents", "embedder.documents")
        self.pipeline.connect("embedder.documents", "writer.documents")

    def ingest_documents(self, documents: List[Document], doc_type: str = "generic"):
        """Ingest documents with advanced chunking into Qdrant"""
        processed_docs = []

        for doc in documents:
            # Apply structure-aware chunking
            structured_chunks = self.chunking_strategy.chunk_by_structure(doc.content, doc_type)

            if not structured_chunks:
                structured_chunks = self.chunking_strategy._recursive_chunk(doc.content)

            # Add metadata to chunks
            for chunk in structured_chunks:
                chunk.meta.update({
                    "source": doc.meta.get("source", "unknown"),
                    "document_type": doc_type,
                    "original_id": doc.id
                })

            processed_docs.extend(structured_chunks)

        # Run through pipeline
        result = self.pipeline.run({"splitter": {"documents": processed_docs}})
        logger.info(f"Ingested {len(processed_docs)} chunks into Qdrant")
        return result