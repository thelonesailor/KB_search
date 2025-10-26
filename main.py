from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from database.qdrant_manager import qdrant_manager
from ingestion.pipeline import QdrantIngestionPipeline
from retrieval.core import QdrantRAGCore
from config import settings

app = FastAPI(title="Wand AI Advanced RAG System with Qdrant", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components
rag_core = None
orchestrator = None
evaluator = None
ingestion_pipeline = None

load_dotenv()


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system with Qdrant"""
    global rag_core, orchestrator, evaluator, ingestion_pipeline

    try:
        # Initialize Qdrant
        if not qdrant_manager.health_check():
            raise Exception("Qdrant health check failed")

        print("Qdrant initialized successfully")

        # Initialize ingestion pipeline
        ingestion_pipeline = QdrantIngestionPipeline(embedding_model=settings.embedding_model)

        # Initialize RAG core with Qdrant
        rag_core = QdrantRAGCore()

        # Initialize orchestrator
        from agents.enrichment import EnrichmentOrchestrator
        orchestrator = EnrichmentOrchestrator(rag_core)

        # Initialize evaluator
        from evaluation.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator()

        print("Advanced RAG System with Qdrant initialized successfully")

    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        raise


@app.post("/query")
async def process_query(query: str, evaluate: bool = False):
    """Main query endpoint"""
    try:
        # Import here to avoid circular imports
        from agents.enrichment import EnrichmentOrchestrator

        if orchestrator is None:
            raise HTTPException(status_code=500, detail="System not initialized")

        result_state = orchestrator.process_query(query)

        # LangGraph returns state as a dictionary, not the Pydantic model
        if isinstance(result_state, dict):
            generation_output = result_state.get('generation_output', {})
            final_answer = result_state.get('final_answer', 'No answer generated')
            execution_trace = result_state.get('execution_trace', [])
            retry_count = result_state.get('retry_count', 0)
            enriched_data = result_state.get('enriched_data')
            clarification_response = result_state.get('clarification_response')
            reflection_result = result_state.get('reflection_result', {})
        else:
            # Fallback for if it's actually an object
            generation_output = getattr(result_state, 'generation_output', {})
            final_answer = getattr(result_state, 'final_answer', 'No answer generated')
            execution_trace = getattr(result_state, 'execution_trace', [])
            retry_count = getattr(result_state, 'retry_count', 0)
            enriched_data = getattr(result_state, 'enriched_data', None)
            clarification_response = getattr(result_state, 'clarification_response', None)
            reflection_result = getattr(result_state, 'reflection_result', {})

        # Extract sources and confidence from generation_output
        if isinstance(generation_output, dict):
            sources = generation_output.get('sources', [])
            rag_confidence = generation_output.get('confidence', 0.0)
        else:
            sources = getattr(generation_output, 'sources', [])
            rag_confidence = getattr(generation_output, 'confidence', 0.0)

        # Extract missing_info from reflection_result
        if isinstance(reflection_result, dict):
            missing_elements = reflection_result.get('missing_elements', [])
            reflection_confidence = reflection_result.get('confidence_score', rag_confidence)
            is_complete = reflection_result.get('is_complete', True)
        else:
            missing_elements = getattr(reflection_result, 'missing_elements', [])
            reflection_confidence = getattr(reflection_result, 'confidence_score', rag_confidence)
            is_complete = getattr(reflection_result, 'is_complete', True)

        # Extract enrichment suggestions
        if isinstance(result_state, dict):
            enrichment_suggestions = result_state.get('enrichment_suggestions', [])
        else:
            enrichment_suggestions = getattr(result_state, 'enrichment_suggestions', [])

        response_data = {
            "query": query,
            # Primary response fields
            "answer": final_answer or "No answer generated",
            "confidence": reflection_confidence if reflection_result else rag_confidence,
            "missing_info": missing_elements,

            # Supporting information
            "sources": sources,
            "is_complete": is_complete,

            # Enrichment guidance
            "enrichment_suggestions": enrichment_suggestions,

            # Additional context
            "enrichment_triggered": enriched_data is not None,
            "clarification_triggered": clarification_response is not None,

            # Legacy field for backward compatibility
            "final_answer": final_answer or "No answer generated"
        }

        # Evaluation would be implemented here
        if evaluate and generation_output:
            response_data["evaluation"] = "Evaluation would be implemented here"

        return JSONResponse(content=response_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_query: {error_details}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...), doc_type: str = "generic"):
    """Ingest documents into Qdrant"""
    try:
        from haystack import Document
        import io

        documents = []
        for file in files:
            content_bytes = await file.read()
            filename = file.filename.lower()

            # Extract text based on file type
            try:
                if filename.endswith('.pdf'):
                    # Handle PDF files
                    from pypdf import PdfReader
                    pdf_file = io.BytesIO(content_bytes)
                    pdf_reader = PdfReader(pdf_file)

                    # Extract text from all pages
                    text_content = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

                    if not text_content.strip():
                        raise ValueError(f"No text could be extracted from PDF: {file.filename}")

                    content = text_content

                elif filename.endswith(('.txt', '.md', '.json', '.csv')):
                    # Handle text files
                    content = content_bytes.decode('utf-8')

                else:
                    # Try UTF-8 decode for unknown types
                    try:
                        content = content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported file type: {file.filename}. Supported types: PDF, TXT, MD, JSON, CSV"
                        )

                document = Document(
                    content=content,
                    meta={
                        "source": file.filename,
                        "document_type": doc_type,
                        "file_type": filename.split('.')[-1] if '.' in filename else 'unknown'
                    }
                )
                documents.append(document)

            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process {file.filename}: {str(e)}"
                )

        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents to ingest")

        # Ingest documents using Qdrant pipeline
        result = ingestion_pipeline.ingest_documents(documents, doc_type)

        return {
            "message": f"Successfully ingested {len(documents)} documents into Qdrant",
            "files": [doc.meta["source"] for doc in documents]
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Document ingestion error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@app.get("/qdrant/health")
async def qdrant_health():
    """Qdrant health check endpoint"""
    is_healthy = qdrant_manager.health_check()
    if is_healthy:
        return {"status": "healthy", "database": "Qdrant connected"}
    else:
        raise HTTPException(status_code=503, detail="Qdrant connection failed")


@app.get("/qdrant/stats")
async def qdrant_stats():
    """Get Qdrant collection statistics"""
    stats = qdrant_manager.get_collection_info()
    return stats


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        # Get all points with their metadata
        points = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]

        # Extract unique documents with their metadata
        documents_map = {}
        for point in points:
            if point.payload:
                source = point.payload.get('source', 'Unknown')
                doc_type = point.payload.get('document_type', 'generic')
                file_type = point.payload.get('file_type', 'txt')

                # Extract file extension from source if file_type is unknown
                if file_type == 'unknown' and source != 'Unknown':
                    if '.' in source:
                        file_type = source.rsplit('.', 1)[1].lower()
                    else:
                        file_type = 'txt'

                if source not in documents_map:
                    documents_map[source] = {
                        'source': source,
                        'document_type': doc_type,
                        'file_type': file_type,
                        'chunk_count': 0
                    }
                documents_map[source]['chunk_count'] += 1

        documents = list(documents_map.values())
        return {"documents": documents, "total": len(documents)}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error listing documents: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/qdrant/cleanup")
async def cleanup_qdrant(source: Optional[str] = None):
    """Clean up documents by source"""
    if source:
        deleted_count = qdrant_manager.delete_points_by_filter({"source": source})
        return {"message": f"Deleted {deleted_count} chunks from source: {source}"}
    else:
        return {"message": "Please specify a source to clean up"}


@app.get("/documents")
async def list_documents():
    """Get list of all uploaded documents with metadata"""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        # Get all points with their metadata
        points = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]

        # Extract unique documents by source
        documents_dict = {}
        for point in points:
            source = point.payload.get('source', 'Unknown')
            if source not in documents_dict:
                documents_dict[source] = {
                    'name': source,
                    'type': point.payload.get('document_type', 'generic'),
                    'file_type': point.payload.get('file_type', 'unknown'),
                    'chunks': 1
                }
            else:
                documents_dict[source]['chunks'] += 1

        # Convert to list and sort by name
        documents = sorted(documents_dict.values(), key=lambda x: x['name'])

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error listing documents: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/")
async def root():
    """Redirect to frontend"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.get("/style.css")
async def get_style():
    """Serve CSS file"""
    from fastapi.responses import FileResponse
    return FileResponse("static/style.css")

@app.get("/app.js")
async def get_app_js():
    """Serve JavaScript file"""
    from fastapi.responses import FileResponse
    return FileResponse("static/app.js")

@app.get("/api")
async def api_root():
    """API info endpoint"""
    qdrant_info = qdrant_manager.get_collection_info()
    return {
        "system": "Wand AI Advanced RAG with Qdrant",
        "status": "operational",
        "qdrant": qdrant_info
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)