# Advanced RAG System with Agentic Enrichment

A Retrieval-Augmented Generation (RAG) system with intelligent completeness checking and enrichment suggestions. It uses a vector database for storage and retrieval, an OpenAI-compatible LLM API for generation and reflection, and an agentic workflow to control quality.

## Core Features
- Document upload (PDF, TXT, MD, JSON, CSV) with chunking and vector storage
- Natural language querying with grounded answers and source citations
- Reflection-based completeness detection and actionable enrichment suggestions
- Simple web UI (drag-and-drop upload, ask questions, view results)
- Maintenance endpoints (health, stats, list documents, cleanup by source)

---

## Design Decisions

1) Local Embeddings (Sentence Transformers)
- Why: Control, cost-efficiency, privacy; no external embedding API required
- Impact: Deterministic, fast inference; 384–768 dims depending on model
- Outcome: Stable, zero-cost embeddings after model caching

2) Vector Database (Qdrant)
- Why: Production-ready, fast, hybrid-friendly, rich payload filtering
- Impact: store document chunks with metadata for precise cleanup and source tracking
- Outcome: Scalable retrieval with chunk-level management

3) OpenAI-Compatible LLM for Generation and Reflection
- Why: High-quality generation, easy client integration via OpenAI-compatible API
- Impact: Generation and reflection are decoupled from embeddings; reflection checks completeness/ambiguity
- Outcome: Better answer reliability with minimal architecture complexity

4) Haystack for Document Processing Pipeline
- Why: Robust, modular framework for building RAG pipelines with pre-built components
- Framework: Haystack provides document converters, preprocessors, and pipeline abstractions
- Impact: Handles diverse file formats (PDF, TXT, MD, JSON, CSV) with minimal custom code
- Outcome: Reliable document ingestion with chunking, metadata extraction, and error handling

5) Agentic Orchestration with LangGraph
- Why: Move beyond linear pipelines to handle ambiguity, retries, and enrichment using state machine graphs
- Framework: LangGraph's StateGraph enables declarative workflow definition with conditional routing
- Impact: Nodes for analysis → RAG → reflection → conditional route (complete/ambiguous/incomplete/retry)
- Outcome: Clear, maintainable control flow with instrumentation (execution trace) and easy debugging of agent states

6) Static Web UI Served by Backend
- Why: Zero extra services; works out-of-the-box
- Impact: One process serves API and UI; static assets under a single origin
- Outcome: Simple deployment and testing experience

---

## Trade-offs Due to the 24h Constraint

What I prioritized
- End-to-end path: ingest → retrieve → generate → reflect → respond
- Practical completeness checks and helpful enrichment suggestions
- Robust file handling (PDF text extraction; safe fallbacks for others)
- Useful maintenance endpoints (health, stats, list docs, cleanup)

What I simplified or deferred
- Security: broad CORS, no auth (add OAuth/JWT + RBAC for production)
- Chunking: pragmatic chunking over fully semantic/structure-aware segmentation
- Evaluation: light placeholders; full RAG evaluation dashboards deferred
- Observability: basic logging; centralized log/metrics/tracing not included
- Scalability hardening: no background workers or queue; no caching/rate limiting
- UI: clean and functional; no streaming tokens, session history, or theming

Why these choices
- Deliver correctness and usability within 24 hours
- Keep operating costs low (local embeddings) while ensuring answer quality (reflection)
- Provide a simple, testable UI and REST contract for rapid feedback

---

## How to Run

Prerequisites
- Python 3.13+
- Docker (for Qdrant)

Environment
Create a api-key.txt file in project root with the perplexity api key.

Start Qdrant (Docker)
-  docker compose -f docker-compose.qdrant.yml up -d  (macOS, for other OS use docker-compose)

Install dependencies
- pip install -r requirements.txt

Run the backend (serves API + UI)
- python3 main.py
- Open http://localhost:8000/ in your browser

---

## How to Test

From the UI
- Upload documents (drag-and-drop or click) in supported formats
- Ask a question in natural language
- Review the answer, confidence, completeness, sources, missing info, and suggestions
- Use cleanup to remove chunks by source if needed

From the API (examples)
- Health: GET http://localhost:8000/qdrant/health
- Stats: GET http://localhost:8000/qdrant/stats
- List Documents: GET http://localhost:8000/documents
- Ingest:
  curl -X POST "http://localhost:8000/ingest?doc_type=generic" \
    -F "files=@/path/to/file1.pdf" -F "files=@/path/to/file2.txt"
- Query:
  curl -X POST "http://localhost:8000/query?query=$(python -c 'import urllib.parse;print(urllib.parse.quote(\"What is the patient name?\"))')"
- Cleanup by source (deletes chunks for a file name):
  curl -X DELETE "http://localhost:8000/qdrant/cleanup?source=comprehensive-patient-data.txt"

Expected behavior
- Answers are grounded; sources appear either as a list or inline citation
- Reflection marks simple factual answers complete; enrichment suggestions focus on substantive gaps
- Cleanup responses report “Deleted N chunks from source: …”

---

## Project Summary

- Local embeddings + vector DB deliver fast, private retrieval
- Haystack handles robust document processing across multiple formats
- LangGraph orchestrates agentic workflows with state management and conditional routing
- Agentic reflection improves answer completeness and user guidance
- Single-process API + static UI keeps running/testing simple
- Clear maintenance endpoints for visibility and control

---

## LangGraph Workflow

This project uses LangGraph to orchestrate the agentic RAG flow as a small state machine. LangGraph provides a `StateGraph` that:
- Defines nodes (steps) and edges (transitions)
- Carries a shared state through the graph (`AgentState`)
- Supports conditional routing and controlled loopbacks
- Compiles to a runnable that we invoke per query

### Where it’s implemented
- Orchestrator and graph: `agents/enrichment.py`
  - Graph assembly: `EnrichmentOrchestrator._build_workflow()`
  - Graph execution entry point: `EnrichmentOrchestrator.process_query()`
- API integration: `main.py`
  - Built at startup in `startup_event()` and stored globally
  - Invoked from the `/query` endpoint
- State model: `models.py`
  - `AgentState` describes what flows between nodes

### The shared state (`AgentState`)
`models.py` defines the Pydantic model. Key fields used by the graph:
- `user_query`: original user query
- `query_analysis`: intent, sub‑questions, required data elements
- `generation_output`: model answer, sources, confidence
- `reflection_result`: completeness, ambiguity, missing elements
- `clarification_response`: simulated user clarification text
- `enriched_data`: simulated dynamic augmentation content
- `final_answer`: final string returned to clients
- `execution_trace`: list of trace messages for debugging
- `retry_count`, `enrichment_suggestions`

Note: Although we declare `AgentState` as the graph state type, LangGraph returns a plain dict at runtime. The API normalizes both dict and object forms before formatting the response.

### Graph construction (nodes and edges)
The graph is created in `_build_workflow()`:

High‑level flow:
```
user query
  ↓
analyze_query ──→ execute_rag ──┬─ reflect_on_output ──┬─ complete ─→ generate_final_answer → END
                                │                      ├─ ambiguous ─→ handle_ambiguity ─┐
                                │                      ├─ incomplete ─→ enrich_data ─────┤
                                │                      └── retry ────────────────────────┘
                                └─ skip (high confidence) ─────────────→ generate_final_answer → END
```

### What each node does
- `analyze_query`: Uses Perplexity to infer intent/sub‑questions; writes `query_analysis` and traces.
- `execute_rag`: Calls `QdrantRAGCore.retrieve_and_generate()`; writes `generation_output` (answer, sources, confidence) and increments `retry_count`.
- `reflect_on_output`: Uses a reasoning model to set `reflection_result` (`is_complete`, `ambiguity_detected`, `missing_elements`, `confidence_score`).
- `handle_ambiguity`: Simulates asking the user a clarifying question, sets a `clarification_response`, then loops back to re‑execute RAG.
- `enrich_data`: Simulates dynamic augmentation for the top missing elements and stores them in `enriched_data` before re‑running RAG.
- `generate_final_answer`: Produces `final_answer` (optionally appends enriched data) and creates `enrichment_suggestions` to guide the user.

### Conditional routing rules
- `_should_reflect(state) -> {"reflect"|"skip"}`
  - If `confidence > 0.7` and there are sources, skip reflection and finalize.
  - Otherwise reflect on the output first.
- `_route_after_reflection(state) -> {complete|ambiguous|incomplete|retry}`
  - Chooses a path using `is_complete`,a `confidence_score`, `ambiguity_detected`, `missing_elements`, and `retry_count`.

### How execution is triggered
- At startup (`main.py`), the app builds `QdrantRAGCore`, then instantiates `EnrichmentOrchestrator`, which compiles the LangGraph.
- On `/query`, the API calls `orchestrator.process_query(query)`:
- `recursion_limit` increases the maximum call depth to allow at most a small number of loopbacks.
- The result is a dict‑like state; the API extracts fields (`answer`, `sources`, `confidence`, `missing_info`, `enrichment_suggestions`, etc.) for the HTTP response.
