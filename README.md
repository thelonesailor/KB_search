# Advanced RAG System with Agentic Enrichment

A production-ready Retrieval-Augmented Generation (RAG) system with intelligent completeness checking and enrichment suggestions. It uses a vector database for storage and retrieval, an OpenAI-compatible LLM API for generation and reflection, and an agentic workflow to control quality.

## 🎯 Core Features
- Document upload (PDF, TXT, MD, JSON, CSV) with chunking and vector storage
- Natural language querying with grounded answers and source citations
- Reflection-based completeness detection and actionable enrichment suggestions
- Simple web UI (drag-and-drop upload, ask questions, view results)
- Maintenance endpoints (health, stats, list documents, cleanup by source)

---

## 🎨 Design Decisions

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

4) Agentic Orchestration (Workflow with Conditional Routing)
- Why: Move beyond linear pipelines to handle ambiguity, retries, and enrichment
- Impact: Nodes for analysis → RAG → reflection → conditional route (complete/ambiguous/incomplete/retry)
- Outcome: Clear, maintainable control flow with instrumentation (execution trace)

5) Static Web UI Served by Backend
- Why: Zero extra services; works out-of-the-box
- Impact: One process serves API and UI; static assets under a single origin
- Outcome: Simple deployment and testing experience

---

## ⚖️ Trade-offs Due to the 24h Constraint

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

## 🚀 How to Run

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

## ✅ How to Test

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

## 📦 Project Summary

- Local embeddings + vector DB deliver fast, private retrieval
- Agentic reflection improves answer completeness and user guidance
- Single-process API + static UI keeps running/testing simple
- Clear maintenance endpoints for visibility and control
