from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class DocumentMetadata(BaseModel):
    source: str
    document_type: str
    security_level: str
    last_modified: datetime
    chunk_index: int
    total_chunks: int

class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

class QueryIntent(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    AMBIGUOUS = "ambiguous"

class QueryAnalysis(BaseModel):
    intent: QueryIntent
    sub_questions: List[str]
    required_data_elements: List[str]
    confidence: float

class RetrievalResult(BaseModel):
    chunks: List[DocumentChunk]
    scores: List[float]
    query_time_ms: int

class GenerationOutput(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    reasoning: Optional[str] = None

class ReflectionResult(BaseModel):
    is_complete: bool
    missing_elements: List[str]
    ambiguity_detected: bool
    clarifying_question: Optional[str] = None
    confidence_score: float

class AgentState(BaseModel):
    user_query: str
    query_analysis: Optional[QueryAnalysis] = None
    retrieval_result: Optional[RetrievalResult] = None
    generation_output: Optional[GenerationOutput] = None
    reflection_result: Optional[ReflectionResult] = None
    clarification_response: Optional[str] = None
    enriched_data: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    execution_trace: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    enrichment_suggestions: Optional[List[Dict[str, Any]]] = None
