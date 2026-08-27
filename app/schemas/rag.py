from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the core components of this system?"}, description="The user's question or input")
    thread_id: str = Field(default="default_session", json_schema_extra={"example": "session_123"}, description="Session / thread ID for multi-turn LangGraph checkpointing")
    max_retries: Optional[int] = Field(default=3, ge=1, le=10, description="Max self-correcting retry loops")


class DocumentChunk(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = None
    is_relevant: Optional[bool] = None


class TraceStep(BaseModel):
    node: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    thread_id: str
    answer: str
    route_taken: str
    retries_count: int
    documents_used: List[DocumentChunk] = Field(default_factory=list)
    trace_steps: List[TraceStep] = Field(default_factory=list)
    hallucination_check: Optional[str] = None
    answer_quality_check: Optional[str] = None


class IngestTextRequest(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "Architecture Overview"})
    content: str = Field(..., json_schema_extra={"example": "This platform uses LangGraph with PostgreSQL checkpointing..."})
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    document_title: str
    total_collection_documents: int


class StateHistoryResponse(BaseModel):
    thread_id: str
    current_state: Dict[str, Any]
    checkpoint_history_count: int
