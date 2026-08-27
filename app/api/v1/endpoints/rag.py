from fastapi import APIRouter, HTTPException
from app.schemas.rag import QueryRequest, QueryResponse, StateHistoryResponse
from app.services.rag_service import rag_service
from app.core.logging import logger

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Execute Self-Correcting RAG Query")
def query_rag(request: QueryRequest):
    """
    Executes a query through the Self-Correcting Agentic RAG graph:
    1. Query understanding & routing
    2. Vectorstore retrieval
    3. Document relevance grading
    4. Iterative query rewriting if relevance/grounding fails
    5. Grounded synthesis & hallucination grading
    6. State checkpointed in PostgreSQL
    """
    try:
        response = rag_service.process_query(
            query=request.query,
            thread_id=request.thread_id,
            max_retries=request.max_retries,
        )
        return response
    except Exception as e:
        logger.exception(f"Failed to process RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{thread_id}", summary="Get LangGraph Checkpoint State")
def get_session_state(thread_id: str):
    """Retrieves current session values and next steps from the LangGraph checkpointer."""
    try:
        state_data = rag_service.get_state(thread_id)
        return {
            "thread_id": thread_id,
            "state": state_data,
        }
    except Exception as e:
        logger.exception(f"Failed to get state for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
