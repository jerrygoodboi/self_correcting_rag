from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import rag_graph
from app.schemas.rag import QueryResponse, DocumentChunk, TraceStep
from app.core.config import settings
from app.core.logging import logger


class RAGService:
    """Orchestrates queries through the compiled LangGraph workflow."""

    def __init__(self):
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = rag_graph()
        return self._graph

    def process_query(self, query: str, thread_id: str, max_retries: Optional[int] = None) -> QueryResponse:
        retries_limit = max_retries if max_retries is not None else settings.max_retries
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "query": query,
            "thread_id": thread_id,
            "chat_history": [HumanMessage(content=query)],
            "route": "vectorstore",
            "route_reasoning": None,
            "rewritten_query": None,
            "raw_documents": [],
            "filtered_documents": [],
            "generation": None,
            "doc_relevance": None,
            "hallucination_grade": None,
            "answer_grade": None,
            "retry_count": 0,
            "max_retries": retries_limit,
            "trace_logs": [],
        }

        logger.info(f"Invoking Agentic RAG graph for thread '{thread_id}' with query: '{query}'")
        final_state = self.graph.invoke(initial_state, config=config)

        # Map final state to QueryResponse
        docs_used: List[DocumentChunk] = []
        filtered_docs = final_state.get("filtered_documents", [])
        for doc in filtered_docs:
            docs_used.append(
                DocumentChunk(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    is_relevant=True,
                )
            )

        trace_steps: List[TraceStep] = []
        for log in final_state.get("trace_logs", []):
            trace_steps.append(
                TraceStep(
                    node=log.get("node", "unknown"),
                    action=log.get("action", "unknown"),
                    details=log.get("details", {}),
                )
            )

        return QueryResponse(
            query=query,
            thread_id=thread_id,
            answer=final_state.get("generation") or "No answer generated.",
            route_taken=final_state.get("route", "unknown"),
            retries_count=final_state.get("retry_count", 0),
            documents_used=docs_used,
            trace_steps=trace_steps,
            hallucination_check=final_state.get("hallucination_grade"),
            answer_quality_check=final_state.get("answer_grade"),
        )

    def get_state(self, thread_id: str) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config)
        return {
            "values": state_snapshot.values,
            "next": state_snapshot.next,
            "created_at": getattr(state_snapshot, "created_at", None),
        }


rag_service = RAGService()
