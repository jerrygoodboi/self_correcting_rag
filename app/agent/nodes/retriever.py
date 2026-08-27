from typing import Any, Dict
from app.agent.state import AgentState
from app.services.vector_service import vector_service
from app.core.logging import logger


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that retrieves relevant document chunks from vectorstore using the current query
    (or rewritten query if available).
    """
    query = state.get("rewritten_query") or state.get("query", "")
    logger.info(f"[Retrieve Node] Executing retrieval for: '{query}'")

    v_service = vector_service()
    docs = v_service.similarity_search(query)

    return {
        "raw_documents": docs,
        "trace_logs": [{
            "node": "retriever",
            "action": "retrieve_documents",
            "details": {
                "query_used": query,
                "retrieved_count": len(docs),
                "snippets": [d.page_content[:100] + "..." for d in docs]
            }
        }]
    }
