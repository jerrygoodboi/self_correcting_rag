from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.llm_factory import get_chat_llm
from app.agent.state import AgentState
from app.agent.prompts.rewriting import QUERY_REWRITER_SYSTEM_PROMPT
from app.schemas.evaluation import QueryRewrite
from app.core.config import settings
from app.core.logging import logger


def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Rewrites and enriches the query to improve semantic vector retrieval.
    Increments retry_count.
    """
    original_query = state.get("query", "")
    current_rewritten = state.get("rewritten_query")
    query_to_rewrite = current_rewritten or original_query
    retry_count = state.get("retry_count", 0) + 1
    doc_relevance = state.get("doc_relevance", "unknown")
    hallucination_grade = state.get("hallucination_grade", "unknown")
    answer_grade = state.get("answer_grade", "unknown")

    logger.info(f"[Rewriter Node] (Retry #{retry_count}) Rewriting query: '{query_to_rewrite}'")

    llm = get_chat_llm(temperature=0.2)
    structured_rewriter = llm.with_structured_output(QueryRewrite)

    failure_context = (
        f"- Original Query: {original_query}\n"
        f"- Previous Query Formulation: {query_to_rewrite}\n"
        f"- Document Relevance Status: {doc_relevance}\n"
        f"- Grounding Status: {hallucination_grade}\n"
        f"- Answer Usefulness Status: {answer_grade}\n"
    )

    messages = [
        SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=f"Context of Failure:\n{failure_context}\n\nPlease reformulate an improved search query:")
    ]

    try:
        result: QueryRewrite = structured_rewriter.invoke(messages)
        new_query = result.improved_query
        reasoning = result.reasoning
    except Exception as e:
        logger.error(f"[Rewriter Node] Fallback rewrite due to error: {e}")
        new_query = f"{original_query} details specifications overview"
        reasoning = f"Fallback keyword expansion: {e}"

    logger.info(f"[Rewriter Node] New Query: '{new_query}' | Reason: {reasoning}")

    return {
        "rewritten_query": new_query,
        "retry_count": retry_count,
        "trace_logs": [{
            "node": "rewriter",
            "action": "rewrite_query",
            "details": {
                "retry_count": retry_count,
                "previous_query": query_to_rewrite,
                "new_query": new_query,
                "reasoning": reasoning
            }
        }]
    }
