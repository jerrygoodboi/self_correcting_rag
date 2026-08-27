from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes.router import route_query_node
from app.agent.nodes.retriever import retrieve_node
from app.agent.nodes.doc_grader import grade_documents_node
from app.agent.nodes.rewriter import rewrite_query_node
from app.agent.nodes.generator import (
    generate_rag_answer_node,
    generate_direct_answer_node,
    generate_fallback_node,
)
from app.agent.nodes.hallucination_grader import grade_hallucination_node
from app.agent.nodes.answer_grader import grade_answer_node
from app.agent.checkpointer import get_checkpointer
from app.core.config import settings
from app.core.logging import logger


# Conditional routing functions
def route_decision_edge(state: AgentState) -> Literal["retriever", "generator_direct"]:
    route = state.get("route", "vectorstore")
    if route == "direct_answer":
        return "generator_direct"
    return "retriever"


def doc_grading_edge(state: AgentState) -> Literal["generator_rag", "rewriter", "generator_fallback"]:
    doc_relevance = state.get("doc_relevance", "no")
    filtered_docs = state.get("filtered_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if len(filtered_docs) > 0 and doc_relevance == "yes":
        return "generator_rag"
    
    if retry_count < max_retries:
        logger.info(f"[Graph Condition] Doc relevance is 'no'. Triggering rewrite loop (retry {retry_count + 1}/{max_retries})")
        return "rewriter"
    
    logger.warning(f"[Graph Condition] Doc relevance is 'no' and retry limit ({max_retries}) reached. Triggering fallback.")
    return "generator_fallback"


def hallucination_edge(state: AgentState) -> Literal["answer_grader", "rewriter", "END"]:
    hallucination_grade = state.get("hallucination_grade", "grounded")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if hallucination_grade == "grounded":
        return "answer_grader"
    
    if retry_count < max_retries:
        logger.info(f"[Graph Condition] Hallucination detected. Triggering rewrite loop (retry {retry_count + 1}/{max_retries})")
        return "rewriter"
    
    logger.warning(f"[Graph Condition] Grounding check failed and max retries ({max_retries}) reached.")
    return "END"


def answer_quality_edge(state: AgentState) -> Literal["END", "rewriter"]:
    answer_grade = state.get("answer_grade", "useful")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if answer_grade == "useful":
        return "END"
    
    if retry_count < max_retries:
        logger.info(f"[Graph Condition] Answer quality check failed. Triggering rewrite loop (retry {retry_count + 1}/{max_retries})")
        return "rewriter"
    
    logger.warning(f"[Graph Condition] Answer quality not optimal but max retries ({max_retries}) reached.")
    return "END"


def build_self_correcting_rag_graph():
    """Builds and compiles the Self-Correcting Agentic RAG workflow graph."""
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("router", route_query_node)
    builder.add_node("retriever", retrieve_node)
    builder.add_node("doc_grader", grade_documents_node)
    builder.add_node("rewriter", rewrite_query_node)
    builder.add_node("generator_rag", generate_rag_answer_node)
    builder.add_node("generator_direct", generate_direct_answer_node)
    builder.add_node("generator_fallback", generate_fallback_node)
    builder.add_node("hallucination_grader", grade_hallucination_node)
    builder.add_node("answer_grader", grade_answer_node)

    # Add Edges
    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "retriever": "retriever",
            "generator_direct": "generator_direct"
        }
    )

    builder.add_edge("retriever", "doc_grader")

    builder.add_conditional_edges(
        "doc_grader",
        doc_grading_edge,
        {
            "generator_rag": "generator_rag",
            "rewriter": "rewriter",
            "generator_fallback": "generator_fallback"
        }
    )

    builder.add_edge("rewriter", "retriever")

    builder.add_edge("generator_rag", "hallucination_grader")

    builder.add_conditional_edges(
        "hallucination_grader",
        hallucination_edge,
        {
            "answer_grader": "answer_grader",
            "rewriter": "rewriter",
            "END": END
        }
    )

    builder.add_conditional_edges(
        "answer_grader",
        answer_quality_edge,
        {
            "END": END,
            "rewriter": "rewriter"
        }
    )

    builder.add_edge("generator_direct", END)
    builder.add_edge("generator_fallback", END)

    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Compiled Self-Correcting Agentic RAG graph with checkpointer.")
    return graph


# Singleton graph instance
rag_graph = build_self_correcting_rag_graph
