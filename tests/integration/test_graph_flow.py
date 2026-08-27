import pytest
from app.agent.state import AgentState
from app.agent.graph import build_self_correcting_rag_graph
from langchain_core.messages import HumanMessage


def test_graph_compilation():
    graph = build_self_correcting_rag_graph()
    assert graph is not None


def test_direct_answer_routing_execution():
    graph = build_self_correcting_rag_graph()
    initial_state = {
        "query": "Hello, how are you today?",
        "thread_id": "test_thread_direct",
        "chat_history": [HumanMessage(content="Hello, how are you today?")],
        "route": "direct_answer",
        "route_reasoning": "Conversational greeting",
        "rewritten_query": None,
        "raw_documents": [],
        "filtered_documents": [],
        "generation": None,
        "doc_relevance": None,
        "hallucination_grade": None,
        "answer_grade": None,
        "retry_count": 0,
        "max_retries": 3,
        "trace_logs": [],
    }
    
    config = {"configurable": {"thread_id": "test_thread_direct"}}
    result = graph.invoke(initial_state, config=config)
    assert result is not None
    assert "generation" in result
    assert result["generation"] is not None
    assert result["route"] in ["direct_answer", "vectorstore"]
