import pytest
from app.agent.nodes.router import route_query_node
from app.agent.graph import route_decision_edge


def test_router_conversational_edge():
    state = {"route": "direct_answer"}
    edge = route_decision_edge(state)
    assert edge == "generator_direct"


def test_router_vectorstore_edge():
    state = {"route": "vectorstore"}
    edge = route_decision_edge(state)
    assert edge == "retriever"
