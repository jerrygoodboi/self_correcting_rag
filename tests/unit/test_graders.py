import pytest
from app.agent.graph import doc_grading_edge, hallucination_edge, answer_quality_edge
from langchain_core.documents import Document


def test_doc_grading_edge_relevant_docs():
    state = {
        "doc_relevance": "yes",
        "filtered_documents": [Document(page_content="Relevant content")],
        "retry_count": 0,
        "max_retries": 3,
    }
    assert doc_grading_edge(state) == "generator_rag"


def test_doc_grading_edge_triggers_rewrite():
    state = {
        "doc_relevance": "no",
        "filtered_documents": [],
        "retry_count": 1,
        "max_retries": 3,
    }
    assert doc_grading_edge(state) == "rewriter"


def test_doc_grading_edge_exceeds_max_retries():
    state = {
        "doc_relevance": "no",
        "filtered_documents": [],
        "retry_count": 3,
        "max_retries": 3,
    }
    assert doc_grading_edge(state) == "generator_fallback"


def test_hallucination_edge_grounded():
    state = {
        "hallucination_grade": "grounded",
        "retry_count": 0,
        "max_retries": 3,
    }
    assert hallucination_edge(state) == "answer_grader"


def test_hallucination_edge_hallucinated_triggers_rewrite():
    state = {
        "hallucination_grade": "hallucinated",
        "retry_count": 1,
        "max_retries": 3,
    }
    assert hallucination_edge(state) == "rewriter"


def test_answer_quality_edge():
    assert answer_quality_edge({"answer_grade": "useful", "retry_count": 0, "max_retries": 3}) == "END"
    assert answer_quality_edge({"answer_grade": "not_useful", "retry_count": 1, "max_retries": 3}) == "rewriter"
    assert answer_quality_edge({"answer_grade": "not_useful", "retry_count": 3, "max_retries": 3}) == "END"
