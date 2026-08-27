from typing import Any, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from app.core.llm_factory import get_chat_llm
from app.agent.state import AgentState
from app.agent.prompts.grading import DOC_GRADER_SYSTEM_PROMPT
from app.schemas.evaluation import GradeDocument
from app.core.config import settings
from app.core.logging import logger


def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the relevance of each retrieved document to the user query.
    Filters out irrelevant documents. If no documents are relevant, sets doc_relevance='no'.
    """
    query = state.get("rewritten_query") or state.get("query", "")
    documents = state.get("raw_documents", [])
    
    logger.info(f"[Doc Grader Node] Grading {len(documents)} retrieved documents against query: '{query}'")

    if not documents:
        logger.warning("[Doc Grader Node] No documents retrieved from vector store.")
        return {
            "filtered_documents": [],
            "doc_relevance": "no",
            "trace_logs": [{
                "node": "doc_grader",
                "action": "grade_documents",
                "details": {"relevant_count": 0, "total_count": 0, "doc_relevance": "no"}
            }]
        }

    llm = get_chat_llm(temperature=0.0)
    structured_grader = llm.with_structured_output(GradeDocument)

    filtered_docs: List[Document] = []
    grades_info = []

    for idx, doc in enumerate(documents):
        messages = [
            SystemMessage(content=DOC_GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Question: {query}\n\nRetrieved Document Content:\n{doc.page_content}")
        ]
        try:
            grade: GradeDocument = structured_grader.invoke(messages)
            is_rel = grade.binary_score.strip().lower() == "yes"
            grades_info.append({"index": idx, "score": grade.binary_score, "reason": grade.explanation})
            if is_rel:
                filtered_docs.append(doc)
        except Exception as e:
            logger.error(f"[Doc Grader Node] Error evaluating document {idx}: {e}")
            # Fallback: keep document if grading fails
            filtered_docs.append(doc)
            grades_info.append({"index": idx, "score": "yes (fallback)", "reason": str(e)})

    doc_relevance = "yes" if len(filtered_docs) > 0 else "no"
    logger.info(f"[Doc Grader Node] Passed {len(filtered_docs)}/{len(documents)} documents. Relevance: {doc_relevance}")

    return {
        "filtered_documents": filtered_docs,
        "doc_relevance": doc_relevance,
        "trace_logs": [{
            "node": "doc_grader",
            "action": "grade_documents",
            "details": {
                "total_retrieved": len(documents),
                "relevant_count": len(filtered_docs),
                "doc_relevance": doc_relevance,
                "grades": grades_info
            }
        }]
    }
