from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.state import AgentState
from app.agent.prompts.grading import HALLUCINATION_GRADER_SYSTEM_PROMPT
from app.schemas.evaluation import GradeHallucinations
from app.core.config import settings
from app.core.logging import logger


def grade_hallucination_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer is grounded in and faithful to the reference facts.
    """
    documents = state.get("filtered_documents", [])
    generation = state.get("generation", "")
    
    logger.info("[Hallucination Grader Node] Checking factual consistency and grounding")

    context_text = "\n\n".join([doc.page_content for doc in documents])
    
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        google_api_key=settings.gemini_api_key,
    )
    structured_grader = llm.with_structured_output(GradeHallucinations)

    messages = [
        SystemMessage(content=HALLUCINATION_GRADER_SYSTEM_PROMPT),
        HumanMessage(content=f"Reference Facts:\n{context_text}\n\nGenerated Answer:\n{generation}")
    ]

    try:
        grade: GradeHallucinations = structured_grader.invoke(messages)
        is_grounded = grade.binary_score.strip().lower() == "yes"
        status = "grounded" if is_grounded else "hallucinated"
        reason = grade.explanation
    except Exception as e:
        logger.error(f"[Hallucination Grader Node] Fallback on error: {e}")
        status = "grounded"
        reason = f"Fallback assumption grounded due to evaluation error: {e}"

    logger.info(f"[Hallucination Grader Node] Result: {status} | Reason: {reason}")

    return {
        "hallucination_grade": status,
        "trace_logs": [{
            "node": "hallucination_grader",
            "action": "grade_grounding",
            "details": {"status": status, "reason": reason}
        }]
    }
