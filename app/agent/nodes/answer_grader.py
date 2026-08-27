from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.state import AgentState
from app.agent.prompts.grading import ANSWER_GRADER_SYSTEM_PROMPT
from app.schemas.evaluation import GradeAnswer
from app.core.config import settings
from app.core.logging import logger


def grade_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer successfully answers and resolves the user's question.
    """
    query = state.get("query", "")
    generation = state.get("generation", "")
    
    logger.info(f"[Answer Grader Node] Grading answer quality against query: '{query}'")

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        google_api_key=settings.gemini_api_key,
    )
    structured_grader = llm.with_structured_output(GradeAnswer)

    messages = [
        SystemMessage(content=ANSWER_GRADER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Question:\n{query}\n\nGenerated Answer:\n{generation}")
    ]

    try:
        grade: GradeAnswer = structured_grader.invoke(messages)
        is_useful = grade.binary_score.strip().lower() == "yes"
        status = "useful" if is_useful else "not_useful"
        reason = grade.explanation
    except Exception as e:
        logger.error(f"[Answer Grader Node] Fallback on error: {e}")
        status = "useful"
        reason = f"Fallback assumption useful due to error: {e}"

    logger.info(f"[Answer Grader Node] Result: {status} | Reason: {reason}")

    return {
        "answer_grade": status,
        "trace_logs": [{
            "node": "answer_grader",
            "action": "grade_answer_usefulness",
            "details": {"status": status, "reason": reason}
        }]
    }
