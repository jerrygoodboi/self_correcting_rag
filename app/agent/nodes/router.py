from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.state import AgentState
from app.agent.prompts.router import ROUTER_SYSTEM_PROMPT
from app.schemas.evaluation import RouteQuery
from app.core.config import settings
from app.core.logging import logger


def route_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that analyzes the user query to decide whether to route to vectorstore retrieval
    or to direct conversational answer.
    """
    query = state.get("query", "")
    logger.info(f"[Router Node] Evaluating routing decision for query: '{query}'")

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        google_api_key=settings.gemini_api_key,
    )
    
    structured_router = llm.with_structured_output(RouteQuery)
    
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: {query}")
    ]
    
    try:
        decision: RouteQuery = structured_router.invoke(messages)
        datasource = decision.datasource
        reasoning = decision.reasoning
    except Exception as e:
        logger.error(f"[Router Node] Fallback due to error: {e}")
        datasource = "vectorstore"
        reasoning = f"Default fallback to vectorstore due to parsing error: {e}"

    logger.info(f"[Router Node] Decision: {datasource} | Reason: {reasoning}")
    
    return {
        "route": datasource,
        "route_reasoning": reasoning,
        "trace_logs": [{
            "node": "router",
            "action": "route_decision",
            "details": {"route": datasource, "reasoning": reasoning}
        }]
    }
