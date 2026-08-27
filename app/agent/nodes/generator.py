from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.state import AgentState
from app.agent.prompts.generation import RAG_GENERATION_SYSTEM_PROMPT, DIRECT_ANSWER_SYSTEM_PROMPT
from app.core.config import settings
from app.core.logging import logger


def generate_rag_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes an answer grounded strictly in the filtered relevant documents.
    """
    query = state.get("query", "")
    documents = state.get("filtered_documents", [])
    
    logger.info(f"[Generator Node - RAG] Generating answer for: '{query}' using {len(documents)} context chunks")

    context_text = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')} | Chunk: {idx+1}]\n{doc.page_content}"
        for idx, doc in enumerate(documents)
    ])

    system_prompt = RAG_GENERATION_SYSTEM_PROMPT.format(context=context_text)
    
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key,
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "generation": answer_text,
        "trace_logs": [{
            "node": "generator_rag",
            "action": "generate_answer",
            "details": {
                "context_chunks_count": len(documents),
                "generation_snippet": answer_text[:150] + "..."
            }
        }]
    }


def generate_direct_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a conversational or direct answer for non-retrieval queries.
    """
    query = state.get("query", "")
    logger.info(f"[Generator Node - Direct] Generating conversational response for: '{query}'")

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.7,
        google_api_key=settings.gemini_api_key,
    )
    
    messages = [
        SystemMessage(content=DIRECT_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "generation": answer_text,
        "trace_logs": [{
            "node": "generator_direct",
            "action": "direct_answer",
            "details": {"generation": answer_text}
        }]
    }


def generate_fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a graceful fallback when max retries are exceeded or no relevant docs exist.
    """
    query = state.get("query", "")
    retries = state.get("retry_count", 0)
    logger.warning(f"[Generator Node - Fallback] Max retries ({retries}) reached or no documents found for '{query}'")

    fallback_answer = (
        f"I searched the knowledge base across {retries} refinement iterations, but could not find sufficiently verified "
        f"information to answer: '{query}'. "
        f"Please verify if the relevant documentation is ingested or try rephrasing your request."
    )

    return {
        "generation": fallback_answer,
        "trace_logs": [{
            "node": "generator_fallback",
            "action": "fallback_response",
            "details": {"reason": "Max retries exceeded or no relevant docs", "answer": fallback_answer}
        }]
    }
