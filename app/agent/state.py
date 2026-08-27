from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    State representing the graph execution context in Self-Correcting Agentic RAG.
    """
    # User Input & Session
    query: str
    thread_id: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    
    # Routing & Search
    route: str  # "vectorstore" | "direct_answer"
    route_reasoning: Optional[str]
    rewritten_query: Optional[str]
    
    # Documents
    raw_documents: List[Document]
    filtered_documents: List[Document]
    
    # Generation & Evaluation
    generation: Optional[str]
    doc_relevance: Optional[str]  # "yes" | "no"
    hallucination_grade: Optional[str]  # "grounded" | "hallucinated"
    answer_grade: Optional[str]  # "useful" | "not_useful"
    
    # Self-Correction Control
    retry_count: int
    max_retries: int
    
    # Trace & Observability Logs
    trace_logs: Annotated[List[Dict[str, Any]], operator.add]
