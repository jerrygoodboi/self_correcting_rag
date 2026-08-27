from typing import Literal
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route user query to vectorstore or direct conversational answer."""
    datasource: Literal["vectorstore", "direct_answer"] = Field(
        ...,
        description="Given a user query, choose whether to route it to 'vectorstore' for knowledge retrieval or 'direct_answer' for chit-chat, greetings, or common reasoning.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this routing decision was made.",
    )


class GradeDocument(BaseModel):
    """Binary score for document relevance to the query."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Whether the document is relevant to the user query ('yes' or 'no').",
    )
    explanation: str = Field(
        ...,
        description="Brief justification of why the document is or is not relevant.",
    )


class GradeHallucinations(BaseModel):
    """Binary score for factual grounding against documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Whether the answer is grounded in and supported by the retrieved facts ('yes' for grounded, 'no' for hallucinated/unsupported).",
    )
    explanation: str = Field(
        ...,
        description="Brief justification of whether the answer is grounded in facts.",
    )


class GradeAnswer(BaseModel):
    """Binary score to assess if the answer resolves the user query."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Whether the generated answer directly resolves and addresses the user's question ('yes' or 'no').",
    )
    explanation: str = Field(
        ...,
        description="Brief justification of how well the answer addresses the query.",
    )


class QueryRewrite(BaseModel):
    """Rewritten query optimized for vector store retrieval."""
    improved_query: str = Field(
        ...,
        description="The reformulated, clear, keyword-rich query optimized for semantic document retrieval.",
    )
    reasoning: str = Field(
        ...,
        description="Rationale behind the query transformation.",
    )
