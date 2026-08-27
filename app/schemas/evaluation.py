from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource or direct generation."""
    datasource: str = Field(
        description="Routing decision: strictly 'vectorstore' for factual/document questions or 'direct_answer' for greetings/persona queries."
    )
    reasoning: str = Field(description="Brief reason for the routing choice.")


class GradeDocument(BaseModel):
    """Binary score for document topical relevance check."""
    binary_score: str = Field(
        description="Strictly 'yes' if document text discusses or mentions the question's topic/keywords (even if counterfactual or custom). 'no' ONLY if entirely off-topic."
    )
    explanation: str = Field(description="Brief explanation of topical relation.")


class GradeHallucinations(BaseModel):
    """Binary score for factual grounding against reference documents."""
    binary_score: str = Field(
        description="Strictly 'yes' if the answer is grounded in and faithful to reference facts. 'no' if it invents unsupported facts."
    )
    explanation: str = Field(description="Brief explanation of grounding assessment.")


class GradeAnswer(BaseModel):
    """Binary score to assess if the answer addresses the user question."""
    binary_score: str = Field(
        description="Strictly 'yes' if the answer resolves the question using the available context. 'no' if evasive or unhelpful."
    )
    explanation: str = Field(description="Brief explanation of usefulness assessment.")


class QueryRewrite(BaseModel):
    """Refined query produced by the rewriter node."""
    rewritten_query: str = Field(description="The reformulated, optimized search query.")
    reasoning: str = Field(description="Why the query was rewritten.")
