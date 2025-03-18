from pydantic import BaseModel, Field

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GradeDocument(BaseModel):
    grade: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class QuestionRouter(BaseModel):
    route_to: str = Field(desrciption="RAG or LLM or Irrelevant")
