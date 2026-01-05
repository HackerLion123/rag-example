from dataclasses import dataclass
from langchain_core.documents import Document

from typing import List, Optional
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class RerankResult:
    document: Document
    score: float

class Citation(BaseModel):
    source: str = Field(description="Source document path or name")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    slide: Optional[int] = Field(default=None, description="Slide number if applicable")
    relevant_text: str = Field(description="Relevant text span for citation")


class RAGResponse(BaseModel):
    answer: str = Field(description="The generated answer")
    citations: List[dict] = Field(default_factory=list, description="List of citations")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1")
    
    def to_dict(self):
        return self.model_dump()


class QueryRequest(BaseModel):
    question: str = Field(description="The user question")
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve")


class AddDocumentRequest(BaseModel):
    file_paths: List[str] = Field(description="List of file paths to add")


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    components: dict = Field(default_factory=dict)


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    category: str = "general"


class EvalResult(BaseModel):
    case_id: str
    question: str
    generated_answer: str
    expected_answer: Optional[str]
    citations: List[dict]
    confidence: float
    scores: dict
    passed: bool
    failure_reason: Optional[str] = None
