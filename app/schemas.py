from typing import Any

from pydantic import BaseModel, Field

from .config import settings


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    collection_name: str
    mode: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=10)


class RetrievedChunk(BaseModel):
    index: int
    content: str
    metadata: dict[str, Any]


class AnswerResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    mode: str
