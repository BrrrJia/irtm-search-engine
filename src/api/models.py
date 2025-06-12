from pydantic import BaseModel, Field
from typing import List


class Document(BaseModel):
    tweet_id: str
    text: str


class QueryResponse(BaseModel):
    results: List[Document]
    total_count: int
    query: str
    mode: str


class ClassificationRequest(BaseModel):
    name: str = Field(
        default="Die Simpsons", min_length=1, max_length=100, description="Name field"
    )
    title: str = Field(
        default="Super", min_length=1, max_length=200, description="Title field"
    )
    review: str = Field(
        default="ich LIEBE dieses spiel",
        min_length=1,
        max_length=1000,
        description="Review text",
    )


class ClassificationResponse(BaseModel):
    label: str
