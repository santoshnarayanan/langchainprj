from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    sources: List[Source] = Field(description="Sources used to answer the question", default_factory=list)
