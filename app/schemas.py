from pydantic import BaseModel,Field


class AIConcept(BaseModel):
    topic: str
    definition: str
    difficulty: str
    keywords: list[str] = Field(min_length=1)
    
concept=AIConcept(
    topic="RAG",
    definition="Retrieval-Augmented Generation",
    difficulty="beginner",
    keywords=["LLM", "retrieval", "embeddings"]
)

