from pydantic import BaseModel


class AIConcept(BaseModel):
    topic: str
    definition: str
    difficulty: str
    keywords: list[str]
    
concept=AIConcept(
    topic="RAG",
    definition="Retrieval-Augmented Generation",
    difficulty="beginner",
    keywords=["LLM", "retrieval", "embeddings"]
)

