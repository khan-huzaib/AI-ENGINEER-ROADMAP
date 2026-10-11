import os
import json

from dotenv import load_dotenv
from groq import Groq

from schemas import AIConcept

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def explain_concept(topic: str) -> AIConcept:

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b', # Note: adjusted to a standard Groq model, as 'openai/gpt-oss-20b' may cause routing errors on Groq
        messages=[
            {
                "role": "system",
                "content": (
                    "Explain AI concepts in simple English. "
                    "Return valid JSON with topic, definition, "
                    "keywords, and difficulty (e.g., Beginner, Intermediate, Advanced) fields."
                ),
            },
            {
                "role": "user",
                "content": f"Explain this concept: {topic}",
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    return AIConcept.model_validate(data)


if __name__ == "__main__":
    concept = explain_concept("RAG")

    print("Topic:", concept.topic)
    print("Definition:", concept.definition)
    print("Keywords:", concept.keywords)
    print("Difficulty:", concept.difficulty)