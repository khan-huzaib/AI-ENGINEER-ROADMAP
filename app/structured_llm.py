import os
import json

from dotenv import load_dotenv
from groq import Groq
from schemas import AIConcept
from prompts import concept_prompt


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

schema = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string"
        },
        "definition": {
            "type": "string"
        },
        "difficulty": {
            "type": "string"
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "topic",
        "definition",
        "difficulty",
        "keywords"
    ],
    "additionalProperties": False
}

respons = client.chat.completions.create(
    model="openai/gpt-oss-20b",

    messages=[
        {
            "role": "system",
            "content": "You extract information about artificial intelligence concepts."
        },
        {
            "role": "user",
            "content": concept_prompt("RAG")
        }
    ],

    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ai_concept",
            "strict": True,
            "schema": schema
        }
    }
)

result = json.loads(
    respons.choices[0].message.content
)

concept= AIConcept(**result)

print(concept)
print(concept.topic)
print(concept.keywords)