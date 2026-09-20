import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

messages = [
    {
        "role": "system",
        "content": """You are an AI teacher helping a beginner learn artificial intelligence.

Rules:
- Use simple English.
- Give examples.
- Use bullet points when useful.
- If the question is unrelated to AI, politely say that you specialize in AI.
"""
    }
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            stream=True
        )

        assistant_response = ""

        print("AI: ", end="")

        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="")
            assistant_response += content

        print()

        messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

    except Exception as e:
        print(f"LLM Error: {e}")