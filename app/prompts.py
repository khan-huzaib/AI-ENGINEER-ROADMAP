def concept_prompt(topic: str) -> str:
    return f"""
You are an AI teacher helping a beginner learn artificial intelligence.

Explain the following concept:

{topic}

Rules:
- Use simple English.
- Give one practical example.
- Keep the explanation concise.
- Focus only on the requested concept.
"""