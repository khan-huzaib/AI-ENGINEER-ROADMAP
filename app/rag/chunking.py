text = """
Retrieval-Augmented Generation is a technique that allows
language models to use external information.

The retrieval component searches a collection of documents.
The retrieved documents are then provided to the language model.

Embeddings represent text as numerical vectors.
Vector databases store these vectors and allow similarity search.

Reranking can improve retrieval quality by reconsidering
the initial retrieved documents.
"""

def create_chunks(text, chunk_size=100):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks

chunks = create_chunks(text, chunk_size=30)

for i, chunk in enumerate(chunks):
    print(f"\nCHUNK {i}")
    print(chunk)
    