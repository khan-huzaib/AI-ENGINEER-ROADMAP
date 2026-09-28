from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import re

documents = [
    "RAG retrieves external information before generating an answer.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store embeddings and support similarity search.",
    "FastAPI is a Python framework for building APIs.",
    "Docker packages applications into containers.",
]


def tokenize(text):
    # lowercase and keep only words, so "answer." matches "answer"
    return re.findall(r"\w+", text.lower())


# ---------- BM25 setup ----------
bm25 = BM25Okapi([tokenize(doc) for doc in documents])

# ---------- Dense setup (your original code) ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# ---------- Each search returns a ranked list of document numbers ----------
def bm25_search(query, k=5):
    scores = bm25.get_scores(tokenize(query))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def dense_search(query, k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [int(i) for i in indices[0]]


# ---------- Combine with Reciprocal Rank Fusion ----------
def hybrid_search(query, k=3, rrf_k=60):
    points = {}
    for ranked_list in (bm25_search(query), dense_search(query)):
        for rank, doc_id in enumerate(ranked_list):
            points[doc_id] = points.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
    best = sorted(points, key=points.get, reverse=True)[:k]
    return [documents[i] for i in best]


query = "How does RAG retrieve information?"

print("BM25 results:")
for i in bm25_search(query, k=3):
    print(" -", documents[i])

print("\nDense results:")
for i in dense_search(query, k=3):
    print(" -", documents[i])

print("\nHybrid results:")
for doc in hybrid_search(query, k=3):
    print(" -", doc)