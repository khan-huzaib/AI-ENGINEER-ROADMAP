import json

from app.evaluation.metrices import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    reciprocal_rank
)

from app.rag.dense_retrieval import (
    bm25_search,
    dense_search,
    hybrid_search
)


# -----------------------------
# Evaluation dataset
# -----------------------------

questions = [
    {
        "question": "What is RAG?",
        "relevant_documents": [0]
    },
    {
        "question": "What are embeddings?",
        "relevant_documents": [1]
    },
    {
        "question": "What is a vector database?",
        "relevant_documents": [2]
    },
    {
        "question": "What is FastAPI?",
        "relevant_documents": [3]
    },
    {
        "question": "What is Docker?",
        "relevant_documents": [4]
    }
]


# -----------------------------
# Evaluation function
# -----------------------------

def evaluate_retriever(retriever, retriever_name):

    recall_scores = []
    precision_scores = []
    mrr_scores = []
    ndcg_scores = []

    print("\n==============================")
    print(retriever_name)
    print("==============================")

    for item in questions:

        question = item["question"]

        relevant_documents = item[
            "relevant_documents"
        ]

        retrieved_documents = retriever(
            question,
            k=3
        )

        recall = recall_at_k(
            retrieved_documents,
            relevant_documents,
            k=3
        )

        precision = precision_at_k(
            retrieved_documents,
            relevant_documents,
            k=3
        )

        mrr = reciprocal_rank(
            retrieved_documents,
            relevant_documents
        )

        ndcg = ndcg_at_k(
            retrieved_documents,
            relevant_documents,
            k=3
        )

        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        print("\nQuestion:", question)
        print("Relevant:", relevant_documents)
        print("Retrieved:", retrieved_documents)
        print("Recall@3:", recall)
        print("Precision@3:", precision)
        print("RR:", mrr)
        print("NDCG@3:", ndcg)


    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    average_mrr = (
        sum(mrr_scores)
        / len(mrr_scores)
    )

    average_ndcg = (
        sum(ndcg_scores)
        / len(ndcg_scores)
    )


    print("\n--------- AVERAGE ---------")

    print(
        f"Recall@3: {average_recall:.3f}"
    )

    print(
        f"Precision@3: {average_precision:.3f}"
    )

    print(
        f"MRR: {average_mrr:.3f}"
    )

    print(
        f"NDCG@3: {average_ndcg:.3f}"
    )


# -----------------------------
# Run evaluation
# -----------------------------

evaluate_retriever(
    bm25_search,
    "BM25"
)

evaluate_retriever(
    dense_search,
    "Dense Retrieval"
)

evaluate_retriever(
    hybrid_search,
    "Hybrid Retrieval"
)
