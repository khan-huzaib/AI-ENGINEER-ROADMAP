import math

def recall_at_k(
    retrieved_documents,
    relevant_documents,
    k
):
    retrieved = retrieved_documents[:k]
    relevant = set(relevant_documents)

    for document in retrieved:
        if document in relevant:
            return 1.0

    return 0.0


def precision_at_k(
    retrieved_documents,
    relevant_documents,
    k
):
    retrieved = retrieved_documents[:k]
    relevant = set(relevant_documents)

    relevant_count = sum(
        1 for document in retrieved
        if document in relevant
    )

    return relevant_count / k


def reciprocal_rank(
    retrieved_documents,
    relevant_documents
):
    relevant = set(relevant_documents)

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):
        if document in relevant:
            return 1 / rank

    return 0.0



def ndcg_at_k(
    retrieved_documents,
    relevant_documents,
    k
):
    relevant = set(relevant_documents)
    retrieved = retrieved_documents[:k]

    dcg = 0.0

    for rank, document in enumerate(retrieved, start=1):
        if document in relevant:
            dcg += 1 / math.log2(rank + 1)

    ideal_count = min(len(relevant), k)

    if ideal_count == 0:
        return 0.0

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal_dcg