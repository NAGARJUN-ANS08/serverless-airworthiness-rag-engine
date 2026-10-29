import faiss
import numpy as np


def create_faiss_index(vectors):
    vectors = np.array(vectors, dtype="float32")

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index

def search_faiss(index, query_vector, top_k=3):
    query_vector = np.array(
        [query_vector],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    return scores[0], indices[0]
