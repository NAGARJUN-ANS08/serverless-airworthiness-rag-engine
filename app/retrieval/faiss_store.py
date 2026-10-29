
import faiss
import numpy as np


def create_faiss_index(vectors):
    """Create an exact cosine-similarity index from embedding vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Vectors must be a non-empty 2D array.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain invalid numeric values.")

    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Zero-length vectors cannot be normalized.")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def search_faiss(
    index,
    query_vector,
    chunks,
    top_k=3,
    min_similarity=0.35,
    metadata_filter=None,
):
    """
    Search FAISS and return matching chunks with source metadata.

    metadata_filter matches exact metadata values.
    min_similarity is a configurable relevance threshold,
    not a probability of correctness.
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunks must have matching sizes.")

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(
        query_vector, dtype=np.float32
    ).reshape(1, -1)

    if query_vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match index.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains invalid numeric values.")

    if np.linalg.norm(query_vector) == 0:
        raise ValueError("Query vector cannot be zero.")

    faiss.normalize_L2(query_vector)

    # Search all indexed chunks so metadata filtering
    # does not miss matching chunks outside the initial top_k.
    scores, indices = index.search(query_vector, index.ntotal)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        document = chunks[int(chunk_index)]
        metadata = document.metadata or {}

        if metadata_filter:
            matches = all(
                metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            if not matches:
                continue

        if float(score) < min_similarity:
            continue

        results.append({
            "chunk_index": int(chunk_index),
            "score": float(score),
            "text": document.page_content,
            "metadata": metadata,
        })

        if len(results) == top_k:
            break

    return results

