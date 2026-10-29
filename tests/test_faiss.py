
from app.ingestion.pdf_loader import load_pdf
from app.processing.chunker import split_documents
from app.embeddings.embedder import create_embedding_model
from app.retrieval.faiss_store import (
    create_faiss_index,
    search_faiss,
)

pdf_path = "data/raw/AD.pdf"

# 1. Load and split the FAA PDF.
documents = load_pdf(pdf_path)
chunks = split_documents(documents)

# 2. Generate embeddings.
embedding_model = create_embedding_model()
texts = [chunk.page_content for chunk in chunks]
vectors = embedding_model.embed_documents(texts)

# 3. Build the FAISS index.
index = create_faiss_index(vectors)

# 4. Embed the user's question.
query = "What inspection is required for the aircraft?"
query_vector = embedding_model.embed_query(query)

# 5. Retrieve relevant chunks with metadata filtering.
results = search_faiss(
    index=index,
    query_vector=query_vector,
    chunks=chunks,
    top_k=3,
    min_similarity=0.35,
    metadata_filter={"source": pdf_path},
)

print("Query:", query)
print("Retrieved results:", len(results))

if not results:
    print("\nInsufficient relevant evidence found.")
else:
    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Source: {result['metadata'].get('source')}")
        print(f"Page: {result['metadata'].get('page_label')}")
        print("Text:")
        print(result["text"][:500])
        print("-" * 60)

