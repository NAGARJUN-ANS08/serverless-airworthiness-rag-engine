from app.ingestion.pdf_loader import load_pdf
from app.processing.chunker import split_documents
from app.embeddings.embedder import create_embedding_model
from app.retrieval.faiss_store import create_faiss_index, search_faiss


pdf_path = "data/raw/AD.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

embedding_model = create_embedding_model()

texts = [chunk.page_content for chunk in chunks]

vectors = embedding_model.embed_documents(texts)

index = create_faiss_index(vectors)

query = "What inspection is required for the aircraft?"

query_vector = embedding_model.embed_query(query)

scores, indices = search_faiss(
    index,
    query_vector,
    top_k=3
)

print("Query:")
print(query)

print("\nTop results:")

for score, index_number in zip(scores, indices):
    print(
        f"Chunk index: {index_number}, "
        f"Similarity: {score:.4f}"
    )
    print(chunks[index_number].page_content[:500])
    print("-" * 60)
