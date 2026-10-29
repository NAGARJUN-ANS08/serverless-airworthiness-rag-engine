from app.ingestion.pdf_loader import load_pdf
from app.processing.chunker import split_documents
from app.embeddings.embedder import create_embedding_model


pdf_path = "data/raw/AD.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

embedding_model = create_embedding_model()

texts = [chunk.page_content for chunk in chunks]

vectors = embedding_model.embed_documents(texts)

print("Number of chunks:", len(chunks))
print("Number of vectors:", len(vectors))
print("Dimensions of each vector:", len(vectors[0]))

print("\nFirst 10 values of first vector:")
print(vectors[0][:10])