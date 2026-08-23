from app.ingestion.pdf_loader import load_pdf
from app.processing.chunker import split_documents


pdf_path = "data/raw/AD.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

print("Original documents:", len(documents))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst chunk metadata:")
print(chunks[0].metadata)

print("\nFirst chunk length:")
print(len(chunks[0].page_content))