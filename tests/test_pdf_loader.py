from app.ingestion.pdf_loader import load_pdf


pdf_path = "data/raw/AD.pdf"

documents = load_pdf(pdf_path)

print("Number of pages:", len(documents))

print("\nFirst page:")
print(documents[0].page_content[:1000])

print("\nMetadata:")
print(documents[0].metadata)