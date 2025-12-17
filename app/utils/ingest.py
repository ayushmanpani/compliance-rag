import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.rag import RAGStore
import uuid
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pass

    if len(text.strip()) < 20:
        pages = convert_from_path(pdf_path, dpi=200)
        for img in pages:
            text += pytesseract.image_to_string(img) + "\n"

    return text


def create_chunks(text: str, base_metadata: dict, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    **base_metadata,
                    "chunk_id": i
                }
            )
        )

    return documents


def ingest_uploaded_pdf(file_path: str, original_filename: str):
    doc_id = str(uuid.uuid4())

    base_metadata = {
        "doc_id": doc_id,
        "original_filename": original_filename
    }

    print("[INFO] Extracting text...")
    text = extract_text_from_pdf(file_path)

    print("[INFO] Creating chunks...")
    documents = create_chunks(text, base_metadata)

    print(f"[INFO] Total chunks: {len(documents)}")

    rag = RAGStore()
    rag.load_store_if_exists()
    rag.add_documents(documents)

    print("[INFO] PDF ingested successfully.")

    return doc_id
