import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.rag import RAGStore
import uuid
import os
import re


def is_index_like(text: str) -> bool:
    # High ratio of numbers / codes = index or table
    digits = sum(c.isdigit() for c in text)
    ratio = digits / max(len(text), 1)

    # Detect circular-style codes
    if re.search(r"DNBS|No\s+\d+/\d+", text):
        return True

    return ratio > 0.25  # 25% numbers = likely index

def extract_text_from_pdf(pdf_path: str):
    """
    Returns a list of (page_number, page_text) tuples.
    """
    pages_content = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and not is_index_like(text):
                pages_content.append((page_num, text))

    # OCR fallback ONLY if nothing useful extracted
    if not pages_content:
        images = convert_from_path(pdf_path, dpi=200)
        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img)
            if text and not is_index_like(text):
                pages_content.append((page_num, text))

    return pages_content


def create_chunks(text: str,
    base_metadata: dict,
    page_num: int,
    chunk_size=400,
    chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        
        if is_index_like(chunk):
            continue  # 🚫 skip index-like content

        documents.append(
            Document(
                page_content=f"passage: {chunk}",
                metadata={
                    **base_metadata,
                    "chunk_id": i,
                    "page": page_num
                }
            )
        )
        
    print("\n[DEBUG] Retrieved chunks:")
    for i, doc in enumerate(documents):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:300])
    

    return documents


def ingest_uploaded_pdf(file_path: str,
    original_filename: str,
    doc_id: str):

    base_metadata = {
        "doc_id": doc_id,
        "original_filename": original_filename
    }

    print("[INFO] Extracting text and Creating Chunks...")
    pages = extract_text_from_pdf(file_path)

    documents = []
    for page_num, page_text in pages:
        page_documents = create_chunks(
            text=page_text,
            base_metadata=base_metadata,
            page_num=page_num
        )
        documents.extend(page_documents)


    print(f"[INFO] Total chunks: {len(documents)}")

    rag = RAGStore()
    rag.load_store_if_exists()
    rag.add_documents(documents)

    print("[INFO] PDF ingested successfully.")

    return doc_id
