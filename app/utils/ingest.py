import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.rag import RAGStore
import uuid
import os
import re 
import json
import time
from dotenv import load_dotenv

load_dotenv()
# Retention configuration
DEFAULT_RETENTION_DAYS = 7

RETENTION_DAYS = int(
    os.getenv("RETENTION_DAYS", DEFAULT_RETENTION_DAYS)
)

RETENTION_SECONDS = RETENTION_DAYS * 24 * 60 * 60


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
METADATA_PATH = os.path.join(DOCS_DIR, "metadata.json")

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

def rebuild_faiss_from_metadata():
    """
    Safely rebuild FAISS index from remaining documents
    after a delete operation.
    Works with metadata.json as a LIST.
    """

    if not os.path.exists(METADATA_PATH):
        print("[INFO] No metadata.json found. Skipping FAISS rebuild.")
        return

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if not metadata:
        print("[INFO] Metadata empty. Clearing FAISS index.")
        rag = RAGStore()
        rag.vstore = None
        return

    rag = RAGStore()
    rag.vstore = None  # force fresh rebuild

    total_chunks = 0

    for entry in metadata:
        doc_id = entry["doc_id"]
        original_filename = entry["original_filename"]
        stored_filename = entry["stored_filename"]

        pdf_path = os.path.join(DOCS_DIR, stored_filename)
        if not os.path.exists(pdf_path):
            continue

        print(f"[INFO] Re-ingesting: {original_filename}")

        base_metadata = {
            "doc_id": doc_id,
            "original_filename": original_filename
        }

        pages = extract_text_from_pdf(pdf_path)

        documents = []
        for page_num, page_text in pages:
            documents.extend(
                create_chunks(
                    text=page_text,
                    base_metadata=base_metadata,
                    page_num=page_num
                )
            )

        total_chunks += len(documents)
        rag.add_documents(documents)

        print(f"[INFO] FAISS rebuilt successfully with {total_chunks} chunks.")

def cleanup_expired_documents():
    """
    Delete documents older than retention window
    and rebuild FAISS index.
    Works with metadata.json as a LIST.
    """
    if not os.path.exists(METADATA_PATH):
        print("[INFO] No metadata.json found. Cleanup skipped.")
        return

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if not metadata:
        print("[INFO] Metadata empty. Nothing to clean.")
        return

    now = int(time.time())
    retained_entries = []
    expired_count = 0

    for entry in metadata:
        uploaded_at = entry.get("uploaded_at")

        if uploaded_at and (now - uploaded_at) > RETENTION_SECONDS:
            pdf_path = os.path.join(DOCS_DIR, entry["stored_filename"])
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            expired_count += 1
        else:
            retained_entries.append(entry)

    if expired_count == 0:
        print("[INFO] No expired documents found.")
        return

    # Save cleaned metadata list
    with open(METADATA_PATH, "w") as f:
        json.dump(retained_entries, f, indent=2)

    print(f"[INFO] Removed {expired_count} expired documents.")

    # Rebuild FAISS index from remaining docs
    rebuild_faiss_from_metadata()
