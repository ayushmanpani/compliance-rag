# app/api/endpoints.py

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import os
import shutil
import uuid
import json
from datetime import datetime
from typing import Optional

from app.services.rag import RAGStore
from app.utils.ingest import ingest_uploaded_pdf

# -----------------------
# Router & global objects
# -----------------------

router = APIRouter()

UPLOAD_DIR = "data/docs"
METADATA_FILE = "data/docs/metadata.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load RAG store once at startup
rag = RAGStore()
rag.load_store_if_exists()

# -----------------------
# Request models
# -----------------------

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

# -----------------------
# Helper functions
# -----------------------

def save_metadata(entry: dict):
    """
    Append document metadata to metadata.json
    """
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------
# API Endpoints
# -----------------------

@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Ask a question to the RAG system
    """
    if rag.chain is None:
        return {
            "error": "No documents uploaded yet. Please upload a PDF first."
        }
    response = rag.ask(request.question)
    return response


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a compliance PDF, ingest it, and update FAISS
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save metadata
    save_metadata({
        "doc_id": doc_id,
        "original_name": file.filename,
        "stored_name": stored_filename,
        "uploaded_at": datetime.utcnow().isoformat()
    })

    # Ingest PDF into vector store
    ingest_uploaded_pdf(file_path, file.filename)

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename
    }
