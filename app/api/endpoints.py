# app/api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import uuid
import json
from datetime import datetime
from typing import Optional

from app.services.rag import RAGStore
from app.utils.ingest import ingest_uploaded_pdf, rebuild_faiss_from_metadata

# =======================
# Absolute paths (CRITICAL)
# =======================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
METADATA_PATH = os.path.join(DOCS_DIR, "metadata.json")

DEFAULT_FAISS_PATH = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(DOCS_DIR, exist_ok=True)

# =======================
# Router & global objects
# =======================

router = APIRouter()

rag = RAGStore()          # uses absolute FAISS path internally
rag.load_store_if_exists()

# =======================
# Request models
# =======================

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

# =======================
# Helper functions
# =======================

def load_metadata():
    if not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0:
        return []

    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_metadata(entry: dict):
    data = load_metadata()
    data.append(entry)

    with open(METADATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

# =======================
# API Endpoints
# =======================

@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/ask")
async def ask_question(request: QueryRequest):
    if rag.chain is None:
        return {
            "error": "No documents uploaded yet. Please upload a PDF first."
        }

    return rag.ask(
        question=request.question,
        doc_id=request.doc_id
    )


@router.post("/upload")
async def upload_pdf(up_file: UploadFile = File(...)):
 

    if not up_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Generate UUID for document
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}.pdf"
    file_path = os.path.join(DOCS_DIR, stored_filename)

    # Save PDF to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(up_file.file, buffer)
        
    await up_file.close()

    # Save metadata FIRST
    save_metadata({
        "doc_id": doc_id,
        "original_filename": up_file.filename,
        "stored_filename": stored_filename,
        "uploaded_at": datetime.utcnow().isoformat()
    })

    # Ingest PDF into FAISS
    try:
        ingest_uploaded_pdf(file_path=file_path,
                            original_filename=up_file.filename,
                            doc_id=doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # IMPORTANT: reload FAISS into running process
    rag.load_store_if_exists()

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": up_file.filename
    }


@router.get("/documents")
async def list_documents():
    metadata = load_metadata()

    return [
        {
            "doc_id": entry["doc_id"],
            "original_filename": entry["original_filename"]
        }
        for entry in metadata
    ]

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    metadata = load_metadata()

    if not metadata:
        raise HTTPException(status_code=404, detail="No documents found")

    # Find matching document
    entry = next((item for item in metadata if item["doc_id"] == doc_id), None)

    if not entry:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1️⃣ Delete PDF file
    pdf_path = os.path.join(DOCS_DIR, entry["stored_filename"])
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    # 2️⃣ Remove entry from metadata list
    metadata = [item for item in metadata if item["doc_id"] != doc_id]

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3️⃣ Rebuild FAISS safely
    rebuild_faiss_from_metadata()
    
    rag.vstore = None
    rag.retriever = None
    rag.chain = None

    # 4️⃣ Reload FAISS into running process
    rag.load_store_if_exists()

    return {
        "status": "deleted",
        "doc_id": doc_id
    }


@router.post("/reset")
async def reset_knowledge_base():
    """
    Completely reset the knowledge base:
    - Delete FAISS index (disk)
    - Delete all PDFs
    - Delete metadata.json
    - Clear in-memory FAISS
    """

    # 1️⃣ Delete FAISS index (disk)
    if os.path.exists(DEFAULT_FAISS_PATH):
        shutil.rmtree(DEFAULT_FAISS_PATH)

    # 2️⃣ Delete all PDFs (keep docs folder)
    if os.path.exists(DOCS_DIR):
        for f in os.listdir(DOCS_DIR):
            if f.lower().endswith(".pdf"):
                os.remove(os.path.join(DOCS_DIR, f))

    # 3️⃣ Delete metadata.json
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)

    # 4️⃣ 🔥 Clear in-memory FAISS (CRITICAL)
    rag.vstore = None
    rag.retriever = None
    rag.chain = None

    return {
        "status": "reset",
        "message": "Knowledge base cleared successfully"
    }

