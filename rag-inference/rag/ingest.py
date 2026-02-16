import os
import uuid
import json
import pdfplumber

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import DOCS_DIR, FAISS_DIR, METADATA_PATH


async def ingest_pdf(upload_file):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(DOCS_DIR, f"{file_id}.pdf")

    # Save file
    with open(file_path, "wb") as f:
        f.write(await upload_file.read())

    # Extract text
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk, metadata={"source": file_id})
        for chunk in chunks
    ]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2"
    )

    if os.listdir(FAISS_DIR):
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings)
        vectorstore.add_documents(documents)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(FAISS_DIR)

    # Update metadata
    metadata = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            try:
                metadata = json.load(f)
            except:
                metadata = []

    metadata.append({
        "doc_id": file_id,
        "filename": upload_file.filename
    })

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
