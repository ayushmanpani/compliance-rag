# 📄 Intelligent Financial Compliance Assistant (RAG System)

An **enterprise-grade Retrieval-Augmented Generation (RAG) system** that allows users to upload financial/compliance documents (PDFs) and ask questions grounded strictly in those documents — with **source attribution, document scoping, lifecycle management, and audit-safe responses**.

This project is designed as a **real-world backend + ML systems showcase**, not just a demo.

---

## 🚀 Key Features

- 📄 **PDF Upload & OCR Support**  
  Handles both text-based and scanned PDFs using OCR fallback.

- 🔍 **Retrieval-Augmented Question Answering (RAG)**  
  Answers are generated *only* from retrieved document chunks — no hallucinations.

- 🧠 **Local LLM Inference (Free)**  
  Uses **Google Flan-T5 (Small)** for on-prem / CPU-based inference.

- 📚 **Vector Search with FAISS**  
  High-performance semantic search using dense embeddings.

- 🏷 **Metadata-Aware Chunking**  
  Each chunk stores:
  - `doc_id`
  - `original_filename`
  - `chunk_id`

- 🎯 **Scoped Querying**  
  Ask across:
  - All documents, or
  - A specific selected document

- 📌 **Source Attribution**  
  Every answer returns:
  - Original document name
  - Chunk ID
  - Clean, sentence-complete excerpts

- 🗂 **Document Lifecycle Management**
  - Upload
  - List documents
  - Delete individual documents
  - Reset entire knowledge base
  - Time-based retention & cleanup

- 🖥 **Modular Frontend**
  - Streamlit UI
  - Fully decoupled from backend
  - Easily replaceable with React / Next.js

---

## 🧠 Tech Stack

### Backend
- **Python**
- **FastAPI**
- **Uvicorn**
- **LangChain (Runnable API)**
- **FAISS**
- **HuggingFace Transformers**
- **SentenceTransformers**
- **Tesseract OCR**

### Models
- **LLM:** `google/flan-t5-small`
- **Embeddings:** `intfloat/e5-small-v2`
- **(Optional Reranker):** `all-MiniLM-L6-v2`

### Frontend
- **Streamlit**

### Deployment
- **Backend:** Railway (VM-based, persistent disk)
- **Frontend:** Streamlit Community Cloud

---


## 🔄 RAG Flow (Step-by-Step)

1. User uploads a PDF  
2. Text is extracted (OCR if needed)  
3. Text is chunked with overlap  
4. Metadata is attached per chunk  
5. Embeddings are generated  
6. Chunks are stored in FAISS  
7. User asks a question  
8. Relevant chunks are retrieved  
9. LLM answers using retrieved context only  
10. Sources are returned with clean excerpts  

---

## 🧪 Quality & Safety Measures

- ✅ **E5 Query Normalization** (`query:` / `passage:` prefixes)
- ✅ **Adaptive Retrieval (`k`)**
- ✅ **Sentence-Complete Source Excerpts**
- ✅ **Explicit “Not Found” behavior**
- ✅ **No hallucinations outside context**

---

## 🧹 Retention & Cleanup

- Time-based retention policy (configurable TTL)
- Manual cleanup endpoint
- Safe FAISS rebuild strategy
- Explicit reset of entire knowledge base

---

## ⚙ CI/CD (Optional but Supported)

This project is **CI/CD-ready** using **GitHub Actions**.

Typical pipeline:
- On push to `main`
- Install dependencies
- Run basic checks
- Deploy backend (Railway)
- Deploy frontend (Streamlit Cloud auto-sync)

> CI/CD is intentionally lightweight to avoid overengineering while remaining production-aligned.

---

## 🌐 Live Demo

🚧 **Live demo link will be added here after deployment**

> The application can also be run locally or demonstrated on request.

---

## ▶️ Run Locally

### Backend
```bash
uvicorn app.main:app --reload
```

### Frontend
```streamlit
run frontend/streamlit_app/app.py
```


