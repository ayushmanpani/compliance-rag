# app/main.py
from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="Compliance RAG Assistant")

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"status": "ok", "service": "compliance-rag"}
