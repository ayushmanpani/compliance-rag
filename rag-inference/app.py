from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from rag.ingest import ingest_pdf
from rag.rag_pipeline import ask_question

app = FastAPI(title="Compliance RAG API")


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    await ingest_pdf(file)
    return {"message": "Document indexed successfully"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask(req: QuestionRequest):
    answer = ask_question(req.question)
    return {"answer": answer}
