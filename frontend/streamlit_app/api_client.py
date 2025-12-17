import requests
from config import BACKEND_URL


def upload_pdf(file):
    files = {"file": (file.name, file, "application/pdf")}
    response = requests.post(f"{BACKEND_URL}/upload", files=files)
    return response.json()


def ask_question(question, doc_id=None):
    payload = {"question": question}
    if doc_id:
        payload["doc_id"] = doc_id

    response = requests.post(f"{BACKEND_URL}/ask", json=payload)
    return response.json()
