import requests
from config import BACKEND_URL
import requests
from io import BytesIO


BASE_URL = "http://localhost:8000/api"



def upload_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()  # bytes
    file_buffer = BytesIO(file_bytes)      # file-like object with seek()
    response = requests.post(
        f"{BASE_URL}/upload",
        files={
            "up_file": (
                uploaded_file.name,
                file_buffer,                # ✅ file-like object
                "application/pdf"
            )
        }
    )
    return response.json()



def ask_question(question, doc_id=None):
    payload = {"question": question}
    if doc_id:
        payload["doc_id"] = doc_id

    response = requests.post(f"{BACKEND_URL}/ask", json=payload)
    return response.json()

def list_documents():
    response = requests.get(f"{BACKEND_URL}/documents")
    return response.json()

