import os

# Base directory logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
