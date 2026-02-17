import os
os.environ["HF_HOME"] = "/tmp/huggingface"

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import FAISS_DIR


# ----------------------------
# Groq Client
# ----------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# ----------------------------
# Embeddings (lightweight)
# ----------------------------
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            # model_name="intfloat/e5-small-v2"
        )
    return embeddings




# ----------------------------
# Load FAISS Vectorstore
# ----------------------------
def load_vectorstore():
    return FAISS.load_local(
        FAISS_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


# ----------------------------
# Call Groq LLM
# ----------------------------
def call_llm(context: str, question: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance assistant.\n\n"
                    "You must answer ONLY using the provided context.\n\n"
                    "Rules:\n"
                    # "- If the answer is not explicitly stated in the context, respond exactly with:\n"
                    # "\"I do not find this information in the provided documents.\"\n"
                    "- Do NOT use outside knowledge.\n"
                    "- Do NOT infer.\n"
                    "- Do NOT guess.\n"
                    "- If any exact keyword is not found in context, do not invent new words."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ----------------------------
# Main RAG Function
# ----------------------------
def ask_question(question: str):
    vectorstore = load_vectorstore()

    expanded_query = f"query: {question}"
    docs = vectorstore.similarity_search(expanded_query, k=3)
    
    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i} ---")
        print(doc.page_content[:300])


    context = "\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 50:
        return "I do not find this information in the provided documents."

    answer = call_llm(context, question)

    return answer
