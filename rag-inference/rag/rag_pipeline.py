import os
os.environ["HF_HOME"] = "/tmp/huggingface"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from .config import FAISS_DIR

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2"
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

def load_vectorstore():
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


def ask_question(question: str):
    vectorstore = load_vectorstore()
    expanded_query = f"query: {question}"
    docs = vectorstore.similarity_search(expanded_query, k=3)


    context = "\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 50:
        return "I do not find this information in the provided documents."
    
    prompt = f"""
    You are a compliance assistant.

    You must answer ONLY using the provided context.

    Rules:
    - If the answer is not explicitly stated in the context, respond exactly with:
    "I do not find this information in the provided documents."
    - Do NOT use outside knowledge.
    - Do NOT infer.
    - Do NOT guess.
    - If any exact keyword is not found in context dont make up new words.

    Context:
    {context}

    Question:
    {question}
    """


    
    answer = llm.invoke(prompt)


    return answer

