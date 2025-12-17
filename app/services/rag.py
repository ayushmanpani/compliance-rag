from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os


def format_docs(docs):
    """Convert list of Documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGStore:
    def __init__(self, db_path="data/faiss_index"):
        self.db_path = db_path

        # Embeddings
        self.embedding = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2"
        )

        # Local LLM (FLAN-T5)
        model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial compliance assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present, say exactly:
  "The provided documents do not contain this information."

Context:
{context}

Question:
{question}

Answer (concise, factual):
"""
        )

    def load_store_if_exists(self):
        """
        Load FAISS only if it exists.
        Do NOT create empty FAISS.
        """
        if os.path.exists(self.db_path):
            print("[INFO] Loading existing FAISS index...")
            self.vstore = FAISS.load_local(
                self.db_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )

            self.retriever = self.vstore.as_retriever(search_kwargs={"k": 5})

            self.chain = (
                {
                    "context": self.retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            print("[INFO] FAISS index not found. RAG disabled until first upload.")
            self.vstore = None
            self.retriever = None
            self.chain = None



    def ask(self, question: str, doc_id: str = None):
    # 1️⃣ Choose retriever behavior
        if doc_id:
            retriever = self.vstore.as_retriever(
                search_kwargs={"k": 5, "filter": {"doc_id": doc_id}}
            )
        else:
            retriever = self.vstore.as_retriever(
                search_kwargs={"k": 5}
            )

        # 2️⃣ Retrieve ONCE
        docs = retriever.invoke(question)

        # 3️⃣ Combine retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in docs])

        # 4️⃣ Call LLM explicitly with the SAME context
        answer = self.llm.invoke(
            self.prompt.format(
                context=context,
                question=question
            )
        )

        # 5️⃣ Build sources from the SAME docs
        sources = [
            {
                "doc_id": doc.metadata.get("doc_id"),
                "original_filename": doc.metadata.get("original_filename"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "excerpt": doc.page_content[:300]
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    
    def add_documents(self, documents):
        """
        Create FAISS if it doesn't exist, otherwise append.
        """
        if self.vstore is None:
            print("[INFO] Creating FAISS index with first document batch...")
            self.vstore = FAISS.from_documents(documents, self.embedding)
        else:
            self.vstore.add_documents(documents)

        self.vstore.save_local(self.db_path)

        # Rebuild retriever & chain
        self.retriever = self.vstore.as_retriever(search_kwargs={"k": 5})

        self.chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

