from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DEFAULT_FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index")

def format_docs(docs):
    """Convert list of Documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGStore:
    def __init__(self, db_path: str = DEFAULT_FAISS_PATH):
        self.db_path = db_path

        # Embeddings
        self.embedding = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        # Local LLM (FLAN-T5)
        model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0,
            truncation=True 
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the question using ONLY the context below.\n"
                "If the answer is not in the context, say:\n"
                "\"NOT_FOUND\"\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            )
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
        print("\n[DEBUG] Requested doc_id:", repr(doc_id))

        # inspect one stored document's metadata
        if self.vstore:
            sample = self.vstore.docstore._dict
            first_doc = next(iter(sample.values()))
            print("[DEBUG] Sample stored metadata:", first_doc.metadata)

    
    # 1️⃣ Choose retriever behavior
        if doc_id:
            retriever = self.vstore.as_retriever(
                search_kwargs={"k": 3, "filter": {"doc_id": doc_id}}
            )
        else:
            retriever = self.vstore.as_retriever(
                search_kwargs={"k": 3}
            )

        # 2️⃣ Retrieve ONCE
        expanded_query = (
            f"query: {question}. "
            "Answer from sections related to KYC, Customer Due Diligence, "
            "Ongoing Due Diligence, or updation of records."
        )
        docs = retriever.invoke(expanded_query)
        if not docs:
            fallback_query = (
                "query: updation of KYC records ongoing due diligence risk category"
            )
            docs = retriever.invoke(fallback_query)


        # 3️⃣ Combine retrieved docs into context
        MAX_CONTEXT_CHARS = 1200  # safe for FLAN-T5-small

        context = "\n\n".join([doc.page_content for doc in docs])
        context = context[:MAX_CONTEXT_CHARS]


        # 4️⃣ Call LLM explicitly with the SAME context
        raw_answer = self.llm.invoke(
            self.prompt.format(
                context=context,
                question=question
            )
        )
        
        # 🔧 Normalize HuggingFacePipeline output
        if isinstance(raw_answer, list):
            answer = raw_answer[0].get("generated_text", "").strip()
        elif isinstance(raw_answer, dict):
            answer = raw_answer.get("generated_text", "").strip()
        else:
            answer = str(raw_answer).strip()
            
        if answer == "NOT_FOUND":
            answer = "The provided documents do not contain this information."

        # 5️⃣ Build sources from the SAME docs
        sources = [
            {
                "doc_id": doc.metadata.get("doc_id"),
                "original_filename": doc.metadata.get("original_filename"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:300]
            }
            for doc in docs
        ]

        return {
            "answer": answer if answer else "The provided documents do not contain this information.",
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

