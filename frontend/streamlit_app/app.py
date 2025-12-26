import streamlit as st
from api_client import (
    upload_pdf,
    ask_question,
    list_documents,
    delete_document,
    reset_knowledge_base
)

st.set_page_config(
    page_title="Compliance RAG Assistant",
    layout="wide"
)

st.title("📄 Financial Compliance Assistant")

# ======================================================
# Upload Section
# ======================================================

st.header("Upload Compliance Document")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if uploaded_file:
    if st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            result = upload_pdf(uploaded_file)
            st.success("Document ingested successfully")
            st.json(result)
            st.session_state.uploaded = True

st.divider()

# ======================================================
# Question Section
# ======================================================

st.header("Ask a Question")

question = st.text_input("Enter your question")

documents = []
try:
    documents = list_documents()
except Exception as e:
    st.error("Failed to load documents")
    st.write(e)

doc_options = {"All documents": None}
for doc in documents:
    doc_options[doc["original_filename"]] = doc["doc_id"]

selected_doc = st.selectbox(
    "Select document (optional)",
    options=list(doc_options.keys())
)

selected_doc_id = doc_options[selected_doc]

# ---------------- Delete Button ---------------- #

if selected_doc_id:
    if st.button("🗑 Delete selected document"):
        with st.spinner("Deleting document and rebuilding index..."):
            result = delete_document(selected_doc_id)
            st.success("Document deleted successfully")
            st.json(result)
            st.rerun()

st.divider()

# ---------------- Ask Button ---------------- #

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            response = ask_question(
                question,
                selected_doc_id if selected_doc_id else None
            )

            if "answer" in response:
                st.subheader("Answer")
                st.write(response["answer"])

                st.subheader("Sources")
                for src in response.get("sources", []):
                    with st.expander(
                        f"{src.get('original_filename')} "
                        f"(page {src.get('page', '?')})"
                    ):
                        st.write(src.get("excerpt", ""))
            else:
                st.error("Backend did not return an answer.")
                st.json(response)

st.divider()

# ======================================================
# Reset Section (Danger Zone)
# ======================================================

st.header("⚠️ Danger Zone")

st.warning(
    "This will permanently delete ALL documents, metadata, "
    "and vector embeddings."
)

if st.button("🚨 Reset Knowledge Base"):
    with st.spinner("Resetting entire knowledge base..."):
        result = reset_knowledge_base()
        st.success("Knowledge base reset successfully")
        st.json(result)
        st.rerun()
