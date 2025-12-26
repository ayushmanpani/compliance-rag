import streamlit as st
from api_client import (
    upload_pdf,
    ask_question,
    list_documents,
    delete_document,
    reset_knowledge_base
)

# ======================================================
# Page Config
# ======================================================

st.set_page_config(
    page_title="Compliance RAG Assistant",
    layout="wide"
)

st.title("📄 Financial Compliance Assistant")
st.caption(
    "Ask questions grounded strictly in your uploaded compliance documents"
)

# ======================================================
# Session State
# ======================================================

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

# ======================================================
# Document Management
# ======================================================

st.header("📁 Document Management")

col1, col2 = st.columns(2)

# ---------------- Upload ---------------- #

with col1:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("➕ Ingest Document"):
            with st.spinner("Processing document..."):
                result = upload_pdf(uploaded_file)
                st.success("Document ingested successfully")
                st.json(result)
                st.rerun()

# ---------------- List + Delete ---------------- #

with col2:
    documents = []
    try:
        documents = list_documents()
    except Exception as e:
        st.error("Failed to load documents")
        st.write(e)

    if documents:
        doc_map = {
            doc["original_filename"]: doc["doc_id"]
            for doc in documents
        }

        selected_doc_name = st.selectbox(
            "Select document",
            options=["All documents"] + list(doc_map.keys())
        )

        selected_doc_id = (
            None
            if selected_doc_name == "All documents"
            else doc_map[selected_doc_name]
        )

        if selected_doc_id:
            if st.button("🗑 Delete selected document"):
                st.session_state.confirm_delete = True

            if st.session_state.confirm_delete:
                st.warning("This action is permanent.")
                if st.checkbox("I understand and want to delete this document"):
                    with st.spinner(
                        "Deleting document and rebuilding index..."
                    ):
                        result = delete_document(selected_doc_id)
                        st.success("Document deleted successfully")
                        st.json(result)
                        st.session_state.confirm_delete = False
                        st.rerun()
    else:
        selected_doc_id = None
        st.info("No documents uploaded yet.")

st.divider()

# ======================================================
# Question Answering
# ======================================================

st.header("❓ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What are the KYC requirements?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            response = ask_question(
                question,
                selected_doc_id
            )

            if "answer" in response:
                st.subheader("✅ Answer")
                st.write(response["answer"])

                st.subheader("📌 Sources")
                if response.get("sources"):
                    for src in response["sources"]:
                        with st.expander(
                            f"{src.get('original_filename')} "
                            f"(chunk {src.get('chunk_id')})"
                        ):
                            st.write(src.get("excerpt", ""))
                else:
                    st.info("No sources returned.")
            else:
                st.error("Backend did not return a valid answer.")
                st.json(response)

st.divider()

# ======================================================
# Danger Zone — Reset
# ======================================================

st.header("⚠️ Danger Zone")

st.warning(
    "This will permanently delete ALL documents, metadata, "
    "and vector embeddings."
)

if st.button("🚨 Reset Knowledge Base"):
    st.session_state.confirm_reset = True

if st.session_state.confirm_reset:
    if st.checkbox("I understand and want to reset everything"):
        with st.spinner("Resetting entire knowledge base..."):
            result = reset_knowledge_base()
            st.success("Knowledge base reset successfully")
            st.json(result)
            st.session_state.confirm_reset = False
            st.rerun()
