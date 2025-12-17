import streamlit as st
from api_client import upload_pdf, ask_question

st.set_page_config(page_title="Compliance RAG Assistant", layout="wide")

st.title("📄 Financial Compliance Assistant")

# ---------- Upload Section ----------
st.header("Upload Compliance Document")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    if st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            result = upload_pdf(uploaded_file)
            st.success("Document ingested successfully")
            st.json(result)

# ---------- Question Section ----------
st.header("Ask a Question")

question = st.text_input("Enter your question")

doc_id = st.text_input(
    "Optional: Document ID (leave empty to search all documents)"
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Generating answer..."):
            response = ask_question(question, doc_id if doc_id else None)

            st.subheader("Answer")
            st.write(response["answer"])

            st.subheader("Sources")
            for src in response["sources"]:
                with st.expander(
                    f"{src['original_filename']} (chunk {src['chunk_id']})"
                ):
                    st.write(src["excerpt"])
    else:
        st.warning("Please enter a question.")
