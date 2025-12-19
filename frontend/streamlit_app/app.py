import streamlit as st
from api_client import upload_pdf, ask_question, list_documents


st.set_page_config(page_title="Compliance RAG Assistant", layout="wide")

st.title("📄 Financial Compliance Assistant")

# ---------- Upload Section ----------

st.header("Upload Compliance Document")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if uploaded_file:
    if st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            result = upload_pdf(uploaded_file)
            st.success("Document ingested successfully")
            st.json(result)


# ---------- Question Section ----------
st.header("Ask a Question")

question = st.text_input("Enter your question")


docus = []  

try:
    docus = list_documents()
except Exception as e:
    st.error("Failed to load documents")
    st.write(e)


doc_options = {"All documents": None}
if docus:
    for doc in docus:
        doc_options[doc["original_filename"]] = doc["doc_id"]

selected_doc = st.selectbox(
    "Select document (optional)",
    options=list(doc_options.keys())
)

selected_doc_id = doc_options[selected_doc]


if st.button("Ask"):
    if question.strip():
        with st.spinner("Generating answer..."):
            response = ask_question(question, selected_doc_id if selected_doc_id else None)

            if "answer" in response:
                st.subheader("Answer")
                st.write(response["answer"])

                st.subheader("Sources")
                for src in response.get("sources", []):
                    with st.expander(
                        f"{src.get('original_filename')} (page {src.get('page', '?')})"
                    ):
                        st.write(src.get("excerpt", ""))
            else:
                st.error("Backend did not return an answer.")
                st.json(response)
    else:
        st.warning("Please enter a question.")
