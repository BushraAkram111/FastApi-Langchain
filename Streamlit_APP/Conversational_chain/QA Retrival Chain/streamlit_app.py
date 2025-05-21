import streamlit as st
from Conversational_chain.utils import (
    load_split_pdf_file,
    load_split_docx_file,
    load_split_text_file,
    QA_Chain_Retrieval,
    QdrantInsertRetrievalAll
)
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "testing"

# Initialize embeddings and Qdrant handler
embeddings = OpenAIEmbeddings()
qdrant_handler = QdrantInsertRetrievalAll(api_key=QDRANT_API_KEY, url=QDRANT_URL)

# Streamlit UI
st.set_page_config(page_title="📁 AI Document Q&A", layout="centered")
st.title("📄 Upload Documents & Ask Questions")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True)

all_chunks = []

if uploaded_files:
    st.subheader("📂 File Upload Summary")
    for file in uploaded_files:
        ext = file.name.split('.')[-1].lower()
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        if ext == "pdf":
            chunks = load_split_pdf_file(temp_path)
        elif ext == "docx":
            chunks = load_split_docx_file(temp_path)
        elif ext == "txt":
            chunks = load_split_text_file(temp_path)
        else:
            st.warning(f"❌ Unsupported file type: {file.name}")
            continue

        st.success(f"✅ {file.name} processed.")
        all_chunks.extend(chunks)
        os.remove(temp_path)

    if all_chunks:
        try:
            qdrant_handler.insertion(all_chunks, embeddings, COLLECTION_NAME)
            st.success("🎉 All documents embedded and stored in vector DB.")
        except Exception as e:
            st.error(f"❌ Embedding failed: {str(e)}")

# Query input
st.divider()
query = st.text_input("💬 Ask something about the uploaded documents:")

if query:
    with st.spinner("Thinking..."):
        try:
            vector_store = qdrant_handler.retrieval(collection_name=COLLECTION_NAME, embeddings=embeddings)
            response = QA_Chain_Retrieval(query=query, qdrant_vectordb=vector_store)
            st.markdown("### 🧠 Answer:")
            st.write(response.content)
        except Exception as e:
            st.error(f"❌ Retrieval error: {str(e)}")
