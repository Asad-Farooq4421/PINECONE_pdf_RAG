import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from ingestion import process_and_index_pdf
from rag_chain import generate_rag_response

# Load Environment Variables from .env file
load_dotenv()

st.set_page_config(
    page_title="RAG System | Pinecone & Groq",
    page_icon="📚",
    layout="wide"
)

INDEX_NAME = "pdf-rag-index"

# Initialize Session Memory for Query History
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# --- SIDEBAR: Document Upload & Controls ---
with st.sidebar:
    st.header("⚙️ Configuration & Indexing")
    
    uploaded_file = st.file_uploader("Upload PDF Document (Max 20MB)", type=["pdf"])
    
    st.divider()
    st.subheader("🛠️ Intermediate Hyperparameters")
    
    chunk_size = st.slider("Chunk Size (Characters)", min_value=200, max_value=1000, value=500, step=50)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=3)
    similarity_threshold = st.slider("Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    if uploaded_file and st.button("Upload & Index PDF", use_container_width=True):
        # 20MB Limit Validation Check
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("File size exceeds the 20 MB limit.")
        else:
            with st.spinner("Extracting text, chunking, and upserting vectors to Pinecone..."):
                # Save uploaded file temporarily to disk for PyPDF processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Clean namespace identifier from file name
                namespace = uploaded_file.name.replace(" ", "_").lower()
                st.session_state.namespace = namespace
                st.session_state.doc_name = uploaded_file.name

                try:
                    num_chunks = process_and_index_pdf(
                        pdf_path=tmp_path,
                        index_name=INDEX_NAME,
                        namespace=namespace,
                        chunk_size=chunk_size
                    )
                    st.success(f"Success! Indexed {num_chunks} chunks under namespace: `{namespace}`")
                except Exception as e:
                    st.error(f"Error indexing document: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# --- MAIN DASHBOARD CONTENT ---
st.title("📚 PDF Retrieval-Augmented Generation (RAG) System")
st.caption("Powered by Pinecone Vector DB, SentenceTransformers, and Groq (Llama 3.3)")

if "doc_name" in st.session_state:
    st.info(f"📄 Active Document: **{st.session_state.doc_name}**")
else:
    st.warning("👈 Please upload and index a PDF from the sidebar to start asking questions.")

query = st.text_input("Ask a question strictly based on the PDF content:")

if query:
    if "namespace" not in st.session_state:
        st.error("Please upload and index a PDF document before asking questions.")
    else:
        with st.spinner("Searching Pinecone index & generating answer..."):
            try:
                response = generate_rag_response(
                    query=query,
                    index_name=INDEX_NAME,
                    namespace=st.session_state.namespace,
                    top_k=top_k,
                    threshold=similarity_threshold
                )

                # Update Session History
                st.session_state.query_history.append({
                    "query": query,
                    "answer": response["answer"],
                    "sources": response["sources"]
                })

                # Display Model Response
                st.markdown("### 🤖 Answer")
                st.write(response["answer"])

                # Display Source Attribution
                if response["sources"]:
                    st.markdown("### 📌 Traceable Source Attribution")
                    for idx, src in enumerate(response["sources"], 1):
                        with st.expander(f"Source {idx} | Page {src['page']} | Confidence Score: {src['score']}"):
                            st.write(f"**Document Name:** {src['doc_name']}")
                            st.write(f"**Excerpt:** {src['excerpt']}")

            except Exception as e:
                st.error(f"An error occurred during retrieval: {str(e)}")

# --- QUERY HISTORY SECTION ---
if st.session_state.query_history:
    st.divider()
    with st.expander("📜 Session Query History Log"):
        for item in reversed(st.session_state.query_history):
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")
            st.caption("---")