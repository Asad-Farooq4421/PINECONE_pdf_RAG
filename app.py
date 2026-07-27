import os
import streamlit as st
from ingestion import process_and_index_pdf
from rag_chain import generate_rag_response

# 1. Page Configuration
st.set_page_config(
    page_title="PDF RAG Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Helper function to load external CSS & HTML assets
def load_asset(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Inject CSS
css_content = load_asset("styles.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Render Header HTML
header_html = load_asset("assets/header.html")
if header_html:
    st.markdown(header_html, unsafe_allow_html=True)

# Initialize Session States
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = None

INDEX_NAME = "pdf-rag-index"

# 3. Sidebar Controls
with st.sidebar:
    st.markdown("### 🛠️ Ingestion & Retrieval Tuning")
    
    uploaded_file = st.file_uploader("Upload Target PDF Document", type=["pdf"])
    
    st.markdown("---")
    st.markdown("#### Hyperparameters")
    
    chunk_size = st.slider("Chunk Size (Characters)", min_value=300, max_value=1500, value=1000, step=50)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5, step=1)
    threshold = st.slider("Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    
    st.markdown("---")
    
    if st.button("Upload & Index PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF layout & generating Pinecone vectors..."):
                namespace, num_chunks = process_and_index_pdf(
                    pdf_file=uploaded_file,
                    index_name=INDEX_NAME,
                    chunk_size=chunk_size,
                    chunk_overlap=50
                )
                st.session_state.current_namespace = namespace
                st.success(f"Successfully indexed into namespace: `{namespace}` ({num_chunks} chunks)")
        else:
            st.error("Please select a PDF file before indexing.")

# 4. Query & Results Main Canvas
query_input = st.text_input(
    "Ask a question strictly based on the PDF content:",
    placeholder="e.g., What is the comprehensive package price?"
)

if query_input:
    if not st.session_state.current_namespace:
        st.warning("Please upload and index a PDF document before asking questions.")
    else:
        with st.spinner("Searching vectors & generating response..."):
            result = generate_rag_response(
                query=query_input,
                index_name=INDEX_NAME,
                namespace=st.session_state.current_namespace,
                top_k=top_k,
                threshold=threshold
            )
            
            # Save to Session Log
            st.session_state.session_history.insert(0, {
                "query": query_input,
                "answer": result["answer"],
                "sources": result["sources"]
            })

# Display Current Response
if st.session_state.session_history:
    latest = st.session_state.session_history[0]
    
    st.markdown("### 🎯 Answer")
    st.markdown(f'<div class="answer-card">{latest["answer"]}</div>', unsafe_allow_html=True)
    
    if latest["sources"]:
        st.markdown("### 📌 Traceable Source Attribution")
        for idx, src in enumerate(latest["sources"], 1):
            with st.expander(f"Source {idx} | Page {src['page']} | Confidence Score: {src['score']}"):
                st.markdown(f"**Document Name:** {src['doc_name']}")
                st.markdown(f"**Excerpt:** {src['excerpt']}")

# Session History Log
if len(st.session_state.session_history) > 1:
    st.markdown("---")
    with st.expander("📜 Session Query History Log"):
        for item in st.session_state.session_history[1:]:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")