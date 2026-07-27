import os
import time
from dotenv import load_dotenv  # Add this
load_dotenv()
import streamlit as st
import streamlit.components.v1 as components
from backend.ingestion import process_and_index_pdf
from backend.rag_chain import generate_rag_response

# 1. Page Configuration
st.set_page_config(
    page_title="Spatial PDF Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

INDEX_NAME = "pdf-rag-index"

# 2. Asset Loader
def load_asset(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Inject Custom Styling
css_content = load_asset("styles.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# 3. Session State Initialization
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = None

# Header Title Banner
st.markdown("""
<div class="main-header-container">
    <h1 class="main-header-title">⚡ Spatial PDF Intelligence Engine</h1>
    <p class="main-header-subtitle">
        Enterprise multi-tenant RAG platform powered by Pinecone serverless vector isolation, Groq zero-temperature guardrails, and real-time source attributions.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# MAIN PAGE GRID CONTROLS (Replacing Left-to-Right Sidebar Menu)
# ==========================================================================
st.markdown("### 🛠️ Workspace Controls & Document Ingestion")

col_upload, col_params = st.columns([1.2, 1], gap="medium")

with col_upload:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown("#### 📄 Document Upload")
    uploaded_file = st.file_uploader("Select PDF file to vectorize:", type=["pdf"])
    
    # Active Namespace Status Badge
    if st.session_state.current_namespace:
        st.markdown(f'**Active Index:** <span class="status-pill-active">✓ {st.session_state.current_namespace}</span>', unsafe_allow_html=True)
    else:
        st.markdown('**Active Index:** <span class="status-pill-inactive">✕ No Document Indexed</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_params:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Vector Hyperparameters")
    chunk_size = st.slider("Chunk Size (Chars)", min_value=300, max_value=1500, value=1000, step=50)
    top_k = st.slider("Top-K Retrieved Nodes", min_value=1, max_value=10, value=5, step=1)
    threshold = st.slider("Similarity Confidence Threshold", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    st.markdown('</div>', unsafe_allow_html=True)

# Trigger Button for Ingestion
if st.button("🚀 Upload & Index PDF Vector Space"):
    if uploaded_file is not None:
        # Show Interactive 3D Rocket Launch Modal
        rocket_html = load_asset(os.path.join("frontend", "rocket_loader.html"))
        rocket_placeholder = st.empty()
        
        with rocket_placeholder.container():
            if rocket_html:
                st.components.v1.html(rocket_html, height=230)
            st.info("Ingesting spatial document coordinates into Pinecone...")

        # Process Ingestion (Keyword matched: pdf_file)
        namespace, num_chunks = process_and_index_pdf(
            pdf_file=uploaded_file,
            index_name=INDEX_NAME,
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        st.session_state.current_namespace = namespace
        time.sleep(1) # Brief pause for visual rocket feedback
        rocket_placeholder.empty() # Clear rocket popup
        st.success(f"Successfully vectorized and indexed {num_chunks} chunks into namespace: `{namespace}`!")
    else:
        st.error("Please attach a valid PDF document before launching ingestion.")

st.markdown("---")

# ==========================================================================
# QUERY & RESPONSE CANVAS
# ==========================================================================
st.markdown("### 💬 Vector Intelligence Search")

query_input = st.text_input(
    "Query the document factual knowledge base:",
    placeholder="e.g., What is the total package price or contract timeframe?"
)

if query_input:
    if not st.session_state.current_namespace:
        st.warning("Please upload and index a PDF document before asking questions.")
    else:
        with st.spinner("Searching vector nodes & running guarded inference..."):
            result = generate_rag_response(
                query=query_input,
                index_name=INDEX_NAME,
                namespace=st.session_state.current_namespace,
                top_k=top_k,
                threshold=threshold
            )
            
            # Save query into session history log
            st.session_state.session_history.insert(0, {
                "query": query_input,
                "answer": result["answer"],
                "sources": result["sources"]
            })

# Display Answers with Dynamic Color Coding
if st.session_state.session_history:
    latest = st.session_state.session_history[0]
    answer_text = latest["answer"]
    
    st.markdown("#### 🎯 Generated Response")
    
    # Check if answer is un-hallucinated fallback
    if "answer is not available" in answer_text.lower():
        st.markdown(f'<div class="answer-card-notfound"><b>⚠️ Out of Scope / Missing Information:</b><br>{answer_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="answer-card-success"><b>✅ Verified Fact Extracted:</b><br>{answer_text}</div>', unsafe_allow_html=True)
    
    # Traceable Source Attributions
    if latest["sources"]:
        st.markdown("#### 📌 Traceable Source Attribution")
        for idx, src in enumerate(latest["sources"], 1):
            with st.expander(f"Source Node {idx} | Page {src['page']} | Confidence Score: {src['score']}"):
                st.markdown(f"**Document Name:** `{src['doc_name']}`")
                st.markdown(f"**Extracted Text Excerpt:**")
                st.info(f"\"{src['excerpt']}\"")

# Session History
if len(st.session_state.session_history) > 1:
    st.markdown("---")
    with st.expander("📜 Session Query History"):
        for item in st.session_state.session_history[1:]:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")