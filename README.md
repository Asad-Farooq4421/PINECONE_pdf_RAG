# Spatial 3D PDF Intelligence Engine — Technical Documentation

This repository contains the complete implementation of a multi-tenant PDF Question-Answering system powered by Pinecone vector storage, Groq LLM inference, and a Three.js 3D user interface.

---

## 📂 Repository Structure

```text
PINECONE_pdf_RAG/
├── .env                    # Environment variables (API keys)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python package dependencies
├── styles.css              # Glassmorphism UI & dynamic answer card styles
├── app.py                  # Main Streamlit application entry point
│
├── backend/
│   ├── ingestion.py        # PDF extraction, chunking, & Pinecone vector indexing
│   └── rag_chain.py        # Cosine similarity retrieval & Groq LLM inference
│
└── frontend/
    ├── index.html          # Three.js WebGL canvas container
    ├── app.js              # Three.js 3D animation loop
    └── rocket_loader.html  # Interactive 3D rocket loader modal