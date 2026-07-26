# 📚 Multi-Tenant PDF RAG System with Pinecone & Groq

A robust, enterprise-grade Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **Pinecone Vector DB**, and **Groq (Llama 3.3)**. 

This platform enables users to upload PDF documents, index their content into isolated vector namespaces, and query document content with zero hallucinations, strict source attributions, and real-time parameter tuning.

---

## 🌟 Key Features

* **Layout-Aware PDF Ingestion:** Utilizes `PyPDFLoader` in layout mode to preserve spatial document layout, section structures, and complex data tables.
* **Vector Multi-Tenancy:** Partitioned document indexing via Pinecone namespaces to prevent cross-document data leakage.
* **Local Embedding Inference:** Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` locally on CPU.
* **Traceable Source Attribution:** Displays exact page numbers, similarity confidence scores, and raw text excerpts for every retrieved claim.
* **Zero-Hallucination Guardrails:** Integrates zero-temperature Groq inference (`llama-3.3-70b-versatile`) coupled with strict prompt fallback rules.
* **Dynamic Hyperparameter Sliders:** Allows real-time adjustment of **Chunk Size**, **Top-K Retrieval Counts**, and **Cosine Similarity Thresholds**.
* **Query Session History:** Tracks and logs previous QA pairs within the active session.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Client])
    
    subgraph Streamlit_UI ["Streamlit Interface (app.py)"]
        Sidebar["Sidebar Controls<br/>(Chunk Size, Top-K, Threshold)"]
        PDF_Up["PDF File Uploader"]
        Chat_Input["Query Input & History Log"]
    end

    subgraph Document_Ingestion ["Ingestion Pipeline (ingestion.py)"]
        PDF_Load["PyPDFLoader<br/>(layout mode)"]
        Splitter["RecursiveCharacterTextSplitter"]
        Embed_Local["SentenceTransformers<br/>(all-MiniLM-L6-v2)"]
    end

    subgraph Vector_DB ["Pinecone Vector Index"]
        Index["Index: pdf-rag-index"]
        Namespace["Namespace Partitioning"]
    end

    subgraph Query_Chain ["RAG Engine (rag_chain.py)"]
        Query_Embed["Embed User Query"]
        Similarity_Search["Cosine Similarity Search"]
        Threshold_Filter{"Score >= Threshold?"}
        Fallback["Return: 'Answer not available...'"]
        Prompt_Construct["Assemble Full Context"]
        Groq_LLM["Groq API<br/>(llama-3.3-70b-versatile)<br/>temp=0.0"]
    end

    %% Flow Connections
    User --> PDF_Up
    User --> Chat_Input
    
    PDF_Up --> PDF_Load
    PDF_Load --> Splitter
    Splitter --> Embed_Local
    Embed_Local -->|Upsert Chunks + Metadata| Namespace
    Namespace --> Index

    Chat_Input --> Query_Embed
    Query_Embed --> Similarity_Search
    Similarity_Search -->|Fetch Vectors| Index
    Similarity_Search --> Threshold_Filter
    
    Threshold_Filter -->|No| Fallback
    Threshold_Filter -->|Yes| Prompt_Construct
    
    Prompt_Construct --> Groq_LLM
    Groq_LLM -->|Response + Sources| Chat_Input
    Fallback --> Chat_Input