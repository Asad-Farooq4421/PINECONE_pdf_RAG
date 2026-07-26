# Technical Report: Intermediate PDF RAG System Using Pinecone Vector DB

## 1. Executive Summary & System Architecture
- **Objective:** Design an un-hallucinated, traceable RAG system for PDF querying.
- **Pipeline Architecture:** PDF Ingestion → Chunking → Dense Embeddings → Pinecone Vector Indexing → Cosine Similarity Retrieval → Strict Prompt Injection → Groq LLM Generation.

## 2. Ingestion & Text Processing Strategy
- **Extraction:** Utilized `PyPDFLoader` with `extraction_mode="layout"` to preserve document formatting and tabular text structures.
- **Chunking:** Applied `RecursiveCharacterTextSplitter` with dynamic chunk size (500–1000 characters) and 50-character overlap to retain contextual continuity.
- **Embedding Model:** Standardized on `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
  - *Design Decision:* Selected for lightweight CPU inference speed, high semantic quality, and zero API cost compared to cloud embeddings.

## 3. Vector Database Integration (Pinecone)
- **Index Configuration:** Created a serverless index using Cosine metric and 384 dimensions.
- **Namespace Multi-Tenancy:** Isolated document indexes by assigning lowercased, formatted document names as unique Pinecone namespaces.
- **Metadata Management:** Each vector stores chunk metadata containing `page`, `document_name`, and raw text excerpts for UI traceability.

## 4. Retrieval, Confidence Thresholding, & Guardrails
- **Cosine Distance Retrieval:** Pulled Top-K chunks based on UI parameters.
- **Confidence Scoring Filter:** Chunks with a score below the dynamic threshold (e.g., 0.40) are discarded prior to model execution.
- **Hallucination Prevention:** Enforced zero temperature (`temperature=0.0`) on `llama-3.3-70b-versatile` combined with a strict system prompt instruction to output a standardized fallback string when context is absent.

## 5. Performance Analysis & Empirical Results
- **Fact Retrieval:** Successfully resolved precise financial figures (e.g., Rs. 220,000 package price)[cite: 1].
- **Tabular Data:** Accurately parsed valuation component tables[cite: 1].
- **Negative Testing:** Out-of-bounds queries correctly returned the fallback message without hallucinating facts[cite: 1].

## 6. Challenges Faced & Mitigation Strategies
- *Challenge:* Initial chunk truncation cut off key context sentences prior to prompt formatting.
  - *Fix:* Separated full context injection strings from visual display excerpts.
- *Challenge:* Duplicate vector namespaces from repeated uploads diluted search relevance.
  - *Fix:* Enforced namespace isolation and layout-aware PDF extraction.