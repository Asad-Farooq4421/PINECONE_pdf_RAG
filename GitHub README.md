# ⚡ Spatial 3D PDF Intelligence Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-000000.svg)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3_70B-f3603f.svg)](https://groq.com/)

An enterprise-grade, multi-tenant **Retrieval-Augmented Generation (RAG)** platform featuring layout-aware PDF ingestion, multi-tenant vector namespace isolation, guarded zero-temperature LLM inference, and an interactive 3D spatial user interface built with **Streamlit** and **Three.js**.

---

## 📸 Interface & Highlights

* **3D Interactive Canvas:** Interactive WebGL particle constellation and vector core floating behind the UI.
* **Rocket Ingestion Loader:** Live 3D animated rocket launch modal providing visual feedback during PDF vectorization.
* **Dynamic Color-Coded Answers:** Emerald green highlight cards for verified extractions and crimson red alert cards for out-of-scope or missing information.
* **Traceable Attributions:** Confidence scores and exact source page excerpts provided for every answer.

---

## 🏗️ Architecture Blueprint

```text
                                  +-----------------------+
                                  |   Uploaded PDF File   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | PyPDF Layout Extractor|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Recursive Text Chunk  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | MiniLM-L6-v2 Embedder |
                                  +-----------+-----------+
                                              |
                                              v
  +-----------------------+       +-----------------------+
  | User Query Input      | ----> | Pinecone Serverless   |
  +-----------------------+       | (Isolated Namespace)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Groq Llama-3.3-70B    |
                                  | Guarded Inference     |
                                  +-----------------------+