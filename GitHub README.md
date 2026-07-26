graph TD
    %% Node Definitions
    User([User Client])
    
    subgraph Streamlit_UI ["Streamlit Interface (app.py)"]
        Sidebar["Sidebar Controls<br/>(Chunk Size, Top-K, Threshold)"]
        PDF_Up["PDF File Uploader"]
        Chat_Input["Query Input & History"]
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
        Similarity_Search["Cosine Similarity Search<br/>(Top-K Chunks)"]
        Threshold_Filter{"Score >= Threshold?"}
        Fallback["Return: 'Answer not available...'"]
        Prompt_Construct["Assemble Context String"]
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