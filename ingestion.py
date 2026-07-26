import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Local Embedding Model (SentenceTransformers MiniLM)
# Generates 384-dimensional dense vector embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def process_and_index_pdf(
    pdf_path: str, 
    index_name: str, 
    namespace: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> int:
    """
    Extracts text from PDF, splits into chunks with page metadata, 
    generates embeddings, and uploads vectors into Pinecone under a given namespace.
    """
    # 1. Extract text while preserving page number metadata
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    doc_name = os.path.basename(pdf_path)
    for doc in raw_docs:
        doc.metadata["document_name"] = doc_name

    # 2. Apply intelligent recursive text chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(raw_docs)

    # 3. Initialize Pinecone Client & Ensure Index Exists
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from environment variables.")

    pc = Pinecone(api_key=api_key)
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM-L6-v2 produces 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 4. Upsert vectors + metadata into Pinecone Index under specified Namespace
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    return len(chunks)