import os
import tempfile
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

def process_and_index_pdf(pdf_file, index_name="pdf-rag-index", chunk_size=1000, chunk_overlap=50):
    """
    Extracts text from an uploaded PDF, chunks it using layout-aware parsing,
    and upserts dense 384d vector embeddings into a Pinecone namespace.
    """
    # 1. SaveUploaded File to Temp Location for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        # 2. Extract Text with Layout Preservation
        loader = PyPDFLoader(tmp_path, extraction_mode="layout")
        docs = loader.load()

        # Sanitize filename for namespace isolation
        doc_name = getattr(pdf_file, "name", "document.pdf")
        clean_namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name).lower()

        # Enrich chunk metadata
        for doc in docs:
            doc.metadata["source_name"] = doc_name

        # 3. Recursive Character Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)

        # 4. Generate Dense Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 5. Connect & Index into Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)

        # Create Index if missing
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = pc.Index(index_name)

        # Batch Upsert Vectors
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector_val = embeddings.embed_query(chunk.page_content)
            vectors_to_upsert.append({
                "id": f"{clean_namespace}_chunk_{i}",
                "values": vector_val,
                "metadata": {
                    "text": chunk.page_content,
                    "page": chunk.metadata.get("page", 0) + 1,
                    "source_name": doc_name
                }
            })

        # Upsert into isolated namespace
        index.upsert(vectors=vectors_to_upsert, namespace=clean_namespace)

        return clean_namespace, len(chunks)

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)