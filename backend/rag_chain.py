import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load the matching embedding model (384 dimensions)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# System Prompt Enforcing Context Compliance Without Over-Strictness
STRICT_SYSTEM_PROMPT = """
You are an expert precise RAG assistant. Answer the user question based ONLY on the provided context snippets below.
Extract and state the facts directly and accurately from the context.

If the context snippets do NOT contain the information needed to answer the question, state EXACTLY:
"The answer is not available in the provided document."

Do not use outside knowledge, extrapolate, or make assumptions.

CONTEXT SNIPPETS:
{context}
"""

def generate_rag_response(
    query: str, 
    index_name: str, 
    namespace: str, 
    top_k: int = 3, 
    threshold: float = 0.4
) -> dict:
    """
    Performs cosine similarity search in Pinecone, filters results by confidence threshold,
    and queries Groq LLM with strict context guardrails.
    """
    # 1. Access the Pinecone VectorStore
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    # 2. Retrieve top_k chunks alongside similarity scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    # 3. Filter retrieved chunks by the confidence score threshold
    valid_chunks = [
        (doc, score) for doc, score in results_with_scores if score >= threshold
    ]

    # Return immediate fallback if no chunks meet the confidence score threshold
    if not valid_chunks:
        return {
            "answer": "The answer is not available in the provided document.",
            "sources": []
        }

    # 4. Assemble FULL context block for LLM & build traceable source metadata for UI
    context_str = ""
    sources = []

    for doc, score in valid_chunks:
        raw_page = doc.metadata.get("page", 0)
        display_page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        doc_name = doc.metadata.get("document_name", "Uploaded File")

        # FIX 1: Pass the FULL un-truncated page content to Groq!
        context_str += f"\n--- Excerpt (Page {display_page}) ---\n{doc.page_content}\n"
        
        # FIX 2: Only truncate text for the visual UI expander
        sources.append({
            "page": display_page,
            "doc_name": doc_name,
            "excerpt": doc.page_content[:200] + "...",
            "score": round(float(score), 4)
        })

    # 5. Send payload to Groq API with zero-temperature guardrails
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")

    client = Groq(api_key=groq_api_key)
    formatted_prompt = STRICT_SYSTEM_PROMPT.format(context=context_str)

    completion = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0,  # Zero temperature stops hallucinations
        max_tokens=300
    )

    answer_text = completion.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "sources": sources
    }