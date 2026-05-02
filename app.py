
"""
FastAPI web UI for PDF RAG (Educational):
  - Upload a PDF → embed & store in Qdrant.
  - Query the stored embeddings via GPT‑4o‑mini.
  - Inspect chunks, visualize embedding vectors, and see deduplication in action.
"""

import hashlib
import io
import os
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ---------- Configuration ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_ai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "❌ OPENAI_API_KEY environment variable is missing. "
        "Set it before running the app."
    )

# ---------- FastAPI ----------
app = FastAPI(title="PDF RAG Service", description="Upload PDF and query via GPT‑4o‑mini")

# Shared state for UI polling & education
app.state.upload_status = {
    "status": "idle",
    "message": "No PDF uploaded yet.",
    "chunks_count": 0,
    "dedup_count": 0,
    "skipped": False,
}
app.state.last_chunks: List[str] = []
app.state.seen_hashes: set = set()  # Document-level dedup

# ---------- Helper Functions ----------
def compute_file_hash(pdf_bytes: bytes) -> str:
    """Compute MD5 hash of PDF bytes for deduplication."""
    return hashlib.md5(pdf_bytes).hexdigest()


def load_and_chunk(pdf_bytes: bytes) -> List[Document]:
    """Load PDF from bytes, split into chunks."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    docs = [page.extract_text() for page in reader.pages]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents([Document(page_content=t) for t in docs if t.strip()])
    return chunks


def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Remove near-duplicate chunks by normalizing text and hashing.
    Catches repeated headers, footers, and boilerplate.
    """
    seen = set()
    unique = []
    for chunk in chunks:
        normalized = " ".join(chunk.page_content.lower().split())
        h = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


def embed_and_store(chunks: List[Document]) -> QdrantVectorStore:
    """Generate embeddings and store them in Qdrant."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        prefer_grpc=False,
    )
    return vector_store


def build_rag_chain(vector_store: QdrantVectorStore):
    """Build the RAG chain for querying with MMR diversity."""
    # MMR = Max Marginal Relevance: balances relevance vs. diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_template("""
You are a precise and helpful assistant. Answer the question based ONLY on the provided context.
Provide a thorough, detailed, and well-structured answer. Use all relevant information from the context.
Include specific examples, explanations, and details where available.
If the information is not present in the context, respond with: "I couldn't find that information in the document."

Context:
{context}

Question: {question}
""")

    def format_context(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---------- API Endpoints ----------
@app.post("/upload", response_class=JSONResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF file. The file is processed, embeddings are generated,
    and stored in Qdrant. This operation runs asynchronously.
    """
    if not file.content_type.startswith("application/pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    pdf_bytes = await file.read()
    file_hash = compute_file_hash(pdf_bytes)

    # Document-level deduplication
    if file_hash in app.state.seen_hashes:
        app.state.upload_status = {
            "status": "completed",
            "message": "⚡ PDF already processed (skipped duplicate).",
            "chunks_count": app.state.upload_status.get("chunks_count", 0),
            "dedup_count": app.state.upload_status.get("dedup_count", 0),
            "skipped": True,
        }
        return {"status": "skipped", "reason": "duplicate_file", "collection_name": COLLECTION_NAME}

    app.state.seen_hashes.add(file_hash)
    app.state.upload_status = {
        "status": "processing",
        "message": "Processing PDF...",
        "chunks_count": 0,
        "dedup_count": 0,
        "skipped": False,
    }

    def background_process():
        try:
            # 1. Load & chunk
            chunks = load_and_chunk(pdf_bytes)
            raw_count = len(chunks)

            # 2. Chunk-level deduplication
            chunks = deduplicate_chunks(chunks)
            dedup_count = len(chunks)
            removed = raw_count - dedup_count

            app.state.last_chunks = [c.page_content for c in chunks]
            vector_store = embed_and_store(chunks)
            app.state.collection_name = COLLECTION_NAME
            app.state.upload_status = {
                "status": "completed",
                "message": f"✅ PDF processed! {dedup_count} unique chunks stored ({removed} duplicates removed).",
                "chunks_count": dedup_count,
                "dedup_count": removed,
                "skipped": False,
            }
            print(f"✅ PDF processed: {raw_count} raw → {dedup_count} unique ({removed} removed).")
        except Exception as exc:
            app.state.upload_status = {
                "status": "error",
                "message": f"❌ Error: {exc}",
                "chunks_count": 0,
                "dedup_count": 0,
                "skipped": False,
            }
            print(f"\n❌ Error during processing: {exc}")

    background_tasks.add_task(background_process)
    return {"status": "processing_started", "collection_name": COLLECTION_NAME}


@app.get("/status", response_class=JSONResponse)
async def get_status():
    """Poll this endpoint to check if PDF processing is complete."""
    return app.state.upload_status


@app.get("/query", response_class=JSONResponse)
async def query(question: str):
    """
    Query the stored embeddings using GPT‑4o‑mini.
    Returns the answer text.
    """
    if not hasattr(app.state, "collection_name"):
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet.")

    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=app.state.collection_name,
            embedding=embeddings,
            url=QDRANT_URL,
            prefer_grpc=False,
        )
        rag_chain = build_rag_chain(vector_store)
        answer = rag_chain.invoke(question)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/embed", response_class=JSONResponse)
async def get_embedding(payload: Dict[str, Any] = Body(...)):
    """
    Educational endpoint: pass {"text": "your text"} to see the embedding vector.
    """
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field.")
    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        vector = embeddings.embed_query(text)
        return {"text": text, "dimensions": len(vector), "vector": vector}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chunks", response_class=JSONResponse)
async def get_chunks():
    """Return the text chunks from the last uploaded PDF for educational inspection."""
    return {"chunks": app.state.last_chunks, "count": len(app.state.last_chunks)}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html frontend."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------- Run with uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
