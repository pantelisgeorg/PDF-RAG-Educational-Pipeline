# 📚 PDF RAG Educational Pipeline

A hands-on, educational FastAPI application that demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline. Upload a PDF, inspect how it is chunked and embedded, visualize embedding vectors, and query the knowledge base using GPT‑4o‑mini.

This project is designed for **learning** — every step of the pipeline is visible through the web UI and the API.

---

## 🎯 What You Will Learn

By running and exploring this app, you will understand:

1. **Document ingestion** — how a PDF is read and split into text chunks.
2. **Text chunking** — how `RecursiveCharacterTextSplitter` breaks documents into overlapping pieces.
3. **Deduplication** — how repeated headers, footers, and boilerplate are removed at both the **document level** (hash check) and **chunk level** (normalized text hash).
4. **Embeddings** — how an LLM (via Ollama) converts text into high-dimensional vectors.
5. **Vector storage** — how Qdrant stores and indexes these vectors for fast similarity search.
6. **Retrieval** — how a user query is embedded and matched against stored vectors.
7. **MMR (Max Marginal Relevance)** — how retrieved results are diversified so the LLM gets broad, non-redundant context.
8. **Generation** — how GPT‑4o‑mini answers the question using only the retrieved context.

---

## 🏗️ Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────┐
│ Upload PDF  │────▶│ Extract Text │────▶│ Split Chunks│────▶│ Deduplicate     │────▶│ Embed   │
│  (Browser)  │     │  (pypdf)     │     │(Recursive   │     │ (hash-based)    │     │(Ollama) │
└─────────────┘     └──────────────┘     │  Splitter)  │     └─────────────────┘     └────┬────┘
                                          └─────────────┘                                  │
                                                                                           ▼
                                                                                    ┌─────────────┐
                                                                                    │ Store in    │
                                                                                    │ Qdrant      │
                                                                                    └──────┬──────┘
                                                                                           │
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐     ┌──────▼──────┐
│ Ask Question│────▶│ Embed Query  │────▶│ Retrieve Top-K │────▶│ MMR Diversify│────▶│ Build Prompt│
│  (Browser)  │     │  (Ollama)    │     │   (Qdrant)     │     │             │     │ + Context   │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────┘     └──────┬──────┘
                                                                                           │
                                                                                           ▼
                                                                                    ┌─────────────┐
                                                                                    │  GPT-4o-mini│
                                                                                    │   Answer    │
                                                                                    └─────────────┘
```

---

## 📦 Prerequisites

Before running the app, make sure the following are installed and running on your machine.

### 1. Python 3.10+

```bash
python3 --version
```

### 2. Ollama (for local embeddings)

Download and install from [https://ollama.com](https://ollama.com), then pull the embedding model:

```bash
ollama pull nomic-embed-text-v2-moe:latest
```

Verify it works:

```bash
ollama list
# You should see nomic-embed-text-v2-moe:latest
```

### 3. Qdrant (vector database)

The easiest way is Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or download the binary from [https://qdrant.tech](https://qdrant.tech).

Verify it is running:

```bash
curl http://localhost:6333
# Should return {"title":"qdrant","version":"..."}
```

### 4. OpenAI API Key

You need an OpenAI API key for GPT‑4o‑mini. Get one at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

Set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

(You can also set it in a `.env` file if you use `python-dotenv`.)

---

## 🚀 Installation

1. **Clone or navigate to the project folder:**

```bash
cd /path/to/dir
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv myenv
source myenv/bin/activate   # Linux/Mac
# myenv\Scripts\activate   # Windows
```

3. **Install dependencies:**

```bash
pip install \
    fastapi \
    uvicorn \
    python-multipart \
    pypdf \
    langchain \
    langchain-core \
    langchain-text-splitters \
    langchain-ollama \
    langchain-qdrant \
    langchain-openai \
    qdrant-client \
    openai
```

> ⚠️ **`python-multipart`** is essential — FastAPI uses it to parse uploaded PDF files (`UploadFile`). If it's missing, file uploads will fail.

---

## ▶️ Running the App

```bash
uvicorn app:app --reload
```

Then open your browser at:

```
http://127.0.0.1:8000
```

The UI includes:
- An **interactive flowchart** of the full pipeline
- **Upload & Process** with live status polling
- **Dedup stats** showing unique chunks vs. duplicates removed
- **Extracted Chunks** viewer with one-click copy
- **Embedding Visualizer** to see the vector for any text input
- **Query Interface** to ask questions and get GPT‑4o‑mini answers

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the HTML frontend |
| `/upload` | POST | Upload a PDF file for processing |
| `/status` | GET | Poll for upload completion status |
| `/query` | GET | Ask a question (requires uploaded PDF) |
| `/chunks` | GET | List the deduplicated text chunks |
| `/embed` | POST | Generate an embedding vector for any text |

---

## 🗄️ Database / Collection Management

The app stores all embedding vectors in **Qdrant** inside a **collection**.

By default, the collection name is:

```python
COLLECTION_NAME = "pdf_knowledge_base"
```

> ⚠️ **Important:** If you upload a PDF, quit the server, and restart it later, the **old vectors are still in Qdrant**. Your new queries will search the old data mixed with any new uploads. Below are the recommended ways to manage this.

---

### Option 1: Change the Collection Name (Recommended for Quick Testing)

The simplest way to start fresh is to use a **different collection name** each time you want a clean slate.

You can do this via an **environment variable** without touching any code:

```bash
export COLLECTION_NAME="my_new_run"
uvicorn app:app --reload
```

Or edit `app.py` directly:

```python
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_knowledge_base_v2")
```

**Pros:** Zero risk of losing old data; great for comparing experiments.  
**Cons:** Old collections remain on disk and consume storage until you clean them up.

---

### Option 2: Delete the Collection Before Restarting

If you want to **wipe the database and start completely fresh**, delete the collection before running the app.

#### Via `curl` (Qdrant REST API)

```bash
curl -X DELETE http://localhost:6333/collections/pdf_knowledge_base
```

#### Via Python (one-liner)

```bash
python3 -c "from qdrant_client import QdrantClient; QdrantClient(url='http://localhost:6333').delete_collection('pdf_knowledge_base')"
```

#### Via the Qdrant Dashboard (if you have it running)

Visit `http://localhost:6333/dashboard`, select the collection, and delete it.

**Pros:** Truly clean state; no leftover data.  
**Cons:** You lose all previously uploaded PDFs.

---

### Option 3: Auto-Delete on Startup (Convenience Snippet)

If you are actively developing or teaching and **always want a fresh start** on every server restart, add this snippet to the bottom of `app.py`, right before `if __name__ == "__main__":`:

```python
# ---------- Optional: Auto-delete collection on startup ----------
def _reset_collection():
    """Delete the collection on startup so every run starts fresh."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted old collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"⚠️  Could not reset collection: {e}")

_reset_collection()
```

> 🔴 **Warning:** Only use this during development/demoing. It will permanently erase uploaded data every time the server restarts.

---

### Option 4: Per-Session Collections (Advanced)

For a multi-user classroom scenario, you can generate a unique collection name per browser session or per user. This requires a small backend change (e.g., generate a UUID on first visit and store it in the session state), but it allows every student to have their own isolated vector database without interfering with each other.

---

### Summary: Which Should I Use?

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick test, want clean data | **Option 1:** change `COLLECTION_NAME` env var |
| Finished a demo, cleaning up | **Option 2:** `curl -X DELETE` the collection |
| Active development / teaching | **Option 3:** add auto-delete snippet to `app.py` |
| Classroom with many students | **Option 4:** generate per-session collection names |

---

## 🧪 Educational Features

### 1. Live Flowchart
The homepage shows a **two-phase flowchart** with color-coded stages:
- 🟠 **Orange** = Deduplication steps
- 🟢 **Green** = MMR diversity step
- 🔵 **Blue** = Standard pipeline steps

### 2. Dedup Stats
After uploading, you will see:
- **Unique Chunks** — how many chunks were kept after deduplication
- **Duplicates Removed** — how many redundant chunks (headers, footers, repeats) were filtered out

### 3. Chunk Inspector
Every chunk is displayed in a card with a **Copy** button. This lets students read exactly what context the LLM will receive.

### 4. Embedding Visualizer
Type any text and see:
- A **bar chart** of the first 128 dimensions
- The **full vector** as JSON
- Min/max values

This makes the abstract concept of "high-dimensional vector" concrete and visual.

### 5. MMR in Action
The query endpoint uses `search_type="mmr"` with `fetch_k=20, k=5`. This means:
- Qdrant finds the 20 most similar chunks
- The retriever then picks the **5 most diverse** among them
- Students get broader context instead of 5 copies of the same paragraph

---

## 🛠️ Customization

### Change the embedding model

```python
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")
```

Make sure the model is pulled in Ollama first:

```bash
ollama pull <model-name>
```

### Change chunk size / overlap

In `app.py`, modify the splitter:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # smaller = more chunks, finer granularity
    chunk_overlap=50,    # overlap avoids losing context at boundaries
    ...
)
```

### Change the LLM

```python
llm = ChatOpenAI(
    model="gpt-4o",      # or "gpt-3.5-turbo", etc.
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)
```

### Adjust MMR diversity

```python
search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
```

- `lambda_mult` closer to **1.0** = more relevance, less diversity
- `lambda_mult` closer to **0.0** = more diversity, less relevance

---

## 🐛 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `SyntaxError: parameter without a default` | FastAPI parameter order | Already fixed; parameters are ordered correctly |
| `'bytes' object has no attribute 'seek'` | `PdfReader` needs a file-like object | Already fixed; `io.BytesIO()` is used |
| `QdrantVectorStore.from_existing_collection() got multiple values` | Wrong positional argument order | Already fixed; uses keyword args |
| `Internal Server Error` on query | Querying before PDF is ready | Wait for the green "completed" status badge |
| Can't connect to Qdrant | Qdrant not running | Start Qdrant Docker container |
| Can't connect to Ollama | Ollama not running | Start Ollama app or `ollama serve` |
| `OPENAI_API_KEY is missing` | Env var not set | `export OPENAI_API_KEY="sk-..."` |
| Multipart/form-data error on upload | `python-multipart` not installed | `pip install python-multipart` |

---

## 📁 Project Structure

```
gptoss_rag/
├── app.py              # FastAPI backend
├── index.html          # Educational frontend
├── my8env/             # Python virtual environment
├── README.md           # This file
└── qdrant_storage/     # Qdrant data (created by Docker)
```

---

## 📄 License

This project is provided for **educational purposes**. Feel free to fork, modify, and use it in classrooms, workshops, or personal learning.

---

## 🙋 Questions?

If something doesn't work, check:
1. Is Ollama running and the model pulled?
2. Is Qdrant running on `localhost:6333`?
3. Is `OPENAI_API_KEY` exported in your terminal?
4. Did you change the collection name or delete the old one if starting fresh?

Happy learning! 🎓
