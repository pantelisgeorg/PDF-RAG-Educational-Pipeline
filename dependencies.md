# Dependencies

Create a fresh virtual environment:

```bash
python3 -m venv my8env
source my8env/bin/activate   # Linux/Mac
# my8env\Scripts\activate   # Windows
```

Install all required packages:

```bash
pip install \
    fastapi \
    uvicorn \
    python-multipart \
    langchain \
    langchain-core \
    langchain-text-splitters \
    langchain-community \
    langchain-ollama \
    langchain-qdrant \
    langchain-openai \
    pypdf \
    qdrant-client \
    openai
```

> **Note:** `python-multipart` is required by FastAPI to handle file uploads (`UploadFile`, `File(...)`).
> Without it, uploading PDFs will fail with a multipart parsing error.
