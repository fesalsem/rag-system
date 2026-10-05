# Intellect — RAG Document Intelligence System

> Upload your PDFs, then ask questions in plain language. Get precise answers backed by exact page-number citations.

## 🚀 Try it now

**https://fesalsem-rag.streamlit.app**

No install, no setup — just open the link and start asking.

---

## How to use

1. **Open** the app: https://fesalsem-rag.streamlit.app
2. **Upload** one or more PDFs in the left sidebar
3. **Click "Index Documents"** (this prepares your files for searching)
4. **Type a question** in the chat box and press Enter

Every answer is shown with **source chips** naming the file and page it drew on. Those chips are built by the UI from the retrieval metadata itself, and the model is given the same file-and-page labels in its context, so any page it mentions is grounded in a chunk that was actually retrieved. The model does not write its own "Sources:" section.

---

## What it gives you

- **Ask in plain English:** no special syntax, just talk to your documents
- **Source attribution:** every answer names the file and page it drew on
- **Conversation memory:** follow-up questions keep their context within your session, up to the configured number of turns
- **Multiple PDFs:** index several documents and search across all of them at once

---

## How it works (in plain terms)

When you upload a PDF, the app:

1. Splits it into small chunks of text
2. Converts each chunk into a "vector" (a mathematical fingerprint of its meaning)
3. Stores them in a local search index
4. When you ask a question, it finds the most relevant chunks and asks an LLM to answer **using only that material** — so answers stay grounded in your documents, with no guesswork from the model's memory

**Tech Stack:** LangChain · Groq (GPT-OSS-20B) · all-MiniLM-L6-v2 embeddings · FAISS · Streamlit

> The Groq model is chosen with the `GROQ_MODEL` environment variable (default `openai/gpt-oss-20b`), so it can be changed without touching pipeline code. The deployed app originally ran Llama 3.1 and moved to GPT-OSS-20B when Llama was retired.

---

## 🛠️ For developers

Want to run or modify it locally?

### 1. Clone & set up (Python 3.12 is the supported target; CI also tests 3.11)

```bash
git clone https://github.com/fesalsem/rag-system.git
cd rag-system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.template .env
# Edit .env → GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
streamlit run app.py
```

### 4. Or run with Docker

No Python setup needed, and it pins the exact interpreter version (`3.12.11`).

```bash
cp .env.template .env        # then put your GROQ_API_KEY in it
docker compose up --build
```

Open http://localhost:8501. Stop with `Ctrl+C`; `docker compose down` removes the container and its network but keeps the `hf_cache` volume:

- `hf_cache` holds the downloaded MiniLM embedding model, so a restart or rebuild does not fetch it again.

There is deliberately **no** volume for the FAISS index. Indexing is per session: each browser session gets its own engine, its own index and its own history, stored under `/tmp/faiss_index/<session-id>`. Documents you upload are visible only to your session, do not survive a restart, and do not cross to another user. That is intended behaviour for a shared deployment, not a bug.

The image installs PyTorch from the CPU-only index. The default wheel pulls several gigabytes of CUDA libraries this app never uses, since embeddings run on CPU.

To run the test suite inside the container, build with the dev flag first. `pytest` is not in `requirements.txt`, so a plain build has no test tooling:

```bash
docker build --build-arg INSTALL_DEV=true -t intellect:dev .
docker run --rm intellect:dev pytest -q
```

CI does exactly this, then starts the container and polls its health endpoint.

### Project structure

```
rag-system/
├── app.py                       # Streamlit UI and per-session wiring
├── rag_engine.py                # RAG pipeline: embeddings, FAISS, LLM chain
├── document_processor.py        # PDF loading and token-based chunking
├── ui_helpers.py                # Escaping, citation labels and safe markdown
├── config.py                    # Centralised settings read from the environment
├── tests/                       # Pytest suite (fully mocked, no network)
├── requirements.txt             # Pinned runtime dependencies
├── pyproject.toml               # pytest and coverage configuration
├── Dockerfile                   # Pinned python:3.12.11-slim-bookworm image
├── docker-compose.yml           # Local run with the Groq key injected
├── runtime.txt                  # Documentation only; see the note below
├── .env.template                # Copy to .env and add your Groq key
└── .github/workflows/ci.yml     # CI: test matrix plus Docker build and smoke test
```

`runtime.txt` is not read by Streamlit Community Cloud. The deployed Python
version is chosen in the Advanced settings step of the deploy dialog and is
fixed once the app exists, so changing it means deleting and redeploying the
app. The file is kept only as a note of the supported target.

### Configuration

Every setting is optional except the API key, and each has a working default.
Set them in `.env` for local and Docker runs, or in Streamlit secrets on Cloud.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq chat model |
| `RAG_INDEX_PATH` | `/tmp/faiss_index` | Parent directory for the per-session FAISS indexes |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RAG_TOKENIZER` | same as `RAG_EMBEDDING_MODEL` | Tokenizer used for token-based chunking |
| `RAG_PERSIST_INDEX` | `1` | Write the FAISS index to disk. The Streamlit app forces it off: its indexes are per-session, so persisting them would leave one directory per visit and never read any of them back |
| `RAG_CHUNK_SIZE` | `256` | Chunk size, in tokens |
| `RAG_CHUNK_OVERLAP` | `32` | Chunk overlap, in tokens |
| `RAG_RETRIEVER_K` | `4` | Chunks retrieved per query |
| `RAG_MAX_DISTANCE` | `1.4` | Largest FAISS squared-L2 distance accepted (normalised embeddings, so 1.4 is roughly cosine 0.3). Chunks beyond it are discarded, and if none pass the app answers from a whole-document overview instead |
| `RAG_MEMORY_TURNS` | `6` | Conversation turns kept per session |
| `RAG_MAX_PAGES` | `300` | Per-PDF page cap |
| `RAG_MAX_UPLOAD_MB` | `50` | Per-file upload size cap, in MB |
| `RAG_LOG_LEVEL` | `INFO` | Logging verbosity |

### Swapping components

These changes work through configuration alone:

| To change | Set |
|---|---|
| Groq model | `GROQ_MODEL` |
| Embedding model | `RAG_EMBEDDING_MODEL` |
| Chunking | `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP` (both in tokens) |

Anything larger is a code change, not a setting. Swapping the LLM provider
(Groq to Ollama or OpenAI) or the vector store (FAISS to Pinecone or Chroma)
means editing `rag_engine.py`: no code path reads a provider field, and none of
those alternatives are dependencies.

---

## License

MIT
