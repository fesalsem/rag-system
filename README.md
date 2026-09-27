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

Every answer comes with a **Sources** section pointing to the exact page in your document, so you can verify where each fact came from.

---

## What it gives you

- **Ask in plain English** — no special syntax, just talk to your documents
- **Source attribution** — every answer cites the page it came from
- **Conversation memory** — ask follow-up questions and it remembers context
- **Multiple PDFs** — index several documents and search across all of them at once

---

## How it works (in plain terms)

When you upload a PDF, the app:

1. Splits it into small chunks of text
2. Converts each chunk into a "vector" (a mathematical fingerprint of its meaning)
3. Stores them in a local search index
4. When you ask a question, it finds the most relevant chunks and asks an LLM to answer **using only that material** — so answers stay grounded in your documents, with no guesswork from the model's memory

**Tech Stack:** LangChain · Groq (GPT-OSS-20B) · all-MiniLM-L6-v2 embeddings · FAISS · Streamlit

> The model is set in `config.py` and can be changed without touching pipeline code. The deployed app originally ran Llama 3.1 and moved to GPT-OSS-20B when Llama was retired.

---

## 🛠️ For developers

Want to run or modify it locally?

### 1. Clone & set up (requires Python 3.12)

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

No Python setup needed, and it pins the exact interpreter version.

```bash
cp .env.template .env        # then put your GROQ_API_KEY in it
docker compose up --build
```

Open http://localhost:8501. Stop with `Ctrl+C`; `docker compose down` removes the containers but keeps the volumes.

Two named volumes persist across restarts:

- `faiss_index` is mounted at `/tmp/faiss_index`, because `config.py` defaults `index_path` there and a container's `/tmp` is wiped on every restart. Your uploaded documents stay indexed.
- `hf_cache` holds the downloaded MiniLM model, so a rebuild does not fetch it again.

The image installs PyTorch from the CPU-only index. The default wheel pulls several gigabytes of CUDA libraries this app never uses, since embeddings run on CPU.

To run the test suite inside the container, build with the dev flag first. `pytest` is not in `requirements.txt`, so a plain build has no test tooling:

```bash
docker build --build-arg INSTALL_DEV=true -t intellect:dev .
docker run --rm intellect:dev pytest -q
```

CI does exactly this, then starts the container and polls its health endpoint.

### Project structure

```
rag_system/
├── app.py                  # Streamlit UI
├── rag_engine.py           # LangChain RAG pipeline
├── document_processor.py   # PDF loading & chunking
├── config.py               # Centralised Pydantic settings
├── requirements.txt
└── .env.template
```

### Swapping components

All configuration lives in `config.py` — no pipeline code changes needed.

| To swap | Change in `config.py` |
|---|---|
| Groq → Ollama | `llm.provider = "ollama"`, update `model_name` |
| Change Groq model | `llm.model_name = "openai/gpt-oss-20b"` (or set `GROQ_MODEL` in `.env`/secrets) |
| FAISS → Pinecone | `vector_store.provider = "pinecone"` |
| MiniLM → BGE | `embedding.model_name = "BAAI/bge-small-en-v1.5"` |

---

## License

MIT
