# Intellect — RAG Document Intelligence System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-F55036?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Local_Vector_DB-0467DF?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A production-ready, modular **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF documents and ask natural language questions. Answers include precise source attribution (page numbers) and full conversation memory.

Built entirely on **free-tier tools** — no cloud vector DB, no paid embeddings.

---

## Architecture

```
rag_system/
├── app.py                  # Streamlit UI (Apple-inspired design)
├── rag_engine.py           # LangChain RAG pipeline
├── document_processor.py   # PDF loading & recursive chunking
├── config.py               # Centralised Pydantic settings
├── requirements.txt
├── .env.template
├── .gitignore
└── README.md
```

### Data flow

```
PDF Upload
    │
    ▼
DocumentProcessor
(PyPDFLoader → RecursiveCharacterTextSplitter)
    │  chunks with metadata
    ▼
RAGEngine.add_documents()
(HuggingFace all-MiniLM-L6-v2 embeddings)
    │  vectors
    ▼
FAISS Index (persisted to disk)
    │
    │  user question
    ▼
ConversationalRetrievalChain
(top-k retrieval → Groq Llama 3.1 8B)
    │
    ▼
Answer + Source Attribution
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain 0.3 | Modular chains, memory, retrieval |
| LLM | Llama 3.1 8B via Groq | Free tier, ~500 tok/s |
| Embeddings | all-MiniLM-L6-v2 | Local, free, fast, 384-dim |
| Vector DB | FAISS | Local persistence, no cloud cost |
| Frontend | Streamlit | Rapid Python UI |
| Config | Pydantic v2 | Type-safe settings |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
```

### 2. Python environment (requires Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. API key

```bash
cp .env.template .env
# Edit .env → GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run

```bash
streamlit run app.py
```

---

## Swapping Components

All configuration lives in `config.py`. No pipeline code changes needed.

| To swap | Change in `config.py` |
|---|---|
| Groq → Ollama | `llm.provider = "ollama"`, update `model_name` |
| Llama 3.1 8B → 70B | `llm.model_name = "llama-3.3-70b-versatile"` |
| FAISS → Pinecone | `vector_store.provider = "pinecone"` |
| MiniLM → BGE | `embedding.model_name = "BAAI/bge-small-en-v1.5"` |

---

## License

MIT
