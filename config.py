"""
config.py — Centralized configuration for the RAG system.
All model names, paths, and hyperparameters live here.
Swap models or paths without touching pipeline logic.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig(BaseModel):
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for local embeddings.",
    )
    device: str = Field(
        default="cpu",
        description="Device for embedding inference: 'cpu' or 'cuda'.",
    )


class LLMConfig(BaseModel):
    provider: str = Field(
        default="groq",
        description="LLM provider identifier. Swap to 'ollama' or 'openai' as needed.",
    )
    model_name: str = Field(
        default="llama-3.1-8b-instant",
        description="Model identifier on the chosen provider.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", ""),
        description="Groq API key loaded from .env.",
    )


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Priority-ordered separators for RecursiveCharacterTextSplitter.",
    )


class VectorStoreConfig(BaseModel):
    provider: str = Field(
        default="faiss",
        description="Vector store provider. Swap to 'pinecone' or 'chroma'.",
    )
    index_path: Path = Field(
        default=Path("./vector_store/faiss_index"),
        description="Local path where the FAISS index is persisted.",
    )


class MemoryConfig(BaseModel):
    memory_key: str = Field(default="chat_history")
    return_messages: bool = Field(default=True)
    max_token_limit: int = Field(default=4000)


class RAGConfig(BaseModel):
    """Root configuration object — pass this around instead of globals."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retriever_k: int = Field(
        default=4, description="Number of chunks returned by the retriever."
    )
    log_level: str = Field(default="INFO")


# Module-level singleton — import this everywhere.
settings = RAGConfig()
