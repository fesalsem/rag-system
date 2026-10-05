"""
config.py — Centralized configuration for the RAG system.
All model names, paths, and hyperparameters live here.
Swap models or paths without touching pipeline logic.

Every field that is worth changing between environments reads its default from
an environment variable, so a deployment can be retuned without editing code.
The `RAG_*` names are documented in README.md.
"""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# The embedding model doubles as the default tokenizer, because chunk sizes are
# measured in the tokens of whichever model will embed them. Keep the two in
# step unless you have a reason to tokenize with something else.
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer — falling back to %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number — falling back to %s", name, raw, default)
        return default


def _env_path(name: str, default: str) -> Path:
    return Path(_env_str(name, default))


class EmbeddingConfig(BaseModel):
    model_name: str = Field(
        default_factory=lambda: _env_str("RAG_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        description="HuggingFace sentence-transformer model for local embeddings.",
    )
    device: str = Field(
        default="cpu",
        description="Device for embedding inference: 'cpu' or 'cuda'.",
    )


class LLMConfig(BaseModel):
    model_name: str = Field(
        default_factory=lambda: _env_str("GROQ_MODEL", "openai/gpt-oss-20b"),
        description="Model identifier served by Groq.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", ""),
        description="Groq API key. Never logged — see RAGConfig.redacted().",
    )


class ChunkingConfig(BaseModel):
    """
    Chunk sizes are measured in **tokens**, not characters.

    They used to be characters, which overran the embedding model: MiniLM
    truncates its input at 256 WordPiece tokens, and 1000 characters of
    token-dense text (numbers, tables, code) crosses that line well before the
    character budget runs out. Everything past the cutoff was invisible to
    retrieval while remaining visible to the LLM, so a chunk could be retrieved
    for text it does not actually match.
    """

    tokenizer_name: str = Field(
        default_factory=lambda: _env_str(
            "RAG_TOKENIZER", _env_str("RAG_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
        ),
        description="Tokenizer used to count tokens while splitting.",
    )
    chunk_size: int = Field(
        default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 256), gt=0
    )
    chunk_overlap: int = Field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 32), ge=0
    )
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Priority-ordered separators for RecursiveCharacterTextSplitter.",
    )


class VectorStoreConfig(BaseModel):
    index_path: Path = Field(
        default_factory=lambda: _env_path("RAG_INDEX_PATH", "/tmp/faiss_index"),
        description="Directory where the FAISS index is persisted.",
    )


class MemoryConfig(BaseModel):
    """
    Conversation memory is a fixed window of recent turns.

    It used to be an unbounded ConversationBufferMemory, and the only knob
    offered (`max_token_limit`) was never passed to anything, so history grew
    for the life of the process. Every turn re-sends that history twice: once
    to condense the follow-up question, once to answer it.
    """

    window_turns: int = Field(
        default_factory=lambda: _env_int("RAG_MEMORY_TURNS", 6), gt=0
    )


class RAGConfig(BaseModel):
    """Root configuration object — pass this around instead of globals."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    retriever_k: int = Field(
        default_factory=lambda: _env_int("RAG_RETRIEVER_K", 4),
        gt=0,
        description="Number of chunks returned by the retriever.",
    )
    max_distance: float = Field(
        default_factory=lambda: _env_float("RAG_MAX_DISTANCE", 1.4),
        gt=0.0,
        description=(
            "Largest FAISS squared-L2 distance still treated as relevant. "
            "Embeddings are L2-normalised, so distance = 2 - 2*cos and the "
            "default 1.4 corresponds to cosine similarity 0.3. Retrieval that "
            "returns nothing inside this bound is treated as 'no relevant "
            "passage' and answered from a whole-document overview instead."
        ),
    )
    max_pages: int = Field(
        default_factory=lambda: _env_int("RAG_MAX_PAGES", 300),
        gt=0,
        description="Largest PDF, in pages, that will be accepted.",
    )
    max_upload_mb: int = Field(
        default_factory=lambda: _env_int("RAG_MAX_UPLOAD_MB", 50),
        gt=0,
        description="Largest upload, in megabytes, that will be accepted.",
    )
    log_level: str = Field(
        default_factory=lambda: _env_str("RAG_LOG_LEVEL", "INFO")
    )

    def redacted(self) -> dict:
        """
        Config as a dict, safe to log.

        `model_dump()` includes `llm.groq_api_key`, and this object used to be
        logged whole on every engine construction, which wrote the live key to
        stdout. Anything that wants to log the config must use this instead.
        """
        dumped = self.model_dump()
        if dumped.get("llm", {}).get("groq_api_key"):
            dumped["llm"]["groq_api_key"] = "***redacted***"
        return dumped


# Module-level singleton — import this everywhere.
settings = RAGConfig()
