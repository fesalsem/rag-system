"""
tests/test_rag_engine.py — Unit tests for rag_engine.py
All external calls (embeddings, LLM, FAISS) are mocked so tests
run without API keys, GPU, or network access.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, create_autospec
from langchain_groq import ChatGroq

from rag_engine import RAGEngine
from config import RAGConfig, VectorStoreConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """RAGConfig pointing vector store at a temp directory."""
    cfg = RAGConfig()
    cfg.vector_store = VectorStoreConfig(
        index_path=tmp_path / "faiss_index"
    )
    cfg.llm.groq_api_key = "test-key-not-real"
    return cfg


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"file_name": "faiss.pdf", "page_label": "Page 1", "chunk_index": 0},
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications.",
            metadata={"file_name": "langchain.pdf", "page_label": "Page 2", "chunk_index": 1},
        ),
    ]


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestRAGEngineInit:
    def test_engine_creates_index_dir(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        assert tmp_config.vector_store.index_path.parent.exists()

    def test_index_does_not_exist_initially(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        assert engine.index_exists is False

    def test_query_raises_if_not_initialised(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        with pytest.raises(RuntimeError, match="not ready"):
            engine.query("What is FAISS?")

    def test_load_existing_index_raises_if_missing(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        with pytest.raises(FileNotFoundError):
            engine.load_existing_index()


# ---------------------------------------------------------------------------
# add_documents tests
# ---------------------------------------------------------------------------

class TestAddDocuments:
    def test_raises_on_empty_documents(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        with pytest.raises(ValueError, match="empty"):
            engine.add_documents([])

    def test_add_documents_builds_chain(self, tmp_config, sample_docs):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384] * len(sample_docs)
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_vs = MagicMock()
        mock_vs.index.ntotal = len(sample_docs)
        mock_vs.as_retriever.return_value = MagicMock()

        mock_llm = create_autospec(ChatGroq, instance=True)

        with patch("rag_engine.HuggingFaceEmbeddings", return_value=mock_embeddings), \
             patch("rag_engine.FAISS.from_documents", return_value=mock_vs), \
             patch("rag_engine.ChatGroq", return_value=mock_llm), \
             patch("rag_engine.ConversationalRetrievalChain.from_llm") as mock_chain, \
             patch.object(RAGEngine, "_save_index"):
            mock_chain.return_value = MagicMock()
            engine = RAGEngine(config=tmp_config)
            engine.add_documents(sample_docs)
            assert engine._vector_store is not None
            assert engine._chain is not None


# ---------------------------------------------------------------------------
# get_index_stats tests
# ---------------------------------------------------------------------------

class TestGetIndexStats:
    def test_stats_when_no_index(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        stats = engine.get_index_stats()
        assert stats["indexed"] is False
        assert stats["total_vectors"] == 0

    def test_stats_when_indexed(self, tmp_config, sample_docs):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384] * len(sample_docs)
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_vs = MagicMock()
        mock_vs.index.ntotal = 2
        mock_vs.as_retriever.return_value = MagicMock()

        mock_llm = create_autospec(ChatGroq, instance=True)

        with patch("rag_engine.HuggingFaceEmbeddings", return_value=mock_embeddings), \
             patch("rag_engine.FAISS.from_documents", return_value=mock_vs), \
             patch("rag_engine.ChatGroq", return_value=mock_llm), \
             patch("rag_engine.ConversationalRetrievalChain.from_llm") as mock_chain, \
             patch.object(RAGEngine, "_save_index"):
            mock_chain.return_value = MagicMock()
            engine = RAGEngine(config=tmp_config)
            engine.add_documents(sample_docs)
            stats = engine.get_index_stats()

        assert stats["indexed"] is True
        assert stats["total_vectors"] == 2


# ---------------------------------------------------------------------------
# clear_memory tests
# ---------------------------------------------------------------------------

class TestClearMemory:
    def test_clear_memory_does_not_raise_when_uninitialised(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        engine.clear_memory()   # should be a no-op, not an error

    def test_clear_memory_resets_buffer(self, tmp_config):
        engine = RAGEngine(config=tmp_config)
        memory = engine._get_memory()
        memory.chat_memory.add_user_message("hello")
        assert len(memory.chat_memory.messages) == 1
        engine.clear_memory()
        assert len(memory.chat_memory.messages) == 0
