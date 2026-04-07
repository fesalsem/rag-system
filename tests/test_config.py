"""
tests/test_config.py — Unit tests for config.py
Tests that all settings load correctly with proper defaults and types.
"""

import pytest
from pathlib import Path
from config import (
    RAGConfig,
    EmbeddingConfig,
    LLMConfig,
    ChunkingConfig,
    VectorStoreConfig,
    MemoryConfig,
    settings,
)


class TestEmbeddingConfig:
    def test_default_model_name(self):
        cfg = EmbeddingConfig()
        assert cfg.model_name == "all-MiniLM-L6-v2"

    def test_default_device(self):
        cfg = EmbeddingConfig()
        assert cfg.device == "cpu"

    def test_custom_model(self):
        cfg = EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5")
        assert cfg.model_name == "BAAI/bge-small-en-v1.5"


class TestLLMConfig:
    def test_default_model(self):
        cfg = LLMConfig()
        assert cfg.model_name == "llama-3.1-8b-instant"

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            LLMConfig(temperature=3.0)
        with pytest.raises(Exception):
            LLMConfig(temperature=-0.1)

    def test_valid_temperature(self):
        cfg = LLMConfig(temperature=0.7)
        assert cfg.temperature == 0.7

    def test_max_tokens_positive(self):
        with pytest.raises(Exception):
            LLMConfig(max_tokens=0)

    def test_provider_default(self):
        cfg = LLMConfig()
        assert cfg.provider == "groq"


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_separators_is_list(self):
        cfg = ChunkingConfig()
        assert isinstance(cfg.separators, list)
        assert len(cfg.separators) > 0

    def test_custom_chunk_size(self):
        cfg = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 50

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=0)


class TestVectorStoreConfig:
    def test_default_provider(self):
        cfg = VectorStoreConfig()
        assert cfg.provider == "faiss"

    def test_index_path_is_path(self):
        cfg = VectorStoreConfig()
        assert isinstance(cfg.index_path, Path)

    def test_custom_index_path(self):
        cfg = VectorStoreConfig(index_path=Path("/tmp/test_index"))
        assert cfg.index_path == Path("/tmp/test_index")


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.memory_key == "chat_history"
        assert cfg.return_messages is True
        assert cfg.max_token_limit == 4000


class TestRAGConfig:
    def test_root_config_composes_subconfigs(self):
        cfg = RAGConfig()
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.chunking, ChunkingConfig)
        assert isinstance(cfg.vector_store, VectorStoreConfig)
        assert isinstance(cfg.memory, MemoryConfig)

    def test_retriever_k_default(self):
        cfg = RAGConfig()
        assert cfg.retriever_k == 4

    def test_settings_singleton_is_rag_config(self):
        assert isinstance(settings, RAGConfig)

    def test_config_serialisable(self):
        cfg = RAGConfig()
        dumped = cfg.model_dump()
        assert "embedding" in dumped
        assert "llm" in dumped
        assert "chunking" in dumped
