"""
tests/test_config.py - Behaviour tests for the centralized configuration.

config is the contract every other module reads, so these tests pin the parts
that were wrong before: fields that were removed because nothing used them (a
`provider` knob on both the LLM and the vector store, and `max_token_limit`,
which was never passed to anything), and the guarantee that an environment
variable can only replace a default, never break startup with a traceback.

The `redacted()` test is written to document exactly what it protects against:
it asserts that the raw `model_dump()` still carries the API key, so the
masking in `redacted()` is shown to be load-bearing rather than decorative.
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
)

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Every variable config.py reads. Cleared before each test so a developer's
# shell or the repository .env cannot make a "default" assertion pass or fail
# for the wrong reason.
_ENV_VARS = [
    "RAG_EMBEDDING_MODEL",
    "RAG_TOKENIZER",
    "GROQ_MODEL",
    "GROQ_API_KEY",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_MEMORY_TURNS",
    "RAG_RETRIEVER_K",
    "RAG_MAX_DISTANCE",
    "RAG_MAX_PAGES",
    "RAG_MAX_UPLOAD_MB",
    "RAG_LOG_LEVEL",
    "RAG_INDEX_PATH",
    "RAG_PERSIST_INDEX",
]


@pytest.fixture
def clean_env(monkeypatch):
    """The environment with every config variable removed."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class TestEmbeddingConfig:
    def test_defaults(self, clean_env):
        cfg = EmbeddingConfig()
        assert cfg.model_name == _DEFAULT_EMBEDDING_MODEL
        assert cfg.device == "cpu"

    def test_env_overrides_model_name(self, clean_env):
        clean_env.setenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        assert EmbeddingConfig().model_name == "BAAI/bge-small-en-v1.5"

    def test_explicit_value_beats_env(self, clean_env):
        clean_env.setenv("RAG_EMBEDDING_MODEL", "from-env/model")
        cfg = EmbeddingConfig(model_name="passed/in")
        assert cfg.model_name == "passed/in"


class TestLLMConfig:
    def test_defaults(self, clean_env):
        cfg = LLMConfig()
        assert cfg.model_name == "openai/gpt-oss-20b"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 1024
        # No key in the environment means an empty string, so the registry can
        # raise a useful error instead of building a client with None.
        assert cfg.groq_api_key == ""

    def test_env_overrides_model_name(self, clean_env):
        clean_env.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        assert LLMConfig().model_name == "llama-3.3-70b-versatile"

    def test_env_overrides_api_key(self, clean_env):
        clean_env.setenv("GROQ_API_KEY", "gsk_from_env")
        assert LLMConfig().groq_api_key == "gsk_from_env"

    def test_removed_provider_field_is_gone(self, clean_env):
        # `provider` was a switch that selected between... nothing. It never
        # changed a code path, so it was removed. It must not reappear as an
        # accepted field or the config silently accepts a no-op setting.
        assert "provider" not in LLMConfig.model_fields
        assert not hasattr(LLMConfig(), "provider")

    def test_temperature_upper_bound_is_enforced(self):
        with pytest.raises(Exception):
            LLMConfig(temperature=3.0)

    def test_temperature_lower_bound_is_enforced(self):
        with pytest.raises(Exception):
            LLMConfig(temperature=-0.1)

    def test_valid_temperature_accepted(self):
        assert LLMConfig(temperature=0.7).temperature == 0.7

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(Exception):
            LLMConfig(max_tokens=0)


class TestChunkingConfig:
    def test_defaults(self, clean_env):
        # 256 tokens, not 1000 characters: MiniLM truncates at 256 WordPiece
        # tokens, so a character budget could hold text the embedding never saw.
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 256
        assert cfg.chunk_overlap == 32
        assert isinstance(cfg.separators, list)
        assert cfg.separators

    def test_env_overrides_chunk_size_and_overlap(self, clean_env):
        clean_env.setenv("RAG_CHUNK_SIZE", "512")
        clean_env.setenv("RAG_CHUNK_OVERLAP", "64")
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64

    def test_bad_env_falls_back_to_default(self, clean_env):
        clean_env.setenv("RAG_CHUNK_SIZE", "not-a-number")
        # A typo in a deployment variable must not take the app down at import.
        assert ChunkingConfig().chunk_size == 256

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=0)

    def test_chunk_overlap_must_not_be_negative(self):
        with pytest.raises(Exception):
            ChunkingConfig(chunk_overlap=-1)

    def test_tokenizer_defaults_to_embedding_model(self, clean_env):
        # Chunk sizes are measured in the embedding model's tokens, so the two
        # names default to each other unless RAG_TOKENIZER says otherwise.
        clean_env.setenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        assert ChunkingConfig().tokenizer_name == "BAAI/bge-small-en-v1.5"

    def test_tokenizer_defaults_to_builtin_model(self, clean_env):
        assert ChunkingConfig().tokenizer_name == _DEFAULT_EMBEDDING_MODEL

    def test_tokenizer_env_overrides_both(self, clean_env):
        clean_env.setenv("RAG_EMBEDDING_MODEL", "embed/model")
        clean_env.setenv("RAG_TOKENIZER", "tokenizer/model")
        assert ChunkingConfig().tokenizer_name == "tokenizer/model"


class TestVectorStoreConfig:
    def test_defaults(self, clean_env):
        cfg = VectorStoreConfig()
        assert isinstance(cfg.index_path, Path)
        assert cfg.index_path == Path("/tmp/faiss_index")

    def test_env_overrides_index_path(self, clean_env, tmp_path):
        target = tmp_path / "custom_index"
        clean_env.setenv("RAG_INDEX_PATH", str(target))
        assert VectorStoreConfig().index_path == target

    def test_persist_defaults_on(self, clean_env):
        # Default is on, so a single-user deployment keeps its index across
        # restarts. The web app overrides this to off for its session engines.
        assert VectorStoreConfig().persist is True

    @pytest.mark.parametrize("raw, expected", [("1", True), ("0", False)])
    def test_persist_env_override(self, clean_env, raw, expected):
        clean_env.setenv("RAG_PERSIST_INDEX", raw)
        assert VectorStoreConfig().persist is expected

    def test_removed_provider_field_is_gone(self, clean_env):
        assert "provider" not in VectorStoreConfig.model_fields
        assert not hasattr(VectorStoreConfig(), "provider")


class TestMemoryConfig:
    def test_defaults(self, clean_env):
        assert MemoryConfig().window_turns == 6

    def test_env_overrides_window(self, clean_env):
        clean_env.setenv("RAG_MEMORY_TURNS", "12")
        assert MemoryConfig().window_turns == 12

    def test_bad_env_falls_back_to_default(self, clean_env):
        clean_env.setenv("RAG_MEMORY_TURNS", "lots")
        assert MemoryConfig().window_turns == 6

    def test_window_must_be_positive(self):
        with pytest.raises(Exception):
            MemoryConfig(window_turns=0)

    def test_removed_max_token_limit_is_gone(self, clean_env):
        # The old unbounded ConversationBufferMemory exposed `max_token_limit`,
        # which was documented but never wired into anything. History is now a
        # bounded deque of turns instead.
        assert "max_token_limit" not in MemoryConfig.model_fields
        assert "max_token_limit" not in RAGConfig.model_fields
        assert set(MemoryConfig().model_dump()) == {"window_turns"}


class TestRAGConfig:
    def test_composes_subconfigs(self, clean_env):
        cfg = RAGConfig()
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.chunking, ChunkingConfig)
        assert isinstance(cfg.vector_store, VectorStoreConfig)
        assert isinstance(cfg.memory, MemoryConfig)

    def test_defaults(self, clean_env):
        cfg = RAGConfig()
        assert cfg.retriever_k == 4
        assert cfg.max_distance == 1.4
        assert cfg.max_pages == 300
        assert cfg.max_upload_mb == 50
        assert cfg.log_level == "INFO"

    @pytest.mark.parametrize(
        "var, attr, value, expected",
        [
            ("RAG_RETRIEVER_K", "retriever_k", "7", 7),
            ("RAG_MAX_DISTANCE", "max_distance", "0.9", 0.9),
            ("RAG_MAX_PAGES", "max_pages", "10", 10),
            ("RAG_MAX_UPLOAD_MB", "max_upload_mb", "5", 5),
            ("RAG_LOG_LEVEL", "log_level", "DEBUG", "DEBUG"),
        ],
    )
    def test_env_overrides(self, clean_env, var, attr, value, expected):
        clean_env.setenv(var, value)
        assert getattr(RAGConfig(), attr) == expected

    @pytest.mark.parametrize(
        "var, attr, default",
        [
            ("RAG_RETRIEVER_K", "retriever_k", 4),
            ("RAG_MAX_DISTANCE", "max_distance", 1.4),
            ("RAG_MAX_PAGES", "max_pages", 300),
            ("RAG_MAX_UPLOAD_MB", "max_upload_mb", 50),
        ],
    )
    def test_bad_env_falls_back_instead_of_raising(
        self, clean_env, var, attr, default
    ):
        clean_env.setenv(var, "abc")
        # "abc" is a plausible typo. The fallback keeps import from failing.
        assert getattr(RAGConfig(), attr) == default

    def test_validators_reject_nonsense(self):
        with pytest.raises(Exception):
            RAGConfig(retriever_k=0)
        with pytest.raises(Exception):
            RAGConfig(max_distance=0)
        with pytest.raises(Exception):
            RAGConfig(max_pages=0)
        with pytest.raises(Exception):
            RAGConfig(max_upload_mb=0)

    def test_redacted_masks_only_the_api_key(self):
        cfg = RAGConfig()
        secret = "gsk_live_do_not_log_this_42"
        cfg.llm.groq_api_key = secret

        # The raw dump really does contain the key. This is what redacted()
        # exists to prevent from being logged, and asserting it here means the
        # protection cannot be removed without this test noticing.
        assert cfg.model_dump()["llm"]["groq_api_key"] == secret

        redacted = cfg.redacted()
        assert redacted["llm"]["groq_api_key"] == "***redacted***"
        # Everything else is carried through untouched.
        assert redacted["llm"]["model_name"] == cfg.llm.model_name
        assert redacted["llm"]["temperature"] == cfg.llm.temperature
        assert redacted["embedding"] == cfg.model_dump()["embedding"]
        assert redacted["chunking"] == cfg.model_dump()["chunking"]
        assert redacted["vector_store"] == cfg.model_dump()["vector_store"]
        assert redacted["memory"] == cfg.model_dump()["memory"]
        # And redacted() does not mutate the original.
        assert cfg.model_dump()["llm"]["groq_api_key"] == secret

    def test_redacted_leaves_empty_key_alone(self, clean_env):
        cfg = RAGConfig()
        assert cfg.llm.groq_api_key == ""
        assert cfg.redacted()["llm"]["groq_api_key"] == ""

    def test_config_is_serialisable(self, clean_env):
        dumped = RAGConfig().model_dump()
        assert set(dumped) == {
            "embedding",
            "llm",
            "chunking",
            "vector_store",
            "memory",
            "retriever_k",
            "max_distance",
            "max_pages",
            "max_upload_mb",
            "log_level",
        }
