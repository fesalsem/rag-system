"""Engine wiring for hybrid retrieval: enabled, disabled, and query contexts."""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from config import RAGConfig, VectorStoreConfig
from rag_engine import RAGEngine


class FakeLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="answer")


class FakeStore:
    def __init__(self, scored=None):
        self.scored = list(scored or [])

    def similarity_search_with_score(self, query, k=None):
        return list(self.scored)


class FakeHybrid:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.documents)


def make_doc(content="chunk", page=1):
    return Document(
        page_content=content,
        metadata={"file_name": "doc.pdf", "page_label": f"Page {page}"},
    )


def make_engine(tmp_path, store=None, documents=None):
    """Build an engine with the fake LLM cached, then attach a store."""
    config = RAGConfig()
    config.vector_store = VectorStoreConfig(index_path=tmp_path / "idx")
    config.llm.groq_api_key = "test-key"
    engine = RAGEngine(config=config)
    with patch("rag_engine.ChatGroq", return_value=FakeLLM()):
        # Cached inside the patch; a later query would otherwise build a real
        # client and try to reach Groq.
        engine._models.llm()
    engine._vector_store = store
    engine._docs = list(documents or [])
    return engine


def test_hybrid_is_used_when_enabled(tmp_path, monkeypatch):
    document = make_doc()
    engine = make_engine(tmp_path, store=FakeStore(), documents=[document])
    fake = FakeHybrid([document])
    monkeypatch.setattr("rag_engine.HybridRetriever", lambda *args, **kwargs: fake)
    engine.config.hybrid.enabled = True

    assert engine.retrieve("question") == [document]
    assert fake.queries == ["question"]


def test_dense_path_unchanged_when_hybrid_is_off(tmp_path):
    near = (make_doc("near"), 0.5)
    far = (make_doc("far"), 2.0)
    engine = make_engine(tmp_path, store=FakeStore([near, far]))
    engine.config.hybrid.enabled = False

    assert engine.retrieve("question") == [near[0]]


def test_empty_hybrid_result_falls_back_to_dense(tmp_path, monkeypatch):
    near = (make_doc("near"), 0.5)
    engine = make_engine(tmp_path, store=FakeStore([near]), documents=[make_doc()])
    monkeypatch.setattr(
        "rag_engine.HybridRetriever", lambda *args, **kwargs: FakeHybrid([])
    )
    engine.config.hybrid.enabled = True

    assert engine.retrieve("question") == [near[0]]


def test_query_returns_the_retrieved_contexts(tmp_path):
    document = make_doc("the context")
    engine = make_engine(tmp_path, store=FakeStore([(document, 0.5)]))

    result = engine.query("What is in the document?")

    assert result["contexts"] == ["the context"]
