"""
tests/test_rag_engine.py - Behaviour tests for the RAG pipeline.

Every external dependency (HuggingFaceEmbeddings, FAISS, ChatGroq) is patched,
so this runs offline with no API key and no model download. The fakes are small
and inspectable on purpose: a FakeLLM records the prompts it was sent, which is
how the relevance threshold and the condensing step are asserted, rather than
asserting that some mock was called.

The tests here target the fixes that matter:

* per-session isolation - documents and history belong to one engine, models are
  shared, which replaced one process-wide @st.cache_resource engine.
* the relevance threshold - chunks past max_distance never reach the prompt.
* the regression - an answer that merely CONTAINS "could not find information"
  is returned unchanged, where the old code swapped in a whole-document summary.
* _is_broad_request - the replacement for a keyword regex that misrouted
  ordinary questions such as "What are the main points of contact?".
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from config import RAGConfig, VectorStoreConfig
from rag_engine import RAGEngine, ModelRegistry, _is_broad_request
from ui_helpers import source_label

API_KEY = "gsk_test_DISTINCTIVE_9f3a2b_never_log_this"

DASH = "—"  # the separator ui_helpers.source_label emits between name and page


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Stand-in for a FAISS store.

    Records every query string so a test can prove what reached the retriever,
    and writes real files on save_local so the atomic swap in _save_index can be
    exercised without the native library.
    """

    def __init__(self, scored=None, ntotal=0):
        self.scored = list(scored or [])
        self.queries = []
        self.added = []
        self.index = SimpleNamespace(ntotal=ntotal)

    def similarity_search_with_score(self, query, k=None):
        self.queries.append(query)
        return list(self.scored)

    def add_documents(self, documents):
        # The in-memory branch of RAGEngine.add_documents appends here rather
        # than rebuilding the store, which is what keeps earlier chunks when
        # persistence is off.
        self.added.extend(documents)
        self.index.ntotal += len(documents)

    def save_local(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "index.faiss").write_text("fake")
        (path / "index.pkl").write_text("fake")

    def merge_from(self, other):
        self.index.ntotal += other.index.ntotal


class FakeLLM:
    """Returns fixed text while recording every prompt it was sent.

    `_chat` is used for both condensing and answering, so the prompt content
    distinguishes them: the condense prompt ends with "Standalone query:".
    """

    def __init__(self, reply="the answer", condense_reply=None, raise_on_condense=False):
        self.reply = reply
        self.condense_reply = condense_reply
        self.raise_on_condense = raise_on_condense
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        if "Standalone query:" in prompt:
            if self.raise_on_condense:
                raise RuntimeError("condense exploded")
            return SimpleNamespace(
                content=self.condense_reply if self.condense_reply is not None else self.reply
            )
        return SimpleNamespace(content=self.reply)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config with a real API key and a temp index path."""
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    config = RAGConfig()
    config.vector_store = VectorStoreConfig(index_path=tmp_path / "idx")
    config.llm.groq_api_key = API_KEY
    return config


def make_doc(name, page, content):
    metadata = {"file_name": name, "chunk_index": 0}
    if page is not None:
        metadata["page_label"] = page
    return Document(page_content=content, metadata=metadata)


def make_engine(config, store=None, llm=None):
    """Build an engine with patched models, then optionally attach a store.

    The model objects are built inside the patch so they are cached before the
    context exits; otherwise a later query would build a real ChatGroq.
    """
    llm = llm if llm is not None else FakeLLM()
    embeddings = MagicMock()
    with patch("rag_engine.HuggingFaceEmbeddings", return_value=embeddings), patch(
        "rag_engine.ChatGroq", return_value=llm
    ):
        engine = RAGEngine(config=config)
        engine._models.embeddings()
        engine._models.llm()
    if store is not None:
        engine._vector_store = store
    return engine


def add_documents(engine, docs, store):
    """Run the real add_documents against a fake vector store.

    Only FAISS.from_documents is patched; _save_index runs for real so the
    atomic directory swap is exercised.
    """
    with patch("rag_engine.FAISS.from_documents", return_value=store):
        engine.add_documents(docs)


# ---------------------------------------------------------------------------
# Broad-request routing (was the old _BROAD_QUESTION_RE)
# ---------------------------------------------------------------------------


# The 14 questions the new rule calls broad. These are genuine whole-document
# requests and must reach the overview path.
_BROAD_TRUE = [
    "summarise this document",
    "Summarize the paper",
    "give me an overview",
    "tl;dr",
    "tldr",
    "summary",
    "overview",
    "what is this document about",
    "what's this pdf about",
    "key points",
    "key takeaways",
    "main points",
    "main themes",
    "can you summarise this",
]

# The 9 the rule must NOT call broad. Two of them are the regression cases that
# forced the keyword regex to be replaced: "main points" appears inside "main
# points of contact", and "overview" inside a question about a rejected
# overview. Routing either to the whole-document path would throw away the
# retrieval that actually answers it.
_BROAD_FALSE = [
    "What is the revenue on page 4?",
    "The FY2023 audit could not find information to support the reported "
    "revenue, but page 4 states the figure was RM 2.1 million.",
    "Who signed the contract?",
    "summarization technique used in the report",
    "What are the main points of contact?",  # regex false positive
    "overview of the methodology was rejected by whom?",  # regex false positive
    "summarise the revenue figures for 2023",
    "Summarise the indemnity clause in section 4.2 of the agreement",
    "",
]


class TestIsBroadRequest:
    @pytest.mark.parametrize("question", _BROAD_TRUE)
    def test_true_for_whole_document_requests(self, question):
        assert _is_broad_request(question) is True

    @pytest.mark.parametrize("question", _BROAD_FALSE)
    def test_false_for_specific_questions(self, question):
        assert _is_broad_request(question) is False

    def test_none_is_not_broad(self):
        assert _is_broad_request(None) is False


# ---------------------------------------------------------------------------
# Initialisation and lifecycle
# ---------------------------------------------------------------------------


class TestRAGEngineInit:
    def test_creates_index_directory(self, cfg):
        RAGEngine(config=cfg)
        assert cfg.vector_store.index_path.parent.exists()

    def test_index_does_not_exist_initially(self, cfg):
        engine = make_engine(cfg)
        assert engine.index_exists is False

    def test_query_before_documents_raises_not_ready(self, cfg):
        engine = make_engine(cfg)
        with pytest.raises(RuntimeError, match="not ready"):
            engine.query("What is FAISS?")

    def test_load_existing_index_raises_when_missing(self, cfg):
        engine = make_engine(cfg)
        with pytest.raises(FileNotFoundError):
            engine.load_existing_index()

    def test_add_documents_rejects_empty_list(self, cfg):
        engine = make_engine(cfg)
        with pytest.raises(ValueError):
            engine.add_documents([])


# ---------------------------------------------------------------------------
# Per-session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    def test_two_engines_from_same_config_share_no_instance_state(self, cfg):
        a = make_engine(cfg)
        b = make_engine(cfg)

        assert a._docs is not b._docs
        assert a._history is not b._history

        store = FakeVectorStore(ntotal=1)
        doc = make_doc("secret.pdf", "Page 1", "A's private content")
        add_documents(a, [doc], store)

        # A's upload is visible to A and invisible to B. This is the defect the
        # old shared @st.cache_resource engine caused: one visitor's documents
        # were retrievable by the next.
        assert a._docs == [doc]
        assert a._vector_store is store
        assert b._docs == []
        assert b._vector_store is None
        with pytest.raises(RuntimeError, match="not ready"):
            b.query("anything")

    def test_two_engines_have_independent_history(self, cfg):
        a = make_engine(cfg)
        b = make_engine(cfg)
        a._history.append(("q", "a"))
        assert len(a._history) == 1
        assert len(b._history) == 0


# ---------------------------------------------------------------------------
# Model sharing
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_shared_registry_returns_identical_models(self, cfg):
        embeddings = MagicMock()
        llm = FakeLLM()
        with patch("rag_engine.HuggingFaceEmbeddings", return_value=embeddings), patch(
            "rag_engine.ChatGroq", return_value=llm
        ):
            registry = ModelRegistry(cfg)
            a = RAGEngine(config=cfg, models=registry)
            b = RAGEngine(config=cfg, models=registry)
            assert a._get_embeddings() is b._get_embeddings() is embeddings
            assert a._get_llm() is b._get_llm() is llm

    def test_engines_without_registry_get_private_models(self, cfg):
        # A fresh model object per call makes it obvious whether two engines
        # share a registry or each built their own.
        with patch(
            "rag_engine.HuggingFaceEmbeddings", side_effect=lambda **kw: MagicMock()
        ), patch("rag_engine.ChatGroq", side_effect=lambda **kw: FakeLLM()):
            a = RAGEngine(config=cfg)
            b = RAGEngine(config=cfg)
            assert a._get_embeddings() is not b._get_embeddings()
            assert a._get_llm() is not b._get_llm()


# ---------------------------------------------------------------------------
# Retrieval threshold and prompt assembly
# ---------------------------------------------------------------------------


class TestRelevanceThreshold:
    def test_only_in_range_chunks_reach_the_prompt(self, cfg):
        near = make_doc("near.pdf", "Page 1", "the revenue figure was RM 2.1 million")
        far = make_doc("far.pdf", "Page 9", "an unrelated paragraph about the weather")
        # Default max_distance is 1.4: 0.4 is relevant, 2.9 is not.
        store = FakeVectorStore(scored=[(near, 0.4), (far, 2.9)], ntotal=2)
        llm = FakeLLM(reply="The figure was RM 2.1 million.")
        engine = make_engine(cfg, store=store, llm=llm)

        result = engine.query("What was the revenue figure?")

        qa_prompt = [p for p in llm.prompts if "Answer:" in p][-1]
        assert source_label(near.metadata) in qa_prompt
        assert near.page_content in qa_prompt
        # The far chunk is neither labelled nor quoted, so the model cannot
        # invent a citation for it.
        assert source_label(far.metadata) not in qa_prompt
        assert far.page_content not in qa_prompt
        assert result["sources"] == [source_label(near.metadata)]

    def test_all_chunks_above_threshold_routes_to_overview(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "some indexed text")
        store = FakeVectorStore(scored=[(doc, 5.0)], ntotal=1)
        llm = FakeLLM(reply="Overview answer")
        engine = make_engine(cfg, store=store, llm=llm)
        engine._docs = [doc]

        result = engine.query("What was the revenue figure?")

        assert result["summary"] is True
        assert result["answer"] == "Overview answer"

    def test_empty_retrieval_routes_to_overview(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "some indexed text")
        store = FakeVectorStore(scored=[], ntotal=1)
        llm = FakeLLM(reply="Overview answer")
        engine = make_engine(cfg, store=store, llm=llm)
        engine._docs = [doc]

        result = engine.query("What was the revenue figure?")

        assert result["summary"] is True
        assert result["answer"] == "Overview answer"


class TestGroundedAnswerRegression:
    GROUNDED = (
        "The FY2023 audit could not find information to support the reported "
        "revenue, but page 4 states the figure was RM 2.1 million."
    )

    def test_answer_containing_could_not_find_is_returned_unchanged(self, cfg):
        # Retrieval succeeds and the model's answer is grounded. The old code
        # scanned the ANSWER for phrases like "could not find information" and
        # replaced exactly this answer with a whole-document summary, discarding
        # a correct answer. It must now be returned verbatim.
        doc = make_doc(
            "audit.pdf",
            "Page 4",
            "FY2023 revenue was RM 2.1 million according to the audited statement.",
        )
        store = FakeVectorStore(scored=[(doc, 0.3)], ntotal=1)
        llm = FakeLLM(reply=self.GROUNDED)
        engine = make_engine(cfg, store=store, llm=llm)

        result = engine.query("What did the FY2023 audit find about revenue?")

        assert result["answer"] == self.GROUNDED
        assert result.get("summary") is not True
        assert result["sources"] == [source_label(doc.metadata)]


class TestBroadRoutingInQuery:
    def test_broad_question_uses_overview_without_retrieving(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        # Retrieval is primed to return something relevant, so if the engine
        # consulted it we could tell. The overview path must not.
        store = FakeVectorStore(scored=[(doc, 0.1)], ntotal=1)
        llm = FakeLLM(reply="Broad overview")
        engine = make_engine(cfg, store=store, llm=llm)
        engine._docs = [doc]

        result = engine.query("summarise this document")

        assert result["summary"] is True
        assert store.queries == []

    def test_regex_false_positive_still_retrieves(self, cfg):
        # "main points" appears here, but the question is about a page, not the
        # whole document. It must go down the retrieval path.
        doc = make_doc("contacts.pdf", "Page 2", "the main points of contact are listed")
        store = FakeVectorStore(scored=[(doc, 0.3)], ntotal=1)
        llm = FakeLLM(reply="They are listed on page 2.")
        engine = make_engine(cfg, store=store, llm=llm)
        engine._docs = [doc]

        result = engine.query("What are the main points of contact?")

        assert result.get("summary") is not True
        assert store.queries == ["What are the main points of contact?"]


# ---------------------------------------------------------------------------
# Memory and condensing
# ---------------------------------------------------------------------------


class TestMemory:
    def test_history_is_bounded_to_the_window(self, cfg):
        cfg.memory.window_turns = 3
        doc = make_doc("a.pdf", "Page 1", "text about revenue")
        store = FakeVectorStore(scored=[(doc, 0.2)], ntotal=1)
        llm = FakeLLM(reply="ok")
        engine = make_engine(cfg, store=store, llm=llm)

        window = cfg.memory.window_turns
        for i in range(window + 5):
            engine.query(f"What was the revenue figure in attempt {i}?")

        assert len(engine._history) == window

    def test_clear_memory_empties_history(self, cfg):
        engine = make_engine(cfg)
        engine._history.append(("q", "a"))
        assert len(engine._history) == 1
        engine.clear_memory()
        assert len(engine._history) == 0


class TestCondensing:
    def test_question_passes_through_unchanged_without_history(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(scored=[(doc, 0.2)], ntotal=1)
        llm = FakeLLM(reply="answer", condense_reply="SHOULD NOT BE USED")
        engine = make_engine(cfg, store=store, llm=llm)

        engine.query("Who signed the contract?")

        assert store.queries == ["Who signed the contract?"]
        assert all("Standalone query:" not in p for p in llm.prompts)

    def test_condensed_query_reaches_the_retriever(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(scored=[(doc, 0.2)], ntotal=1)
        condensed = "How many employees does Acme have?"
        llm = FakeLLM(reply="the answer", condense_reply=condensed)
        engine = make_engine(cfg, store=store, llm=llm)
        engine._history.append(("Tell me about Acme.", "Acme is a company."))

        engine.query("and how many of them?")

        assert store.queries == [condensed]

    def test_condense_failure_falls_back_to_the_original_question(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(scored=[(doc, 0.2)], ntotal=1)
        llm = FakeLLM(reply="the answer", raise_on_condense=True)
        engine = make_engine(cfg, store=store, llm=llm)
        engine._history.append(("Tell me about Acme.", "Acme is a company."))

        engine.query("and how many of them?")

        # A failed rewrite costs relevance, never the whole request.
        assert store.queries == ["and how many of them?"]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_every_passage_is_preceded_by_its_label(self, cfg):
        d1 = make_doc("report.pdf", "Page 3", "the clause reads as follows")
        d2 = make_doc("report.pdf", "Page 4", "and continues here")
        context = RAGEngine._format_context([d1, d2])

        for doc in (d1, d2):
            label = source_label(doc.metadata)
            assert f"[{label}]" in context
            # The label appears, and the passage follows it.
            assert context.index(f"[{label}]") < context.index(doc.page_content)

    def test_label_uses_the_em_dash_separator(self, cfg):
        doc = make_doc("report.pdf", "Page 3", "content")
        context = RAGEngine._format_context([doc])
        assert f"report.pdf {DASH} Page 3" in context


# ---------------------------------------------------------------------------
# Logging safety
# ---------------------------------------------------------------------------


class TestLoggingSafety:
    def test_api_key_never_reaches_the_logs(self, cfg, caplog):
        # The engine logs its configuration on every construction. If it logs
        # model_dump() rather than redacted(), the live key goes to stdout. This
        # is the test for that leak.
        cfg.llm.groq_api_key = API_KEY
        assert cfg.model_dump()["llm"]["groq_api_key"] == API_KEY

        with caplog.at_level(logging.INFO, logger="rag_engine"):
            RAGEngine(config=cfg)

        assert any(r.name == "rag_engine" for r in caplog.records)
        rendered = "\n".join(r.getMessage() for r in caplog.records)
        assert API_KEY not in caplog.text
        assert API_KEY not in rendered
        # And the masked form really is what was written, so this is not
        # passing merely because no config was logged at all.
        assert "***redacted***" in caplog.text


# ---------------------------------------------------------------------------
# Index persistence and teardown
# ---------------------------------------------------------------------------


class TestIndexPersistence:
    def test_save_index_is_atomic_and_leaves_no_staging_directory(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(ntotal=1)
        engine = make_engine(cfg)

        add_documents(engine, [doc], store)

        index_dir = cfg.vector_store.index_path
        assert (index_dir / "index.faiss").exists()
        # The save writes a sibling directory and renames it over the target, so
        # a reader never sees half a pair of files.
        leftovers = list(index_dir.parent.glob(f"{index_dir.name}.tmp-*"))
        assert leftovers == []

    def test_clear_documents_empties_state_and_disk(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(ntotal=1)
        engine = make_engine(cfg)
        add_documents(engine, [doc], store)
        engine._history.append(("q", "a"))
        index_dir = cfg.vector_store.index_path
        assert index_dir.exists()

        engine.clear_documents()

        assert engine._docs == []
        assert len(engine._history) == 0
        assert engine._vector_store is None
        assert not index_dir.exists()


class TestGetIndexStats:
    def test_shape_when_empty(self, cfg):
        engine = make_engine(cfg)
        assert engine.get_index_stats() == {"indexed": False, "total_vectors": 0}

    def test_shape_when_populated(self, cfg):
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(ntotal=3)
        engine = make_engine(cfg)
        add_documents(engine, [doc], store)

        stats = engine.get_index_stats()
        assert stats["indexed"] is True
        assert stats["total_vectors"] == 3
        assert stats["index_path"] == str(cfg.vector_store.index_path)


class TestPersistDisabled:
    def test_second_upload_keeps_the_first_when_not_persisting(self, cfg):
        # The per-session app engines set persist=False. The old code chose the
        # branch on index_exists, so with no index on disk the second upload
        # rebuilt the store from only the new chunks and silently dropped the
        # first set. The branch must be on the in-memory store instead.
        cfg.vector_store.persist = False
        d1 = make_doc("one.pdf", "Page 1", "first document")
        d2 = make_doc("two.pdf", "Page 1", "second document")

        created = []

        def from_documents(documents, embeddings):
            store = FakeVectorStore(ntotal=len(documents))
            store.added.extend(documents)
            created.append(store)
            return store

        engine = make_engine(cfg)
        with patch("rag_engine.FAISS.from_documents", side_effect=from_documents):
            engine.add_documents([d1])
            engine.add_documents([d2])

        # Only one store was ever built; the second call appended to it.
        assert len(created) == 1
        store = created[0]
        assert store.added == [d1, d2]
        assert store.index.ntotal == 2
        assert engine._docs == [d1, d2]
        # Nothing is written to disk when persistence is off, so no per-session
        # directory is left behind.
        assert not cfg.vector_store.index_path.exists()

    def test_clear_documents_is_safe_when_not_persisting(self, cfg):
        cfg.vector_store.persist = False
        doc = make_doc("a.pdf", "Page 1", "text")
        store = FakeVectorStore(ntotal=1)
        engine = make_engine(cfg)
        add_documents(engine, [doc], store)

        engine.clear_documents()

        assert engine._docs == []
        assert engine._vector_store is None

