"""
tests/test_app.py - Headless rendering tests for the Streamlit app.

app.py cannot be imported: importing it runs the whole script. Streamlit 1.45's
``AppTest`` runs it headlessly instead, which is the only way to reach the wiring
that lives nowhere else: session-state handling, the escaped interpolation
sinks, and the query error path. These are exactly the places the coverage
number was blind to, because pytest could not touch app.py at all.

The model is never exercised. A fake engine is placed in ``st.session_state``
before the first run, so ``_get_engine`` returns it instead of building a real
one. Nothing reaches Groq and nothing downloads embeddings. Assertions are made
against rendered elements, never against internals: this tests what a visitor
would actually see.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
from ui_helpers import GENERIC_ERROR

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def dummy_api_key(monkeypatch):
    # The engine is faked, but a key is kept in the environment so any lazily
    # built model client has something to read rather than raising on import.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_dummy_key_for_app_tests")


class FakeEngine:
    """The slice of RAGEngine's surface that app.py actually touches."""

    def __init__(self, indexed=False, query_impl=None):
        self.indexed = indexed
        self.query_impl = query_impl
        self.memory_cleared = False
        self.documents_cleared = False

    def get_index_stats(self):
        return {
            "indexed": self.indexed,
            "total_vectors": 3 if self.indexed else 0,
            "index_path": "/tmp/fake_index",
        }

    def query(self, question):
        if self.query_impl is not None:
            return self.query_impl(question)
        return {"answer": "fake answer", "sources": []}

    def clear_memory(self):
        self.memory_cleared = True

    def clear_documents(self):
        self.documents_cleared = True


def run_app(engine, *, chat_history=None, timeout=20):
    """Run app.py with a fake engine injected into session state."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    at.session_state["engine"] = engine
    if chat_history is not None:
        at.session_state["chat_history"] = chat_history
    at.run()
    return at


def rendered_markdown(at):
    """Every markdown string the app rendered, including its inline HTML."""
    return "\n".join(element.value for element in at.markdown)


class TestRendersWithoutDocuments:
    def test_no_exception_on_first_load(self):
        at = run_app(FakeEngine(indexed=False))
        assert list(at.exception) == []

    def test_hero_text_is_present(self):
        at = run_app(FakeEngine(indexed=False))
        assert "Ask anything." in rendered_markdown(at)

    def test_chat_input_exists_and_is_disabled(self):
        # The input is disabled until something is indexed, so a question
        # cannot be asked against an empty index.
        at = run_app(FakeEngine(indexed=False))
        assert len(at.chat_input) == 1
        assert at.chat_input[0].disabled is True

    def test_chat_input_enabled_once_indexed(self):
        at = run_app(FakeEngine(indexed=True))
        assert at.chat_input[0].disabled is False


class TestFooterEscaping:
    def test_model_name_markup_is_escaped_in_the_footer(self, monkeypatch):
        # The footer interpolates the configured model name into raw HTML. A
        # name is deployment-controlled, but it is still interpolated without a
        # sanitiser, so it must go through escape().
        payload = '"><img src=x onerror=alert(1)>'
        monkeypatch.setattr(config.settings.llm, "model_name", payload)

        at = run_app(FakeEngine(indexed=False))
        rendered = rendered_markdown(at)

        assert "<img" not in rendered
        assert "&lt;img" in rendered
        # The footer really did render, so this is not passing because the
        # block is missing.
        assert "LANGCHAIN" in rendered


class TestChatHistoryRendering:
    def test_iframe_payload_in_history_is_not_emitted_raw(self):
        payload = '<iframe srcdoc="<script>alert(1)</script>"></iframe>'
        history = [
            {"role": "user", "content": "look at this", "sources": []},
            {
                "role": "assistant",
                "content": payload,
                "sources": ["report.pdf - Page 1"],
            },
        ]
        at = run_app(FakeEngine(indexed=True), chat_history=history)
        rendered = rendered_markdown(at)

        assert list(at.exception) == []
        assert "<iframe" not in rendered
        # The content was escaped and shown, rather than dropped.
        assert "&lt;iframe" in rendered
        # The source chip, escaped separately, is present too.
        assert "report.pdf" in rendered


class TestQueryErrorHandling:
    def test_exception_text_is_never_rendered(self):
        # The engine's exception carries a filesystem path. The app must show
        # the generic message and keep the path in the log only.
        secret_path = r"C:\secret\internal\path.py"

        def boom(question):
            raise RuntimeError(f"failed while reading {secret_path}")

        at = run_app(FakeEngine(indexed=True, query_impl=boom))
        at.chat_input[0].set_value("what does this say?").run()

        assert list(at.exception) == []
        rendered = rendered_markdown(at)
        assert GENERIC_ERROR in rendered
        assert secret_path not in rendered
        assert "secret" not in rendered

    def test_successful_query_answer_is_rendered(self):
        # The same wiring, on the success path, so the error test is not the
        # only thing exercising the chat input.
        at = run_app(
            FakeEngine(
                indexed=True,
                query_impl=lambda q: {"answer": "The answer is 42.", "sources": ["a.pdf - Page 2"]},
            )
        )
        at.chat_input[0].set_value("what is the answer?").run()

        rendered = rendered_markdown(at)
        assert list(at.exception) == []
        assert "The answer is 42." in rendered
        assert "a.pdf" in rendered
