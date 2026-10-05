"""
tests/test_document_processor.py - Behaviour tests for PDF ingestion and chunking.

The processor is size-sensitive in a way that is easy to get wrong: chunk_size
is counted in TOKENS, using the embedding model's tokenizer, not characters.
These tests inject a fake tokenizer that splits on whitespace, which serves two
purposes. It keeps the suite offline (no MiniLM download), and because its token
count differs sharply from a character count it can prove the splitter sizes
chunks by the tokenizer's output rather than len(text).

The page-label tests pin the fix for a confidently wrong citation: a missing
page number used to be labelled "Page 1", so an invented source looked real.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import RAGConfig, ChunkingConfig, VectorStoreConfig
from document_processor import DocumentProcessor


class FakeTokenizer:
    """One whitespace-separated word is one token.

    Deliberately not a real tokenizer. The point is that ``tokenize`` owns the
    chunk boundary: if the processor counted characters, a run of short words
    would pack far more of them into each chunk and the chunk count would drop.
    """

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def make_processor(tmp_path, chunk_size=256, chunk_overlap=32, max_pages=300):
    """A processor with an injected tokenizer and a temp index path."""
    cfg = RAGConfig(
        chunking=ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        max_pages=max_pages,
        vector_store=VectorStoreConfig(index_path=tmp_path / "idx"),
    )
    return DocumentProcessor(config=cfg, tokenizer=FakeTokenizer()), cfg


def fake_pdf(tmp_path, name="test.pdf") -> Path:
    """A file that exists on disk; its bytes are never read because the loader
    is patched. load_and_split only checks existence before calling the loader."""
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def make_pages(contents, with_page=True):
    """Simulate what PyPDFLoader.load() returns."""
    pages = []
    for i, text in enumerate(contents):
        metadata = {"source": "mock.pdf"}
        if with_page:
            metadata["page"] = i
        pages.append(Document(page_content=text, metadata=metadata))
    return pages


class TestTokenizerInjection:
    def test_token_length_uses_injected_tokenizer(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        assert dp._token_length("one two three") == 3

    def test_tokenizer_property_returns_injected_object(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        # No download happens because the injected object short-circuits the
        # loader in the property.
        assert isinstance(dp.tokenizer, FakeTokenizer)


class TestChunkingIsMeasuredInTokens:
    def test_token_counting_yields_more_chunks_than_a_character_budget(self, tmp_path):
        # A run of one-letter words: many tokens, few characters. Counting words
        # (tokens) produces a chunk every 256 words; counting characters would
        # produce far fewer, larger chunks. That difference is the whole bug.
        text = "a " * 5000
        pages = make_pages([text])

        dp, cfg = make_processor(tmp_path, chunk_size=256, chunk_overlap=32)
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_and_split(fake_pdf(tmp_path))

        # The old behaviour: a 1000-character budget. Reproduce it here so the
        # comparison is concrete rather than a magic number.
        old_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=cfg.chunking.separators,
            length_function=len,
        )
        old_chunks = old_splitter.split_documents(pages)

        assert len(chunks) > 0
        assert len(old_chunks) > 0
        assert len(chunks) > len(old_chunks), (
            "token-sized chunks should be smaller, and therefore more numerous, "
            "than character-sized ones for token-dense text"
        )


class TestValidateSize:
    def test_passes_below_limit(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        # 0.5 MB against the 50 MB default. Should not raise.
        dp.validate_size("small.pdf", 512 * 1024)

    def test_raises_above_limit(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        cfg = dp.config
        too_big = (cfg.max_upload_mb + 1) * 1024 * 1024
        with pytest.raises(ValueError, match="MB limit"):
            dp.validate_size("big.pdf", too_big)

    def test_limit_is_inclusive(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        exact = dp.config.max_upload_mb * 1024 * 1024
        # Exactly at the limit is allowed; only "over" is rejected.
        dp.validate_size("exact.pdf", exact)


class TestPageCap:
    def test_raises_when_loader_returns_more_than_max_pages(self, tmp_path):
        dp, _ = make_processor(tmp_path, max_pages=2)
        pages = make_pages(["text one", "text two", "text three"])
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            with pytest.raises(ValueError, match="page limit"):
                dp.load_and_split(fake_pdf(tmp_path))

    def test_exactly_max_pages_is_allowed(self, tmp_path):
        dp, _ = make_processor(tmp_path, max_pages=2)
        pages = make_pages(["text one", "text two"])
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_and_split(fake_pdf(tmp_path))
        assert chunks


class TestPageLabel:
    def test_zero_based_page_zero_is_page_one(self):
        assert DocumentProcessor._page_label(0) == "Page 1"

    def test_page_two_is_page_three(self):
        assert DocumentProcessor._page_label(2) == "Page 3"

    def test_missing_page_is_none(self):
        # A missing key arrives as .get(...) returning None.
        assert DocumentProcessor._page_label(None) is None

    def test_none_is_none(self):
        assert DocumentProcessor._page_label(None) is None

    def test_non_numeric_string_is_none_and_does_not_raise(self):
        # A loader that labels pages differently must not crash ingestion, and
        # must not be turned into a guessed "Page 1" either.
        assert DocumentProcessor._page_label("front-matter") is None


class TestChunkMetadata:
    def test_chunks_carry_file_name_and_sequential_index(self, tmp_path):
        dp, _ = make_processor(tmp_path, chunk_size=4, chunk_overlap=0)
        pages = make_pages(["one two three four five six seven eight"])
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_and_split(fake_pdf(tmp_path, "my_report.pdf"))

        assert chunks
        assert all(c.metadata["file_name"] == "my_report.pdf" for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(
            range(len(chunks))
        )

    def test_page_label_present_when_page_exists(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        pages = make_pages(["Word " * 50])  # page 0 internally
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_and_split(fake_pdf(tmp_path))
        assert chunks[0].metadata["page_label"] == "Page 1"

    def test_page_label_absent_when_loader_supplies_no_page(self, tmp_path):
        # A loader that omits `page` must not have a page invented for it. The
        # key is absent, so the UI shows the file name alone.
        dp, _ = make_processor(tmp_path)
        pages = make_pages(["Word " * 50], with_page=False)
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_and_split(fake_pdf(tmp_path))
        assert chunks
        for chunk in chunks:
            assert "page_label" not in chunk.metadata
            # The other metadata still exists, so the absence is specific to
            # the page number rather than metadata being dropped wholesale.
            assert chunk.metadata["file_name"] == "test.pdf"
            assert "chunk_index" in chunk.metadata


class TestEmptyResults:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        with pytest.raises(FileNotFoundError):
            dp.load_and_split(tmp_path / "nope.pdf")

    def test_no_pages_raises(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = []
            with pytest.raises(ValueError, match="No pages extracted"):
                dp.load_and_split(fake_pdf(tmp_path))

    def test_zero_chunks_raises(self, tmp_path):
        # A page the splitter cannot turn into any chunk (for example a scanned
        # image) must be an explicit error, not an empty index. The splitter is
        # stubbed to return nothing so the guard is tested directly.
        dp, _ = make_processor(tmp_path)
        pages = make_pages(["some text"])
        with patch("document_processor.PyPDFLoader") as loader, patch.object(
            dp.splitter, "split_documents", return_value=[]
        ):
            loader.return_value.load.return_value = pages
            with pytest.raises(ValueError, match="0 chunks"):
                dp.load_and_split(fake_pdf(tmp_path))


class TestLoadMultiple:
    def test_skips_missing_files(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        assert dp.load_multiple(["/nonexistent/a.pdf", "/nonexistent/b.pdf"]) == []

    def test_combines_chunks_from_multiple_files(self, tmp_path):
        dp, _ = make_processor(tmp_path)
        pages = make_pages(["Word " * 50])
        with patch("document_processor.PyPDFLoader") as loader:
            loader.return_value.load.return_value = pages
            chunks = dp.load_multiple([fake_pdf(tmp_path, "a.pdf"), fake_pdf(tmp_path, "b.pdf")])
        assert len(chunks) > 0
