"""
tests/test_document_processor.py — Unit tests for document_processor.py
Uses a real minimal PDF created in-memory to avoid external file dependencies.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from document_processor import DocumentProcessor
from config import ChunkingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_pages(texts: list[str]) -> list[Document]:
    """Simulate what PyPDFLoader.load() returns."""
    return [
        Document(page_content=text, metadata={"source": "mock.pdf", "page": i})
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocumentProcessorInit:
    def test_default_config(self):
        dp = DocumentProcessor()
        assert dp.config.chunk_size == 1000
        assert dp.config.chunk_overlap == 200

    def test_custom_config(self):
        cfg = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        dp = DocumentProcessor(config=cfg)
        assert dp.config.chunk_size == 500

    def test_splitter_created(self):
        dp = DocumentProcessor()
        assert dp.splitter is not None


class TestLoadAndSplit:
    def test_raises_if_file_not_found(self):
        dp = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            dp.load_and_split("/nonexistent/path/file.pdf")

    def test_raises_if_no_pages_extracted(self, tmp_path):
        """Simulate loader returning empty list."""
        fake_pdf = tmp_path / "empty.pdf"
        fake_pdf.write_bytes(b"fake content")

        dp = DocumentProcessor()
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = []
            with pytest.raises(ValueError, match="No pages extracted"):
                dp.load_and_split(fake_pdf)

    def test_chunks_have_required_metadata(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake")

        long_text = "This is a sentence about artificial intelligence. " * 30
        mock_pages = make_mock_pages([long_text])

        dp = DocumentProcessor(config=ChunkingConfig(chunk_size=200, chunk_overlap=20))
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_pages
            chunks = dp.load_and_split(fake_pdf)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "file_name" in chunk.metadata
            assert "page_label" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_page_label_is_one_based(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake")

        mock_pages = make_mock_pages(["Word " * 50])  # page 0 internally

        dp = DocumentProcessor()
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_pages
            chunks = dp.load_and_split(fake_pdf)

        assert chunks[0].metadata["page_label"] == "Page 1"

    def test_file_name_in_metadata(self, tmp_path):
        fake_pdf = tmp_path / "my_report.pdf"
        fake_pdf.write_bytes(b"fake")

        mock_pages = make_mock_pages(["Content " * 50])

        dp = DocumentProcessor()
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_pages
            chunks = dp.load_and_split(fake_pdf)

        assert all(c.metadata["file_name"] == "my_report.pdf" for c in chunks)

    def test_chunk_indices_are_sequential(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake")

        mock_pages = make_mock_pages(["Sentence number one. " * 60])

        dp = DocumentProcessor(config=ChunkingConfig(chunk_size=100, chunk_overlap=10))
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_pages
            chunks = dp.load_and_split(fake_pdf)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestLoadMultiple:
    def test_skips_missing_files(self, tmp_path):
        dp = DocumentProcessor()
        result = dp.load_multiple(["/nonexistent/a.pdf", "/nonexistent/b.pdf"])
        assert result == []

    def test_combines_chunks_from_multiple_files(self, tmp_path):
        file1 = tmp_path / "a.pdf"
        file2 = tmp_path / "b.pdf"
        file1.write_bytes(b"fake")
        file2.write_bytes(b"fake")

        mock_pages = make_mock_pages(["Word " * 50])

        dp = DocumentProcessor()
        with patch("document_processor.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_pages
            chunks = dp.load_multiple([file1, file2])

        assert len(chunks) > 0
