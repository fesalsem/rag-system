"""
document_processor.py — PDF ingestion and text chunking.
Isolated from the rest of the pipeline so loaders can be swapped
(e.g., replace PyPDFLoader with UnstructuredFileLoader) without
touching RAG logic.
"""

import logging
from pathlib import Path
from typing import Any, List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import RAGConfig, settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Loads PDFs and splits them into overlapping chunks with rich metadata.

    Chunks are measured in **tokens**, not characters. The embedding model
    silently truncates at its own token limit, so a character budget lets a
    chunk hold text the embedding never sees: the tail is invisible to
    retrieval while still visible to the LLM, and the chunk can be retrieved
    for text it does not actually match. Sizing by the model's own tokenizer
    removes that gap. See :class:`~config.ChunkingConfig`.

    Parameters
    ----------
    config : RAGConfig, optional
        Root config. Chunking knobs come from ``config.chunking`` and the page
        cap from ``config.max_pages``.
    tokenizer : Any, optional
        Anything with a ``tokenize(text) -> list`` method. Injected by tests;
        in production it is loaded from the configured tokenizer name on first
        use, so constructing a processor is cheap.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config: RAGConfig = config or settings
        self._tokenizer = tokenizer
        chunking = self.config.chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
            separators=chunking.separators,
            length_function=self._token_length,
            add_start_index=True,   # adds `start_index` to metadata
        )
        logger.info(
            "DocumentProcessor initialised — chunk_size=%d tokens, overlap=%d tokens, "
            "tokenizer=%s",
            chunking.chunk_size,
            chunking.chunk_overlap,
            chunking.tokenizer_name,
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """The tokenizer, loaded on first use."""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer(
                self.config.chunking.tokenizer_name
            )
        return self._tokenizer

    @staticmethod
    def _load_tokenizer(name: str) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise RuntimeError(
                "transformers is required to size chunks by token. It ships "
                "with sentence-transformers, so its absence means the "
                "environment is incomplete."
            ) from exc
        logger.info("Loading tokenizer '%s' for chunk sizing …", name)
        return AutoTokenizer.from_pretrained(name)

    def _token_length(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_size(self, name: str, size_bytes: int) -> None:
        """
        Reject an upload before it is read into memory.

        Streamlit caps a single upload at 200 MB, which is far more than this
        app should ever hold: the file is read into RAM, then expanded into
        pages, chunks and vectors, on a process shared with every other user.

        Raises
        ------
        ValueError
            If the file exceeds ``config.max_upload_mb``.
        """
        limit = self.config.max_upload_mb * 1024 * 1024
        if size_bytes > limit:
            raise ValueError(
                f"{name} is {size_bytes / 1024 / 1024:.1f} MB, over the "
                f"{self.config.max_upload_mb} MB limit."
            )

    def load_and_split(self, file_path: str | Path) -> List[Document]:
        """
        Load a PDF and return a list of chunked :class:`Document` objects.

        Each document's metadata will contain:
        - ``source`` : original file path
        - ``page``   : 0-based page number (added by PyPDFLoader)
        - ``start_index`` : character offset within the original page text
        - ``chunk_index`` : sequential chunk number across the whole document
        - ``page_label`` : 1-based display label, present only when the loader
          supplied a usable page number

        Parameters
        ----------
        file_path : str | Path
            Absolute or relative path to the PDF file.

        Returns
        -------
        List[Document]
            Chunked documents ready for embedding.

        Raises
        ------
        FileNotFoundError
            If the PDF does not exist at the given path.
        ValueError
            If the PDF exceeds the page cap, or no text could be extracted
            (e.g. scanned image-only PDF).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info("Loading PDF: %s", file_path)

        try:
            loader = PyPDFLoader(str(file_path))
            pages: List[Document] = loader.load()
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", file_path, exc, exc_info=True)
            raise

        if not pages:
            raise ValueError(f"No pages extracted from {file_path}.")

        if len(pages) > self.config.max_pages:
            raise ValueError(
                f"{file_path.name} has {len(pages)} pages, over the "
                f"{self.config.max_pages} page limit."
            )

        logger.info("Loaded %d page(s) from '%s'", len(pages), file_path.name)

        chunks = self.splitter.split_documents(pages)

        if not chunks:
            raise ValueError(
                f"Text splitter produced 0 chunks for {file_path}. "
                "The PDF may contain only images or unsupported encoding."
            )

        # Enrich metadata with human-readable source label.
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["file_name"] = file_path.name
            page_label = self._page_label(chunk.metadata.get("page"))
            if page_label:
                chunk.metadata["page_label"] = page_label

        logger.info(
            "Split '%s' into %d chunk(s) (chunk_size=%d tokens, overlap=%d tokens)",
            file_path.name,
            len(chunks),
            self.config.chunking.chunk_size,
            self.config.chunking.chunk_overlap,
        )
        return chunks

    @staticmethod
    def _page_label(raw_page: Any) -> str | None:
        """
        Convert a loader's 0-based page number into a display label.

        Returns None rather than a guess when the loader did not supply a
        usable page number. This used to default to 0 and therefore label
        every chunk "Page 1", which turned a missing citation into a
        confidently wrong one. Swapping the loader to one that omits `page`,
        which the module docstring invites, made that the normal case.
        """
        if raw_page is None:
            return None
        try:
            return f"Page {int(raw_page) + 1}"
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric page metadata: %r", raw_page)
            return None

    def load_multiple(self, file_paths: List[str | Path]) -> List[Document]:
        """
        Convenience wrapper: load and split multiple PDFs in one call.

        Parameters
        ----------
        file_paths : List[str | Path]
            Paths to one or more PDF files.

        Returns
        -------
        List[Document]
            Combined list of chunks from all provided PDFs.
        """
        all_chunks: List[Document] = []
        for path in file_paths:
            try:
                chunks = self.load_and_split(path)
                all_chunks.extend(chunks)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping %s — %s", path, exc)
        logger.info("Total chunks across all documents: %d", len(all_chunks))
        return all_chunks
