"""
document_processor.py — PDF ingestion and text chunking.
Isolated from the rest of the pipeline so loaders can be swapped
(e.g., replace PyPDFLoader with UnstructuredFileLoader) without
touching RAG logic.
"""

import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import ChunkingConfig, settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Loads PDFs and splits them into overlapping chunks with rich metadata.

    Attributes
    ----------
    config : ChunkingConfig
        Chunking hyper-parameters (size, overlap, separators).
    splitter : RecursiveCharacterTextSplitter
        LangChain splitter instance built from config.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config: ChunkingConfig = config or settings.chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            add_start_index=True,   # adds `start_index` to metadata
        )
        logger.info(
            "DocumentProcessor initialised — chunk_size=%d, overlap=%d",
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_split(self, file_path: str | Path) -> List[Document]:
        """
        Load a PDF and return a list of chunked :class:`Document` objects.

        Each document's metadata will contain:
        - ``source`` : original file path
        - ``page``   : 0-based page number (added by PyPDFLoader)
        - ``start_index`` : character offset within the original page text
        - ``chunk_index`` : sequential chunk number across the whole document

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
            If no text could be extracted (e.g. scanned image-only PDF).
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
            # Normalise page number to 1-based for display purposes.
            raw_page = chunk.metadata.get("page", 0)
            chunk.metadata["page_label"] = f"Page {int(raw_page) + 1}"

        logger.info(
            "Split '%s' into %d chunk(s) (chunk_size=%d, overlap=%d)",
            file_path.name,
            len(chunks),
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        return chunks

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
