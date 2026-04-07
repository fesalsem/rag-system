"""
rag_engine.py — Core RAG pipeline.
Encapsulates embeddings, vector store, LLM, memory, and the
retrieval-augmented generation chain in a single class so every
concern is swappable without touching app.py or the processor.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from config import RAGConfig, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_QA_TEMPLATE = """You are a precise and helpful AI assistant. Answer the
question using ONLY the provided context. If the answer is not in the context,
say "I could not find information about that in the uploaded documents."

Always end your answer with a "Sources:" section listing the relevant page
references.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=_QA_TEMPLATE,
)


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------


class RAGEngine:
    """
    Production-ready Retrieval-Augmented Generation engine.

    Design principles
    -----------------
    * All external dependencies (embeddings, LLM, vector store) are
      injected via :class:`~config.RAGConfig` so they can be replaced
      by editing config.py only.
    * The FAISS index is persisted to disk; on subsequent runs the engine
      loads the existing index rather than reprocessing documents.
    * Conversation history is maintained in-memory with
      :class:`~langchain.memory.ConversationBufferMemory`.

    Parameters
    ----------
    config : RAGConfig, optional
        Override the module-level ``settings`` singleton for testing.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config: RAGConfig = config or settings
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[FAISS] = None
        self._llm: Optional[ChatGroq] = None
        self._memory: Optional[ConversationBufferMemory] = None
        self._chain: Optional[ConversationalRetrievalChain] = None

        self._ensure_index_dir()
        logger.info("RAGEngine initialised with config: %s", self.config.model_dump())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_index_dir(self) -> None:
        """Create the vector store directory if it doesn't exist."""
        index_dir = self.config.vector_store.index_path.parent
        index_dir.mkdir(parents=True, exist_ok=True)

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-initialise and cache the embedding model."""
        if self._embeddings is None:
            logger.info(
                "Loading embedding model '%s' on %s …",
                self.config.embedding.model_name,
                self.config.embedding.device,
            )
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding.model_name,
                    model_kwargs={"device": self.config.embedding.device},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                logger.error("Failed to load embeddings: %s", exc, exc_info=True)
                raise
        return self._embeddings

    def _get_llm(self) -> ChatGroq:
        """Lazy-initialise and cache the Groq LLM client."""
        if self._llm is None:
            llm_cfg = self.config.llm
            if not llm_cfg.groq_api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file."
                )
            logger.info("Connecting to Groq — model: %s", llm_cfg.model_name)
            try:
                self._llm = ChatGroq(
                    api_key=llm_cfg.groq_api_key,
                    model_name=llm_cfg.model_name,
                    temperature=llm_cfg.temperature,
                    max_tokens=llm_cfg.max_tokens,
                )
            except Exception as exc:
                logger.error("Failed to initialise Groq LLM: %s", exc, exc_info=True)
                raise
        return self._llm

    def _get_memory(self) -> ConversationBufferMemory:
        """Lazy-initialise and cache the conversation memory."""
        if self._memory is None:
            mem_cfg = self.config.memory
            self._memory = ConversationBufferMemory(
                memory_key=mem_cfg.memory_key,
                return_messages=mem_cfg.return_messages,
                output_key="answer",
            )
        return self._memory

    def _build_chain(self) -> ConversationalRetrievalChain:
        """Assemble the ConversationalRetrievalChain from cached components."""
        if self._vector_store is None:
            raise RuntimeError(
                "Vector store is not initialised. "
                "Call add_documents() or load_existing_index() first."
            )
        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retriever_k},
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=self._get_llm(),
            retriever=retriever,
            memory=self._get_memory(),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=False,
        )
        logger.info("ConversationalRetrievalChain built successfully.")
        return chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def index_exists(self) -> bool:
        """Return True if a persisted FAISS index is found on disk."""
        index_path = self.config.vector_store.index_path
        return (index_path / "index.faiss").exists()

    def load_existing_index(self) -> None:
        """
        Load a previously saved FAISS index from disk.

        Raises
        ------
        FileNotFoundError
            If no persisted index is found at the configured path.
        """
        index_path = self.config.vector_store.index_path
        if not self.index_exists:
            raise FileNotFoundError(
                f"No FAISS index found at '{index_path}'. "
                "Process documents first with add_documents()."
            )
        logger.info("Loading FAISS index from '%s' …", index_path)
        try:
            self._vector_store = FAISS.load_local(
                str(index_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc, exc_info=True)
            raise
        self._chain = self._build_chain()
        logger.info("FAISS index loaded — ready to answer questions.")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed ``documents`` and upsert them into the FAISS index.

        If an index already exists on disk it is loaded first and the new
        documents are merged into it (avoiding full re-processing).

        Parameters
        ----------
        documents : List[Document]
            Pre-chunked documents from :class:`~document_processor.DocumentProcessor`.
        """
        if not documents:
            raise ValueError("documents list is empty — nothing to index.")

        embeddings = self._get_embeddings()

        if self.index_exists:
            logger.info("Existing index found — merging new documents.")
            existing = FAISS.load_local(
                str(self.config.vector_store.index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            new_store = FAISS.from_documents(documents, embeddings)
            existing.merge_from(new_store)
            self._vector_store = existing
        else:
            logger.info("Creating new FAISS index with %d chunks …", len(documents))
            try:
                self._vector_store = FAISS.from_documents(documents, embeddings)
            except Exception as exc:
                logger.error("FAISS indexing failed: %s", exc, exc_info=True)
                raise

        self._save_index()
        self._chain = self._build_chain()
        logger.info("Index updated — %d documents added.", len(documents))

    def _save_index(self) -> None:
        """Persist the current FAISS index to disk."""
        if self._vector_store is None:
            return
        index_path = self.config.vector_store.index_path
        index_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(index_path))
        logger.info("FAISS index saved to '%s'.", index_path)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a conversational RAG query and return the answer with sources.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        Dict[str, Any]
            ``answer``  : str — model response.
            ``sources`` : List[str] — deduplicated source labels
            (e.g. "document.pdf — Page 3").

        Raises
        ------
        RuntimeError
            If the chain has not been initialised (no documents ingested).
        """
        if self._chain is None:
            raise RuntimeError(
                "The RAG chain is not ready. "
                "Load or create an index before querying."
            )

        logger.info("Query received: '%s'", question)

        try:
            result = self._chain.invoke({"question": question})
        except Exception as exc:
            logger.error("Chain invocation failed: %s", exc, exc_info=True)
            raise

        answer: str = result.get("answer", "No answer returned.")
        source_docs: List[Document] = result.get("source_documents", [])

        # Build deduplicated, human-readable source list.
        seen: set[str] = set()
        sources: List[str] = []
        for doc in source_docs:
            meta = doc.metadata
            label = (
                f"{meta.get('file_name', 'Unknown')} — "
                f"{meta.get('page_label', 'Unknown page')}"
            )
            if label not in seen:
                seen.add(label)
                sources.append(label)

        logger.info("Query answered. Sources used: %s", sources)
        return {"answer": answer, "sources": sources}

    def clear_memory(self) -> None:
        """Reset the conversation history buffer."""
        if self._memory is not None:
            self._memory.clear()
            logger.info("Conversation memory cleared.")

    def get_index_stats(self) -> Dict[str, Any]:
        """Return basic stats about the current index for display in the UI."""
        if self._vector_store is None:
            return {"indexed": False, "total_vectors": 0}
        return {
            "indexed": True,
            "total_vectors": self._vector_store.index.ntotal,
            "index_path": str(self.config.vector_store.index_path),
        }
