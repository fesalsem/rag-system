"""
rag_engine.py — Core RAG pipeline.
Encapsulates embeddings, vector store, LLM and conversation history so every
concern is swappable without touching app.py or the processor.

Two objects, deliberately split by lifetime:

* :class:`ModelRegistry` holds the embedding model and the LLM client. Both are
  stateless, so one registry is shared by every user and the expensive model
  load happens once per process.
* :class:`RAGEngine` holds the FAISS index and the conversation history. Both
  belong to a single user, so the app creates one engine per browser session.

That split is the fix for the worst defect this code had. The whole engine used
to sit behind `@st.cache_resource`, which Streamlit documents as shared "across
all users, sessions, and reruns", so one visitor's uploads were retrievable by
the next visitor and one visitor pressing "Clear Conversation" wiped everybody's
history.
"""

import logging
import re
import shutil
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import RAGConfig, settings
from ui_helpers import source_label, source_labels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Answers strictly from the retrieved context. Every passage is handed over
# with a [file — page] label.
#
# The prompt used to demand "Always end your answer with a Sources: section
# listing the relevant page references" while the context handed to the model
# was assembled by StuffDocumentsChain's default document_prompt,
# "{page_content}" — no file name, no page, not even a boundary between
# passages. The model could not see a page number, so every citation it wrote
# was invented, and the UI rendered a second, correct set of chips beside it.
# Labels are now in the context, and the chips derived from real retrieval
# metadata are the citation surface, so nothing asks the model to guess.
_QA_TEMPLATE = """You are a precise and helpful assistant answering questions about a user's documents.

Answer using ONLY the context below. If the context does not contain the answer, say
"I could not find information about that in the uploaded documents."

Each context passage is preceded by a label in square brackets giving its file and page.
If you state where a fact came from, use those labels exactly as written. Never name a
page that does not appear in the context.

Context:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:"""


# Rewrites a follow-up into something retrievable on its own. Without this,
# "what about the second one?" retrieves on those words alone.
_CONDENSE_TEMPLATE = """Given the conversation so far and a follow-up message, rewrite the follow-up as a
standalone search query that can be understood without the conversation.

Reply with the query only, on one line, with no preamble. If the follow-up already
stands alone, return it unchanged.

Conversation:
{chat_history}

Follow-up: {question}

Standalone query:"""


# Used when retrieval finds nothing relevant, or when the question is plainly a
# whole-document request. Answers from an evenly-spaced spread of the document
# instead of a top-k retrieval, because there is no single passage to retrieve.
_SUMMARY_TEMPLATE = """You are a helpful assistant. The question below is broad or does not
map to a single passage, so you are given a spread of excerpts from the document(s). Answer
the question as a helpful overview based ONLY on these excerpts. Be concise and use short
bullets where helpful.

Document excerpts:
{context}

Question: {question}

Overview:"""


# Whole-document requests, which retrieval cannot serve: no single chunk answers
# "summarise this". This replaces the old trigger, which scanned the model's
# *answer* for phrases like "could not find information" and therefore threw
# away correctly grounded answers that happened to contain one of them ("the
# FY2023 audit could not find information to support the reported revenue, but
# page 4 states ..."). Matching the question is deterministic, and cannot
# discard a good answer.
_SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summari[sz]e|summar(?:y|ise|ize)|overview|tl;?dr|"
    r"key (?:points|takeaways)|main (?:points|ideas|themes))\b",
    re.IGNORECASE,
)
_DOCUMENT_WORD_RE = re.compile(
    r"\b(?:document|documents?|pdfs?|files?|papers?|docs?|text|upload|uploads)\b",
    re.IGNORECASE,
)
_WHOLE_DOCUMENT_RE = re.compile(
    r"what(?:'s| is)\s+(?:this|the)\s+(?:document|pdf|file|paper|doc|text)\b.*\babout\b",
    re.IGNORECASE,
)
# A request has to be this short before it counts as nothing but the request
# itself. See _is_broad_request for why the bound is deliberately tight.
_BROAD_MAX_WORDS = 4


def _is_broad_request(question: str) -> bool:
    """
    True when the question asks for the document as a whole.

    Keyword matching alone is not enough. "main points" appears in "What are the
    main points of contact?", and "overview" in "overview of the methodology was
    rejected by whom?" — both are ordinary questions about a specific passage,
    and sending them down the overview path would discard the retrieval that
    answers them. So a request also has to name the document, or be short enough
    that there is nothing else in it ("key takeaways", "give me an overview").

    The bound is deliberately tight and the rule errs towards False. A missed
    overview request produces a narrower answer from a real passage, while a
    false positive throws away retrieval entirely, which is the failure this
    whole path exists to avoid. Questions that match nothing relevant are still
    caught, by the distance threshold rather than by wording.
    """
    text = (question or "").strip()
    if not text:
        return False
    if _WHOLE_DOCUMENT_RE.search(text):
        return True
    if not _SUMMARY_INTENT_RE.search(text):
        return False
    return bool(_DOCUMENT_WORD_RE.search(text)) or len(text.split()) <= _BROAD_MAX_WORDS


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Lazily-built embedding model and LLM client.

    Stateless once built, so one registry can serve every session. The lock
    stops two sessions that arrive at the same moment from both paying for the
    model load.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config: RAGConfig = config or settings
        self._lock = threading.Lock()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._llm: Optional[ChatGroq] = None

    def embeddings(self) -> HuggingFaceEmbeddings:
        with self._lock:
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
                        encode_kwargs={
                            "normalize_embeddings": True,
                            "batch_size": 128,
                        },
                    )
                except Exception as exc:
                    logger.error("Failed to load embeddings: %s", exc, exc_info=True)
                    raise
        return self._embeddings

    def llm(self) -> ChatGroq:
        with self._lock:
            if self._llm is None:
                llm_cfg = self.config.llm
                if not llm_cfg.groq_api_key:
                    raise EnvironmentError(
                        "GROQ_API_KEY is not set. Add it to your .env file "
                        "(locally) or to Streamlit secrets (cloud)."
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
                    logger.error(
                        "Failed to initialise Groq LLM: %s", exc, exc_info=True
                    )
                    raise
        return self._llm


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------


class RAGEngine:
    """
    Retrieval-augmented generation over one user's documents.

    Build one per session. Documents, the FAISS index and the conversation
    history are instance state; the embedding model and LLM come from a shared
    :class:`ModelRegistry`.

    Parameters
    ----------
    config : RAGConfig, optional
        Override the module-level ``settings`` singleton for testing.
    models : ModelRegistry, optional
        Shared model registry. A private one is built when omitted, which is
        what tests and one-off scripts want.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        models: ModelRegistry | None = None,
    ) -> None:
        self.config: RAGConfig = config or settings
        self._models: ModelRegistry = models or ModelRegistry(self.config)
        self._vector_store: Optional[FAISS] = None
        # Kept so the overview fallback can sample the whole document without
        # reaching into FAISS internals.
        self._docs: List[Document] = []
        self._history: deque[tuple[str, str]] = deque(
            maxlen=self.config.memory.window_turns
        )
        # Guarded because Streamlit runs each session in its own thread, and a
        # rerun can overlap a previous run of the same session.
        self._lock = threading.Lock()

        self._ensure_index_dir()
        logger.info("RAGEngine initialised with config: %s", self.config.redacted())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_index_dir(self) -> None:
        """Create the vector store directory if it doesn't exist."""
        if not self.config.vector_store.persist:
            # Nothing will be written, so there is no directory to prepare.
            return
        index_dir = self.config.vector_store.index_path.parent
        index_dir.mkdir(parents=True, exist_ok=True)

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        return self._models.embeddings()

    def _get_llm(self) -> ChatGroq:
        return self._models.llm()

    def _chat(self, prompt: str) -> str:
        """Send one prompt and return the text of the reply."""
        response = self._get_llm().invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            # Some providers return a list of content blocks.
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        return str(content).strip()

    def _history_text(self) -> str:
        if not self._history:
            return ""
        return "\n".join(
            f"Human: {question}\nAssistant: {answer}"
            for question, answer in self._history
        )

    def _remember(self, question: str, answer: str) -> None:
        self._history.append((question, answer))

    def _condense(self, question: str) -> str:
        """
        Rewrite a follow-up as a standalone query. Returns the question
        unchanged when there is no history, or when the rewrite fails — a
        failed rewrite should cost relevance, never the whole request.
        """
        history = self._history_text()
        if not history:
            return question
        try:
            condensed = self._chat(
                _CONDENSE_TEMPLATE.format(chat_history=history, question=question)
            )
        except Exception as exc:
            logger.warning("Question condensing failed, using it verbatim: %s", exc)
            return question
        return condensed or question

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """Render passages with their citation labels, so pages can be trusted."""
        return "\n\n".join(
            f"[{source_label(doc.metadata)}]\n{doc.page_content.strip()}"
            for doc in documents
        )

    def _get_all_documents(self) -> List[Document]:
        """
        Every document in the index.

        Served from the list this engine built up. Only after
        :meth:`load_existing_index`, where the documents were never seen in this
        process, does it fall back to the FAISS docstore — whose only handle is
        the private `_dict`. That access used to fail silently and report "No
        documents are indexed yet" for a document that was indexed, so a
        failure is now logged.
        """
        if self._docs:
            return self._docs
        if self._vector_store is None:
            return []
        docstore = getattr(self._vector_store, "docstore", None)
        store = getattr(docstore, "_dict", None)
        if not store:
            logger.warning(
                "Could not read documents back from the FAISS docstore; the "
                "overview fallback will report an empty index."
            )
            return []
        return list(store.values())

    def _summarize_document(self, question: str, max_chunks: int = 12) -> Dict[str, Any]:
        """
        Whole-document overview: build an answer from an evenly-sampled spread
        of the indexed chunks rather than a single top-k retrieval.
        """
        docs = self._get_all_documents()
        if not docs:
            return {"answer": "No documents are indexed yet.", "sources": []}

        if len(docs) <= max_chunks:
            sampled = docs
        else:
            step = len(docs) / max_chunks
            sampled = [docs[int(i * step)] for i in range(max_chunks)]

        prompt = _SUMMARY_TEMPLATE.format(
            context=self._format_context(sampled), question=question
        )

        try:
            answer = self._chat(prompt)
        except Exception as exc:
            logger.error("Document summary failed: %s", exc, exc_info=True)
            return {
                "answer": "I could not generate an overview right now. "
                          "Please try a more specific question.",
                "sources": [],
            }

        sources = source_labels(sampled)
        logger.info("Overview generated from %d chunks.", len(sampled))
        return {"answer": answer, "sources": sources, "summary": True}

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
                # The index directory is written only by this app. Pickling is
                # how FAISS persists its docstore, so loading requires this, but
                # it means anything able to write that directory can execute
                # code here. Keep the path private to the app.
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc, exc_info=True)
            raise
        logger.info("FAISS index loaded — ready to answer questions.")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed ``documents`` and upsert them into the FAISS index.

        If an index already exists it is loaded first and the new documents are
        merged into it (avoiding full re-processing).

        Parameters
        ----------
        documents : List[Document]
            Pre-chunked documents from :class:`~document_processor.DocumentProcessor`.
        """
        if not documents:
            raise ValueError("documents list is empty — nothing to index.")

        embeddings = self._get_embeddings()

        with self._lock:
            if self._vector_store is not None:
                # Adding to the store already in memory. This used to branch on
                # `index_exists` instead, which rebuilt the index from only the
                # new documents whenever the index was not persisted, silently
                # dropping everything added before it.
                logger.info("Adding %d chunks to the index in memory …", len(documents))
                self._vector_store.add_documents(documents)
            elif self.index_exists:
                logger.info("Existing index found — merging new documents.")
                existing = FAISS.load_local(
                    str(self.config.vector_store.index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                existing.add_documents(documents)
                self._vector_store = existing
            else:
                logger.info("Creating new FAISS index with %d chunks …", len(documents))
                try:
                    self._vector_store = FAISS.from_documents(documents, embeddings)
                except Exception as exc:
                    logger.error("FAISS indexing failed: %s", exc, exc_info=True)
                    raise

            self._docs.extend(documents)
            if self.config.vector_store.persist:
                self._save_index()

        logger.info("Index updated — %d documents added.", len(documents))

    def _save_index(self) -> None:
        """
        Persist the current FAISS index to disk.

        Written to a sibling directory and then swapped in, because
        `save_local` writes `index.faiss` and `index.pkl` as two separate
        writes. A reader that arrived between them saw a torn index; the
        directory swap means the files a reader can see are always a complete
        pair.
        """
        if self._vector_store is None:
            return
        target = self.config.vector_store.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex[:8]}")
        self._vector_store.save_local(str(staging))
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        logger.info("FAISS index saved to '%s'.", target)

    def clear_documents(self) -> None:
        """
        Drop this session's documents and index.

        Per-session engines made this possible: previously the index was
        process-wide, so a visitor had no way to remove what they had uploaded
        and no way to avoid inheriting what someone else had.
        """
        with self._lock:
            self._vector_store = None
            self._docs = []
            self._history.clear()
            if self.config.vector_store.persist:
                shutil.rmtree(self.config.vector_store.index_path, ignore_errors=True)
        logger.info("Documents cleared for this session.")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a RAG query and return the answer with sources.

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
            If no documents have been indexed.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "The RAG chain is not ready. "
                "Load or create an index before querying."
            )

        logger.info("Query received: '%s'", question)

        # A whole-document request cannot be served by a top-k retrieval, and
        # neither can a question whose best match is still far away.
        if _is_broad_request(question):
            logger.info("Question asks for an overview — sampling the document.")
            result = self._summarize_document(question)
            self._remember(question, result["answer"])
            return result

        search_query = self._condense(question)
        try:
            scored = self._vector_store.similarity_search_with_score(
                search_query, k=self.config.retriever_k
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            raise

        # FAISS returns squared L2 distance on normalised vectors, so lower is
        # better and the cutoff is a ceiling. Distances used to be ignored
        # entirely, which fed the model the four nearest chunks whether or not
        # any of them were related — and the only guard was the model's own
        # willingness to say it could not find the answer.
        relevant = [
            doc for doc, distance in scored if distance <= self.config.max_distance
        ]
        if not relevant:
            logger.info(
                "No chunk scored better than %.2f — falling back to an overview.",
                self.config.max_distance,
            )
            result = self._summarize_document(question)
            self._remember(question, result["answer"])
            return result

        prompt = _QA_TEMPLATE.format(
            context=self._format_context(relevant),
            chat_history=self._history_text(),
            question=question,
        )
        answer = self._chat(prompt)
        sources = source_labels(relevant)
        self._remember(question, answer)

        logger.info("Query answered. Sources used: %s", sources)
        return {"answer": answer, "sources": sources}

    def clear_memory(self) -> None:
        """Reset this session's conversation history."""
        self._history.clear()
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
