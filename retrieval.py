"""Hybrid retrieval: BM25 plus dense fusion, then cross-encoder reranking.

Imports for the optional pieces are delayed until construction so that the
Streamlit app starts without rank-bm25 or a downloaded cross-encoder when
RAG_HYBRID_ENABLED is off.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Callable, List, Optional, Protocol, Sequence

from langchain_core.documents import Document

from config import RAGConfig

logger = logging.getLogger(__name__)

# How many times the answering budget to rank when a per-page cap is active.
# With a cap of one and a budget of six, every page can supply at most one of
# the six chunks, so the ranking has to be read further down to fill it.
PAGE_CAP_OVERFETCH = 4


def fetch_size(limit: int, cap: int) -> int:
    """How many candidates to request so a page cap cannot shrink ``limit``."""
    return limit if cap <= 0 else limit * PAGE_CAP_OVERFETCH


def cap_per_page(
    documents: Sequence[Document], limit: int, cap: int
) -> List[Document]:
    """Keep at most ``cap`` chunks per page, in order, up to ``limit``."""
    if cap <= 0:
        return list(documents)[:limit]
    kept: List[Document] = []
    per_page: Counter = Counter()
    for document in documents:
        page_key = (
            document.metadata.get("file_name"),
            document.metadata.get("page_label"),
        )
        if per_page[page_key] >= cap:
            continue
        per_page[page_key] += 1
        kept.append(document)
        if len(kept) == limit:
            break
    return kept


_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int, min_score: float
    ) -> List[Document]: ...


def _load_cross_encoder(model_name: str) -> Any:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder '%s' …", model_name)
            _MODEL_CACHE[model_name] = CrossEncoder(model_name)
        return _MODEL_CACHE[model_name]


class CrossEncoderReranker:
    """Orders candidate passages by relevance to the query."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int, min_score: float
    ) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
        scores = _load_cross_encoder(self.model_name).predict(pairs)
        ranked = sorted(
            zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        return [
            document for document, score in ranked if float(score) >= min_score
        ][:top_k]


def _default_bm25(documents: Sequence[Document], k: int) -> Any:
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(list(documents), k=k)


def _default_ensemble(retrievers: Sequence[Any], weights: Sequence[float]) -> Any:
    from langchain.retrievers import EnsembleRetriever

    return EnsembleRetriever(retrievers=list(retrievers), weights=list(weights))


class HybridRetriever:
    """BM25 plus dense retrieval fused with reciprocal rank fusion."""

    def __init__(
        self,
        documents: Sequence[Document],
        vector_store: Any,
        config: RAGConfig,
        reranker: Optional[Reranker] = None,
        bm25_factory: Optional[Callable[[Sequence[Document], int], Any]] = None,
        ensemble_factory: Optional[
            Callable[[Sequence[Any], Sequence[float]], Any]
        ] = None,
    ) -> None:
        if not documents:
            raise ValueError("Hybrid retrieval needs at least one chunk.")
        if vector_store is None:
            raise ValueError("Hybrid retrieval needs a vector store.")
        self.config = config
        self._documents = list(documents)
        self._vector_store = vector_store
        self._reranker = reranker or CrossEncoderReranker(config.hybrid.rerank_model)
        self._bm25_factory = bm25_factory or _default_bm25
        self._ensemble_factory = ensemble_factory or _default_ensemble
        self._ensemble = self._build()

    def _build(self) -> Any:
        hybrid = self.config.hybrid
        bm25 = self._bm25_factory(self._documents, hybrid.candidate_k)
        dense = self._vector_store.as_retriever(
            search_kwargs={"k": hybrid.candidate_k}
        )
        return self._ensemble_factory(
            [bm25, dense], [hybrid.bm25_weight, 1.0 - hybrid.bm25_weight]
        )

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        """
        Return the ``k`` best passages, defaulting to the answering budget.

        The evaluation asks for more passages than an answer needs, so that
        recall@10 measures the ranking rather than repeating recall@4.

        ``max_chunks_per_page`` limits how many passages one page may
        contribute. The reranker is asked for more than the budget when the
        cap is active, so the cap cannot silently shrink ``k``. It already
        scores the whole fused candidate pool, so the wider request costs
        nothing.
        """
        limit = k or self.config.retriever_k
        cap = self.config.max_chunks_per_page
        top_k = fetch_size(limit, cap)
        ranked = self._reranker.rerank(
            query,
            self._ensemble.invoke(query),
            top_k=top_k,
            min_score=self.config.hybrid.rerank_min_score,
        )
        return cap_per_page(ranked, limit, cap)
