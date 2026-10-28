import pytest
from langchain_core.documents import Document

from config import RAGConfig
from retrieval import HybridRetriever


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return list(self.documents)


class FakeStore:
    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever([])


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k, min_score):
        self.calls.append((query, documents, top_k, min_score))
        return documents[:top_k]


def documents(count=3):
    # One page per chunk, so the default page cap does not hide the ranking
    # these tests are about.
    return [
        Document(
            page_content=f"chunk {index}",
            metadata={"file_name": "doc.pdf", "page_label": f"Page {index + 1}"},
        )
        for index in range(count)
    ]


def build_retriever(reranker=None):
    docs = documents()
    store = FakeStore()
    config = RAGConfig()
    config.retriever_k = 2
    config.hybrid.candidate_k = 20
    retriever = HybridRetriever(
        docs,
        store,
        config,
        reranker=reranker or FakeReranker(),
        bm25_factory=lambda chunks, k: FakeRetriever(chunks),
        ensemble_factory=lambda retrievers, weights: FakeRetriever(docs),
    )
    return retriever, store, config


def test_fusion_uses_the_configured_candidate_pool():
    retriever, store, _ = build_retriever()
    assert store.search_kwargs == {"k": 20}
    assert retriever.retrieve("question") == documents()[:2]


def test_reranker_receives_the_fused_candidates_and_final_k():
    reranker = FakeReranker()
    retriever, _, config = build_retriever(reranker)
    config.max_chunks_per_page = 0
    retriever.retrieve("question")
    query, candidates, top_k, min_score = reranker.calls[0]
    assert query == "question"
    assert len(candidates) == 3
    assert top_k == config.retriever_k
    assert min_score == config.hybrid.rerank_min_score


def test_page_cap_widens_the_rerank_budget():
    reranker = FakeReranker()
    retriever, _, config = build_retriever(reranker)
    retriever.retrieve("question")
    # Each page can supply at most one of the two wanted chunks, so the
    # reranker has to be read further down than k.
    assert reranker.calls[0][2] == config.retriever_k * 4


def test_page_cap_keeps_one_chunk_per_page():
    docs = [
        Document(
            page_content=f"page 1 chunk {index}",
            metadata={"file_name": "a.pdf", "page_label": "Page 1"},
        )
        for index in range(3)
    ] + [
        Document(
            page_content="page 2 chunk",
            metadata={"file_name": "a.pdf", "page_label": "Page 2"},
        )
    ]
    config = RAGConfig()
    config.retriever_k = 2
    config.max_chunks_per_page = 1
    retriever = HybridRetriever(
        docs,
        FakeStore(),
        config,
        reranker=FakeReranker(),
        bm25_factory=lambda chunks, k: FakeRetriever(chunks),
        ensemble_factory=lambda retrievers, weights: FakeRetriever(docs),
    )

    kept = retriever.retrieve("question")

    assert [doc.page_content for doc in kept] == [
        "page 1 chunk 0",
        "page 2 chunk",
    ]


def test_empty_documents_are_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([], FakeStore(), RAGConfig())
