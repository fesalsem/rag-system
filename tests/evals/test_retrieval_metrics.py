from langchain_core.documents import Document

from evals.metrics.retrieval import (
    hit_at_k,
    page_of,
    reciprocal_rank,
    summarize,
)


def doc(page):
    return Document(page_content="x", metadata={"page_label": f"Page {page}"})


def test_page_of_prefers_page_label():
    labelled = Document(page_content="x", metadata={"page_label": "Page 7", "page": 2})
    assert page_of(labelled) == 7
    assert page_of(Document(page_content="x", metadata={"page": 2})) == 3
    assert page_of(Document(page_content="x", metadata={})) is None


def test_hit_at_k_respects_the_cutoff():
    retrieved = [doc(1), doc(2)]
    assert hit_at_k(retrieved, {2}, k=2) is True
    assert hit_at_k(retrieved, {2}, k=1) is False


def test_reciprocal_rank_uses_the_first_correct_page():
    assert reciprocal_rank([doc(1), doc(5)], {5}) == 0.5
    assert reciprocal_rank([doc(1)], {9}) == 0.0


def test_summarize_averages_the_rows():
    rows = [
        {"hit@4": True, "hit@10": True, "reciprocal_rank": 1.0},
        {"hit@4": False, "hit@10": True, "reciprocal_rank": 0.5},
    ]
    assert summarize(rows, (4, 10)) == {
        "recall@4": 0.5,
        "recall@10": 1.0,
        "mrr": 0.75,
        "n": 2,
    }
