from types import SimpleNamespace

from evals.run_eval import run_retrieval, summarise


def doc(page):
    return SimpleNamespace(metadata={"page_label": f"Page {page}"})


class FakeEngine:
    def __init__(self, pages):
        self.pages = pages

    def retrieve(self, question):
        return [doc(page) for page in self.pages]


def test_run_retrieval_scores_each_question():
    golden = [
        {
            "id": "q1",
            "type": "fact",
            "question": "Where?",
            "reference_answer": "Page 2.",
            "doc": "attention",
            "gold_pages": [2],
        }
    ]
    rows = run_retrieval(golden, FakeEngine([1, 2]))
    assert rows[0]["hit@4"] is True
    assert rows[0]["reciprocal_rank"] == 0.5


def test_summarise_excludes_unanswerable_questions_from_retrieval():
    rows = [
        {
            "type": "fact",
            "hit@4": True,
            "hit@10": True,
            "reciprocal_rank": 1.0,
            "latency_ms": 10.0,
        },
        {
            "type": "unanswerable",
            "hit@4": False,
            "hit@10": False,
            "reciprocal_rank": 0.0,
            "latency_ms": 20.0,
        },
    ]
    summary = summarise(rows)
    assert summary["recall@4"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["unanswerable"] == 1
    assert summary["p95_ms"] == 20.0
