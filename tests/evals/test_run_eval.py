import json
from types import SimpleNamespace

import pytest

from evals.run_eval import run_answers, run_retrieval, summarise, with_retry


def doc(page):
    return SimpleNamespace(metadata={"page_label": f"Page {page}"})


class FakeEngine:
    def __init__(self, pages):
        self.pages = pages
        self.asked_with = []

    def retrieve(self, question, k=None):
        self.asked_with.append(k)
        return [doc(page) for page in self.pages]


def test_run_retrieval_asks_for_enough_chunks_to_score_recall_at_ten():
    # recall@10 is meaningless when retrieval stops at four chunks, which is
    # what retriever_k returns for answering.
    engine = FakeEngine([1, 2])
    run_retrieval(
        [
            {
                "id": "q1",
                "type": "fact",
                "question": "Where?",
                "reference_answer": "Page 2.",
                "doc": "attention",
                "gold_pages": [2],
            }
        ],
        engine,
    )
    assert engine.asked_with == [10]


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


class RateLimitError(Exception):
    """Stands in for the provider SDK's 429, matched by class name."""


def test_retry_returns_the_first_successful_result():
    assert with_retry(lambda: "answer", sleep=lambda _: None) == "answer"


def test_retry_backs_off_and_then_succeeds():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("429")
        return "answer"

    slept = []
    assert with_retry(flaky, delays=(2, 5), sleep=slept.append) == "answer"
    assert len(attempts) == 3
    assert slept == [2, 5]


def test_retry_reraises_other_errors_without_sleeping():
    slept = []

    def broken():
        raise ValueError("bad prompt")

    with pytest.raises(ValueError):
        with_retry(broken, delays=(2, 5), sleep=slept.append)
    assert slept == []


def test_retry_gives_up_after_the_last_delay():
    def always_limited():
        raise RateLimitError("429")

    slept = []
    with pytest.raises(RateLimitError):
        with_retry(always_limited, delays=(2, 5), sleep=slept.append)
    assert slept == [2, 5]


class DailyCapError(RateLimitError):
    """A 429 whose message names the per-day allowance."""

    def __init__(self):
        super().__init__("tokens per day (TPD): Limit 200000")


def test_retry_does_not_wait_on_a_daily_cap():
    slept = []

    def capped():
        raise DailyCapError()

    with pytest.raises(DailyCapError):
        with_retry(capped, delays=(2, 5), sleep=slept.append)
    assert slept == []


class RecordingEngine:
    def __init__(self):
        self.asked = []

    def query(self, question):
        self.asked.append(question)
        return {"answer": "An answer.", "sources": [], "contexts": ["chunk"]}


class FakeJudge:
    def score_faithfulness(self, question, answer, context):
        return {"score": 1.0, "claims": 1, "supported": 1, "total": 1, "model": "fake"}


def golden_item(qid):
    return {
        "id": qid,
        "type": "fact",
        "question": f"Question {qid}?",
        "reference_answer": "An answer.",
        "doc": "attention",
        "gold_pages": [1],
    }


def test_run_answers_writes_each_row_to_the_checkpoint(tmp_path):
    checkpoint = tmp_path / "answers.jsonl"
    rows = run_answers(
        [golden_item("q1"), golden_item("q2")],
        RecordingEngine(),
        FakeJudge(),
        checkpoint,
    )
    assert [row["id"] for row in rows] == ["q1", "q2"]
    written = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert [row["id"] for row in written] == ["q1", "q2"]


def test_run_answers_skips_questions_already_in_the_checkpoint(tmp_path):
    checkpoint = tmp_path / "answers.jsonl"
    checkpoint.write_text(json.dumps({"id": "q1", "answer": "kept"}) + "\n")
    engine = RecordingEngine()
    rows = run_answers(
        [golden_item("q1"), golden_item("q2")], engine, FakeJudge(), checkpoint
    )
    assert engine.asked == ["Question q2?"]
    assert rows[0]["answer"] == "kept"
    assert [row["id"] for row in rows] == ["q1", "q2"]
