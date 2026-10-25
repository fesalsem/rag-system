import json

import pytest

from evals.golden import GoldenError, load_golden, validate


def row(**overrides):
    base = {
        "id": "fact-001",
        "type": "fact",
        "question": "What is attention?",
        "reference_answer": "A weighted lookup.",
        "doc": "attention",
        "gold_pages": [2],
    }
    base.update(overrides)
    return base


def test_valid_row_has_no_problems():
    assert validate([row()], doc_ids={"attention"}) == []


def test_duplicate_ids_are_reported():
    problems = validate([row(), row()], doc_ids={"attention"})
    assert any("duplicate id" in problem for problem in problems)


def test_unanswerable_rows_must_have_empty_gold_pages():
    problems = validate(
        [row(type="unanswerable", gold_pages=[3])], doc_ids={"attention"}
    )
    assert any("unanswerable" in problem for problem in problems)


def test_unknown_document_is_reported():
    problems = validate([row(doc="missing")], doc_ids={"attention"})
    assert any("unknown document" in problem for problem in problems)


def test_strict_mode_enforces_question_counts():
    problems = validate([row()], doc_ids={"attention"}, strict=True)
    assert any("fact" in problem and "30" in problem for problem in problems)


def test_load_golden_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    assert load_golden(path)[0]["id"] == "fact-001"


def test_load_golden_rejects_bad_json(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(GoldenError):
        load_golden(path)
