"""Aggregation for judged answers and abstention.

Each metric averages only the rows it applies to, so an empty slice reports
None rather than a misleading zero.

A refusal is judged per sentence, not per answer. A partial answer refuses
part of the question and answers the rest; counting that as a refusal hides
the answers the system did give, and it inflates the refusal rate on questions
the corpus does cover.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# The sentence the QA prompt asks for, plus the paraphrase the model produces
# when it restates the instruction before refusing. One or two words may sit
# between the subject and "do not contain", so that "the provided documents"
# and "the excerpts" both match.
REFUSAL_STATEMENTS = (
    re.compile(r"^(?:i\s+)?could not find information\b"),
    re.compile(r"^(?:the\s+)?(?:\w+\s+){0,2}do not contain any information\b"),
)

# Citation markers are appended to the sentence they support, so they are
# stripped before splitting. Otherwise the citation reads as a second sentence
# and turns a whole refusal into a partial answer.
_CITATION = re.compile(r"【[^】]*】|\[[^\]]*[—–-][^\]]*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(answer: str | None) -> list[str]:
    text = " ".join((answer or "").split())
    text = _CITATION.sub("", text).strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip(" .")]


def _is_refusal(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(pattern.match(lowered) for pattern in REFUSAL_STATEMENTS)


def has_refusal(answer: str | None) -> bool:
    """True when any sentence of the answer refuses to answer."""
    return any(_is_refusal(sentence) for sentence in _sentences(answer))


def is_abstention(answer: str | None) -> bool:
    """True only when the whole answer refuses: every sentence is a refusal."""
    sentences = _sentences(answer)
    return bool(sentences) and all(_is_refusal(sentence) for sentence in sentences)


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if "faithfulness" in row]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    answerable = [row for row in rows if row["type"] != "unanswerable"]
    return {
        "faithfulness": _average([row["faithfulness"]["score"] for row in judged]),
        # Did it refuse when the corpus cannot answer?
        "abstention_accuracy": _average(
            [1.0 if row.get("abstained") else 0.0 for row in unanswerable]
        ),
        # Did it refuse when the corpus can answer? This is the cost of the
        # refusal policy, and abstention_accuracy alone never shows it.
        "false_abstention_rate": _average(
            [1.0 if row.get("abstained") else 0.0 for row in answerable]
        ),
        "partial_answers": sum(1 for row in answerable if row.get("partial")),
        "judged_answers": len(judged),
    }
