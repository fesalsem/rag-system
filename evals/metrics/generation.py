"""Aggregation for judged answers and abstention.

Each metric averages only the rows it applies to, so an empty slice reports
None rather than a misleading zero.
"""

from __future__ import annotations

from typing import Any, Sequence

REFUSAL_PHRASES = (
    # The exact sentence the QA prompt asks for, plus the paraphrase the model
    # actually produces when it restates the instruction before refusing.
    "could not find information",
    "do not contain any information",
)


def is_abstention(answer: str | None) -> bool:
    text = (answer or "").lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if "faithfulness" in row]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    return {
        "faithfulness": _average([row["faithfulness"]["score"] for row in judged]),
        "abstention_accuracy": _average(
            [1.0 if row.get("abstained") else 0.0 for row in unanswerable]
        ),
        "judged_answers": len(judged),
    }
