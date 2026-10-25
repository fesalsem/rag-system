"""Deterministic retrieval metrics.

Ground truth is page-level. That is coarser than chunk-level, but it is the
label a human can produce from a PDF without annotating every split.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

_PAGE_RE = re.compile(r"\d+")


def page_of(document: Any) -> int | None:
    """Return the 1-based page number, or None when the chunk has no page."""
    metadata = getattr(document, "metadata", {}) or {}
    label = metadata.get("page_label")
    if label:
        match = _PAGE_RE.search(str(label))
        if match:
            return int(match.group())
    raw = metadata.get("page")
    try:
        return int(raw) + 1
    except (TypeError, ValueError):
        return None


def hit_at_k(
    documents: Sequence[Any], gold_pages: Iterable[int], k: int
) -> bool:
    gold = set(gold_pages)
    return any(page_of(document) in gold for document in documents[:k])


def reciprocal_rank(
    documents: Sequence[Any], gold_pages: Iterable[int]
) -> float:
    gold = set(gold_pages)
    for rank, document in enumerate(documents, start=1):
        if page_of(document) in gold:
            return 1.0 / rank
    return 0.0


def summarize(
    rows: Sequence[dict[str, Any]], k_values: Sequence[int] = (4, 10)
) -> dict[str, float | int]:
    total = len(rows)
    denominator = total or 1
    summary: dict[str, float | int] = {
        f"recall@{k}": round(
            sum(1 for row in rows if row.get(f"hit@{k}")) / denominator, 4
        )
        for k in k_values
    }
    summary["mrr"] = round(
        sum(float(row.get("reciprocal_rank", 0.0)) for row in rows) / denominator, 4
    )
    summary["n"] = total
    return summary
