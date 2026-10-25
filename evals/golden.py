"""Schema and validation for the hand-authored golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

QUESTION_TYPES = ("fact", "table", "multihop", "unanswerable")
MIN_COUNTS = {"fact": 30, "table": 10, "multihop": 10, "unanswerable": 10}
REQUIRED_FIELDS = (
    "id",
    "type",
    "question",
    "reference_answer",
    "doc",
    "gold_pages",
)


class GoldenError(RuntimeError):
    """The golden set could not be read."""


def load_golden(path: str | Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GoldenError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return items


def validate(
    items: Iterable[dict[str, Any]],
    doc_ids: set[str] | None = None,
    strict: bool = False,
    min_counts: dict[str, int] | None = None,
) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    counts = {name: 0 for name in QUESTION_TYPES}

    for index, item in enumerate(items, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            problems.append(f"row {index}: missing fields {missing}")
            continue
        item_id = str(item["id"])
        if item_id in seen:
            problems.append(f"row {index}: duplicate id {item_id!r}")
        seen.add(item_id)

        question_type = item["type"]
        if question_type not in QUESTION_TYPES:
            problems.append(f"row {index}: unknown type {question_type!r}")
            continue
        counts[question_type] += 1

        if not str(item["question"]).strip():
            problems.append(f"row {index}: empty question")

        pages = item["gold_pages"]
        if not isinstance(pages, list) or any(
            not isinstance(page, int) or page < 1 for page in pages
        ):
            problems.append(f"row {index}: gold_pages must be positive integers")
        elif question_type == "unanswerable" and pages:
            problems.append(f"row {index}: unanswerable rows must have empty gold_pages")
        elif question_type != "unanswerable" and not pages:
            problems.append(f"row {index}: answerable rows need at least one gold page")

        if doc_ids is not None and item["doc"] not in doc_ids and item["doc"] != "any":
            problems.append(f"row {index}: unknown document {item['doc']!r}")

    if strict:
        expected = min_counts or MIN_COUNTS
        for question_type, minimum in expected.items():
            if counts.get(question_type, 0) < minimum:
                problems.append(
                    f"strict mode: {question_type} has {counts.get(question_type, 0)} "
                    f"rows, needs at least {minimum}"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(GOLDEN_PATH))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    problems = validate(load_golden(args.path), strict=args.strict)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
