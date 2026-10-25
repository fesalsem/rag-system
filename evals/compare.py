"""Fail when an evaluation metric drops past its allowed amount."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    thresholds: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    baseline_summary = baseline.get("summary", {})
    current_summary = current.get("summary", {})
    for metric, allowed_drop in thresholds.items():
        before = baseline_summary.get(metric)
        after = current_summary.get(metric)
        if before is None or after is None:
            continue
        if after - before < allowed_drop:
            failures.append(
                f"{metric}: {before:.3f} -> {after:.3f} "
                f"(allowed drop {allowed_drop:+.3f})"
            )
    return failures


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--thresholds", required=True)
    args = parser.parse_args(argv)

    failures = compare(
        _load_json(args.baseline),
        _load_json(args.current),
        _load_json(args.thresholds),
    )
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
