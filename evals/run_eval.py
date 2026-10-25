"""Run the retrieval evaluation and write a JSON report."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import RAGConfig
from document_processor import DocumentProcessor
from evals.fetch_corpus import ensure_corpus
from evals.golden import load_golden, validate
from evals.metrics.retrieval import (
    hit_at_k,
    page_of,
    reciprocal_rank,
    summarize,
)
from rag_engine import RAGEngine

LOGGER = logging.getLogger("evals.run_eval")
K_VALUES = (4, 10)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_engine(
    corpus_paths: dict[str, Path],
    index_dir: str | Path,
    reuse_index: bool,
) -> RAGEngine:
    config = RAGConfig()
    config.vector_store.persist = True
    config.vector_store.index_path = Path(index_dir)
    engine = RAGEngine(config=config)
    if reuse_index and engine.index_exists:
        LOGGER.info("Reusing index at %s", index_dir)
        engine.load_existing_index()
        return engine

    processor = DocumentProcessor(config=config)
    documents = []
    for path in corpus_paths.values():
        documents.extend(processor.load_and_split(path))
    engine.clear_documents()
    engine.add_documents(documents)
    return engine


def run_retrieval(
    golden: Sequence[dict[str, Any]], engine: RAGEngine
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in golden:
        started = time.perf_counter()
        documents = engine.retrieve(item["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        gold_pages = item["gold_pages"]
        row: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "gold_pages": gold_pages,
            "retrieved_pages": [page_of(document) for document in documents],
            "reciprocal_rank": round(reciprocal_rank(documents, gold_pages), 4),
            "latency_ms": round(latency_ms, 2),
        }
        for k in K_VALUES:
            row[f"hit@{k}"] = hit_at_k(documents, gold_pages, k)
        rows.append(row)
    return rows


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["type"] != "unanswerable"]
    summary: dict[str, Any] = summarize(scored, K_VALUES)
    latencies = [float(row["latency_ms"]) for row in rows]
    summary.update(
        {
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "unanswerable": sum(
                1 for row in rows if row["type"] == "unanswerable"
            ),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="evals/golden.jsonl")
    parser.add_argument("--out", default="evals/results/current.json")
    parser.add_argument("--index-dir", default="evals/.index")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    corpus = ensure_corpus()
    golden = load_golden(args.golden)
    problems = validate(golden, doc_ids=set(corpus))
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    if args.limit:
        golden = golden[: args.limit]

    engine = build_engine(corpus, args.index_dir, args.reuse_index)
    rows = run_retrieval(golden, engine)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hybrid": False,
        "corpus": {doc_id: path.name for doc_id, path in corpus.items()},
        "summary": summarise(rows),
        "questions": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
