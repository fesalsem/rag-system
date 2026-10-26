"""Run the retrieval evaluation and write a JSON report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from config import RAGConfig
from document_processor import DocumentProcessor
from evals.fetch_corpus import ensure_corpus
from evals.golden import load_golden, validate
from evals.judge import DeepSeekJudge, JudgeConfig
from evals.metrics.generation import aggregate, is_abstention
from evals.metrics.retrieval import (
    hit_at_k,
    page_of,
    reciprocal_rank,
    summarize,
)
from rag_engine import RAGEngine

LOGGER = logging.getLogger("evals.run_eval")
K_VALUES = (4, 10)
RETRY_DELAYS = (5, 15, 30, 60, 60)


def _is_rate_limit(exc: Exception) -> bool:
    return (
        getattr(exc, "status_code", None) == 429
        or "ratelimit" in type(exc).__name__.lower()
    )


def _is_daily_cap(exc: Exception) -> bool:
    return _is_rate_limit(exc) and "per day" in str(exc).lower()


def with_retry(
    call: Callable[[], Any],
    delays: Sequence[float] = RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Run a provider call, backing off while it is rate limited.

    Groq's free tier meters tokens per minute, so a long answer can trip 429
    partway through a run that started clean. A per-day cap is different: it
    cannot clear inside a minute, so the run stops and the next one resumes
    from the checkpoint. Failures other than 429 are not a timing problem and
    are raised immediately.
    """
    for delay in delays:
        try:
            return call()
        except Exception as exc:
            if not _is_rate_limit(exc) or _is_daily_cap(exc):
                raise
            LOGGER.warning("Rate limited (%s); retrying in %ss.", exc, delay)
            sleep(delay)
    return call()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_engine(
    corpus_paths: dict[str, Path],
    hybrid: bool,
    index_dir: str | Path,
    reuse_index: bool,
) -> RAGEngine:
    config = RAGConfig()
    config.vector_store.persist = True
    config.vector_store.index_path = Path(index_dir)
    config.hybrid.enabled = hybrid
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


def _read_checkpoint(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def run_answers(
    golden: Sequence[dict[str, Any]],
    engine: RAGEngine,
    judge: DeepSeekJudge,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Answer every question, appending each row to a checkpoint as it lands.

    A free-tier Groq key can be capped mid-run, and a completed answer costs
    real tokens to reproduce. The checkpoint lets the next run continue where
    this one stopped. Delete it when retrieval settings change, or the report
    mixes answers from two different configurations.
    """
    done = _read_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    for item in golden:
        if item["id"] in done:
            rows.append(done[item["id"]])
            continue
        result = with_retry(lambda: engine.query(item["question"]))
        answer = result["answer"]
        abstained = is_abstention(answer)
        row: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "answer": answer,
            "sources": result["sources"],
            "abstained": abstained,
        }
        # A refusal makes no factual claims, so judging it as unfaithful would
        # punish the behaviour the prompt asks for.
        if item["type"] != "unanswerable" and not abstained:
            row["faithfulness"] = with_retry(
                lambda: judge.score_faithfulness(
                    item["question"],
                    answer,
                    "\n\n".join(result["contexts"]),
                )
            )
        rows.append(row)
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
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
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--answers-cache",
        default="evals/.answers.jsonl",
        help="Checkpoint of answered questions, so a capped run can resume. "
        "Delete it when retrieval settings change.",
    )
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

    engine = build_engine(
        corpus,
        hybrid=args.hybrid,
        index_dir=args.index_dir,
        reuse_index=args.reuse_index,
    )
    rows = run_retrieval(golden, engine)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hybrid": args.hybrid,
        "corpus": {doc_id: path.name for doc_id, path in corpus.items()},
        "summary": summarise(rows),
        "questions": rows,
    }
    if args.full:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set; --full requires it.", file=sys.stderr)
            return 2
        judge = DeepSeekJudge(
            JudgeConfig(api_key=api_key), cache_dir="evals/.judge_cache"
        )
        answers = run_answers(
            golden, engine, judge, checkpoint=Path(args.answers_cache)
        )
        report["generation"] = aggregate(answers)
        report["answers"] = answers
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
