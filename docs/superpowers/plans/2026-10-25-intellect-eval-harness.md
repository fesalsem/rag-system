# Intellect Evaluation Harness and Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Intellect reproducible retrieval and answer-quality numbers, then improve retrieval with hybrid search and reranking, with a CI gate that blocks regressions.

**Architecture:** A dev-only `evals/` package pins a public corpus, scores retrieval deterministically, and scores answer faithfulness with a DeepSeek judge. Hybrid retrieval lives in a new root `retrieval.py`, called from `RAGEngine` only when `RAG_HYBRID_ENABLED=1`. The flag is the rollback and keeps the existing tests untouched.

**Tech Stack:** Python 3.12, LangChain 0.3.25, FAISS, rank-bm25, sentence-transformers CrossEncoder, DeepSeek API over stdlib `urllib`, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-25-intellect-eval-harness-design.md`

## Global Constraints

- Python 3.12 is the target; CI also runs 3.11.
- Every runtime dependency in `requirements.txt` stays pinned with `==`.
- `tests/` must run offline: no network, no API key, no model download.
- `app.py` must never import anything from `evals/`.
- `RAG_HYBRID_ENABLED` defaults to `0`. With it off, behaviour is byte-for-byte the current path.
- The existing suite must stay green after every task.
- Commit after every task. Plain commit messages, no co-author trailer.
- Do not put an em dash in new prose or comments. Existing string labels that use one are left alone.

---

## Task 0: Correct the resume claims this work supersedes

This is not repo code, but do it before you add numbers, so the resume is
truthful while the work is in progress.

**Files:**
- Modify: `../job-search/master-resume.md`
- Modify: `../job-search/resumes/*.html`
- Modify: `../portfolio/index.html` if it repeats the claims

- [x] **Step 1: Verify the real test count (done 2026-10-25)**

CI run `37314920699` collected and passed **169 tests** on all three legs and
inside the Docker image. The resume says 41.

Local collection cannot reproduce this without the project dependencies
installed, so use the CI number.

- [ ] **Step 2: Fix the CI matrix and test-count claim**

The workflow has three legs, not six, and no Python 3.10. It also runs 169
tests, not 41.

Replace:

```
41 pytest tests across a 6-environment matrix (Python 3.10–3.12 × Ubuntu/Windows)
```

with:

```
169 pytest tests across a 3-leg CI matrix (Python 3.11 and 3.12 on Ubuntu, 3.12 on Windows)
```

- [ ] **Step 2b: Fix or drop the Codecov claim**

Every CI run logs `Token required - not valid tokenless upload`, so coverage
is not reaching Codecov. Pick one:

1. Add a `CODECOV_TOKEN` repository secret from codecov.io. Then keep
   "with coverage uploaded to Codecov".
2. Leave it off the resume until the upload succeeds.

Do not keep the claim while the upload is failing.

- [ ] **Step 3: Fix the provider-swap claim**

The README says changing the LLM provider or vector store means editing
`rag_engine.py`. Replace "swaps the LLM, embedding model, or vector store
without touching pipeline code" with:

```
a typed Pydantic configuration layer that makes the LLM model, embedding model,
and chunking environment-swappable, with provider and vector-store changes
contained to one module
```

- [ ] **Step 4: Fix the persistence claim**

`app.py:417` forces persistence off, so the live app does not keep an index
between sessions. Replace the persistence bullet with:

```
Added optional FAISS index persistence for single-user runs; the shared web
deployment keeps per-session indexes so one visitor's documents never outlive
the session or reach another visitor
```

- [ ] **Step 5: Rebuild the PDFs**

```bash
cd ../job-search && ./build-pdfs.sh
```

Expected: all five `Faisal_*.pdf` files rebuild and pass their heading checks.

---

## Task 1: Corpus fetch and hash lock

**Files:**
- Create: `evals/__init__.py`
- Create: `evals/corpus/manifest.json`
- Create: `evals/fetch_corpus.py`
- Create: `tests/evals/__init__.py`
- Create: `tests/evals/test_fetch_corpus.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `evals.fetch_corpus.ensure_corpus(update_lock: bool = False) -> dict[str, Path]`, keyed by document id.
- Produces: `evals.fetch_corpus.CorpusError`.

- [ ] **Step 1: Create the package and the manifest**

`evals/__init__.py` is empty.

`evals/corpus/manifest.json`:

```json
{
  "version": 1,
  "documents": [
    {
      "id": "attention",
      "file": "attention-is-all-you-need.pdf",
      "url": "https://arxiv.org/pdf/1706.03762",
      "title": "Attention Is All You Need"
    },
    {
      "id": "rag-paper",
      "file": "rag-knowledge-intensive-nlp.pdf",
      "url": "https://arxiv.org/pdf/2005.11401",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    },
    {
      "id": "berkshire-2023",
      "file": "berkshire-2023-ar.pdf",
      "url": "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
      "title": "Berkshire Hathaway 2023 Annual Report"
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/evals/__init__.py` is empty.

`tests/evals/test_fetch_corpus.py`:

```python
import hashlib
import json

import pytest

from evals.fetch_corpus import CorpusError, ensure_corpus


def make_corpus(tmp_path, payload=b"%PDF-1.4 test bytes"):
    source = tmp_path / "source.pdf"
    source.write_bytes(payload)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = {
        "documents": [
            {"id": "doc-a", "file": "doc-a.pdf", "url": source.as_uri(), "title": "A"}
        ]
    }
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus, payload


def test_requires_a_committed_lock_on_first_use(tmp_path):
    corpus, _ = make_corpus(tmp_path)
    with pytest.raises(CorpusError):
        ensure_corpus(corpus_dir=corpus)


def test_update_lock_downloads_and_records_sha256(tmp_path):
    corpus, payload = make_corpus(tmp_path)
    paths = ensure_corpus(update_lock=True, corpus_dir=corpus)
    assert paths["doc-a"].read_bytes() == payload
    lock = json.loads((corpus / "lock.json").read_text(encoding="utf-8"))
    assert lock["sha256"]["doc-a"] == hashlib.sha256(payload).hexdigest()


def test_tampered_download_is_rejected(tmp_path):
    corpus, payload = make_corpus(tmp_path)
    ensure_corpus(update_lock=True, corpus_dir=corpus)
    (corpus / "doc-a.pdf").write_bytes(payload + b"tampered")
    with pytest.raises(CorpusError):
        ensure_corpus(corpus_dir=corpus)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/evals/test_fetch_corpus.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'evals.fetch_corpus'`.

- [ ] **Step 4: Write the implementation**

`evals/fetch_corpus.py`:

```python
"""Download and verify the evaluation corpus.

The PDFs are not committed. A manifest names each URL; a lock file records the
SHA-256 of the bytes that were first downloaded. Every later run verifies the
same hashes, so scores always refer to the same documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
LOCK_PATH = CORPUS_DIR / "lock.json"
USER_AGENT = "intellect-eval/1.0 (+https://github.com/fesalsem/rag-system)"


class CorpusError(RuntimeError):
    """The corpus is missing, unverified, or does not match its lock file."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise CorpusError(f"{path} has no documents")
    return documents


def load_lock(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sha256", {})


def save_lock(hashes: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps({"sha256": hashes}, indent=2) + "\n", encoding="utf-8")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        partial.write_bytes(response.read())
    partial.replace(destination)


def ensure_corpus(
    update_lock: bool = False, corpus_dir: Path = CORPUS_DIR
) -> dict[str, Path]:
    manifest = load_manifest(corpus_dir / "manifest.json")
    lock = load_lock(corpus_dir / "lock.json")
    if not lock and not update_lock:
        raise CorpusError(
            "No lock file. Run `python -m evals.fetch_corpus --update-lock` "
            "and commit evals/corpus/lock.json."
        )

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for document in manifest:
        doc_id = document["id"]
        destination = corpus_dir / document["file"]
        if not destination.exists():
            download(document["url"], destination)
        actual = sha256_file(destination)
        expected = lock.get(doc_id)
        if expected is None and not update_lock:
            raise CorpusError(
                f"{destination.name} is not in the lock file. Run "
                "`python -m evals.fetch_corpus --update-lock` and commit the result."
            )
        if expected is not None and actual != expected and not update_lock:
            raise CorpusError(
                f"{destination.name} does not match the lock file. Delete it and "
                "rerun, or pass --update-lock if the corpus is meant to change."
            )
        hashes[doc_id] = actual
        paths[doc_id] = destination

    if update_lock or not lock:
        save_lock(hashes, corpus_dir / "lock.json")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = ensure_corpus(update_lock=args.update_lock)
    except CorpusError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for doc_id, path in paths.items():
        print(f"{doc_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Ignore the downloaded PDFs and local caches**

Append to `.gitignore`:

```
# Evaluation corpus (downloaded, verified by lock.json) and local caches
evals/corpus/*.pdf
evals/.index/
evals/.judge_cache/
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/evals/test_fetch_corpus.py -v`

Expected: 3 passed.

- [ ] **Step 7: Download the corpus and commit the lock**

Run:

```bash
python -m evals.fetch_corpus --update-lock
```

Expected: three lines, one per document. Then:

```bash
git add evals tests/evals .gitignore
git commit -m "Add hash-pinned evaluation corpus fetcher"
```

---

## Task 2: Golden set schema and validator

**Files:**
- Create: `evals/golden.py`
- Create: `evals/golden.jsonl`
- Create: `tests/evals/test_golden.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `evals.golden.load_golden(path: str | Path) -> list[dict]`.
- Produces: `evals.golden.validate(items, doc_ids=None, strict=False, min_counts=None) -> list[str]`.
- Produces: `evals.golden.GoldenError`.

- [ ] **Step 1: Write the failing test**

`tests/evals/test_golden.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/evals/test_golden.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`evals/golden.py`:

```python
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
```

- [ ] **Step 4: Author the golden set**

`evals/golden.jsonl` is your work, not mine. Write one JSON object per line.
The three rows below show the shape. Author the remaining 57 so strict mode
passes: 30 `fact`, 10 `table`, 10 `multihop`, 10 `unanswerable`.

```jsonl
{"id":"fact-001","type":"fact","question":"How many attention heads does the base Transformer use?","reference_answer":"Eight.","doc":"attention","gold_pages":[5]}
{"id":"table-001","type":"table","question":"What was Berkshire's consolidated cash and cash equivalents at the end of 2023, in billions?","reference_answer":"The 2023 consolidated balance sheet reports the figure in the cash section.","doc":"berkshire-2023","gold_pages":[4]}
{"id":"unans-001","type":"unanswerable","question":"What was Intellect's retrieval recall@4 before hybrid search was added?","reference_answer":"The corpus does not contain this information.","doc":"any","gold_pages":[]}
```

Rules for authoring:

1. Read the page first, then write the question with the page closed.
2. Use page numbers as printed in the PDF, not the PDF viewer index.
3. A `table` question must need a number from a table, not prose.
4. A `multihop` question must need two passages, and `doc` may be `"any"`.
5. An `unanswerable` question must be plausible but absent from the corpus.

Validate as you go:

```bash
python -m evals.golden --strict
```

Expected: no output, exit 0.

- [ ] **Step 5: Run the tests and commit**

Run: `python -m pytest tests/evals/test_golden.py -v`

Expected: 7 passed.

```bash
git add evals/golden.py evals/golden.jsonl tests/evals/test_golden.py
git commit -m "Add golden set schema and 60-question evaluation set"
```

---

## Task 3: Retrieval metrics

**Files:**
- Create: `evals/metrics/__init__.py`
- Create: `evals/metrics/retrieval.py`
- Create: `tests/evals/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `page_of(doc) -> int | None`
- Produces: `hit_at_k(documents, gold_pages, k) -> bool`
- Produces: `reciprocal_rank(documents, gold_pages) -> float`
- Produces: `summarize(rows, k_values=(4, 10)) -> dict`

- [ ] **Step 1: Write the failing test**

`tests/evals/test_retrieval_metrics.py`:

```python
from langchain_core.documents import Document

from evals.metrics.retrieval import (
    hit_at_k,
    page_of,
    reciprocal_rank,
    summarize,
)


def doc(page):
    return Document(page_content="x", metadata={"page_label": f"Page {page}"})


def test_page_of_prefers_page_label():
    labelled = Document(page_content="x", metadata={"page_label": "Page 7", "page": 2})
    assert page_of(labelled) == 7
    assert page_of(Document(page_content="x", metadata={"page": 2})) == 3
    assert page_of(Document(page_content="x", metadata={})) is None


def test_hit_at_k_respects_the_cutoff():
    retrieved = [doc(1), doc(2)]
    assert hit_at_k(retrieved, {2}, k=2) is True
    assert hit_at_k(retrieved, {2}, k=1) is False


def test_reciprocal_rank_uses_the_first_correct_page():
    assert reciprocal_rank([doc(1), doc(5)], {5}) == 0.5
    assert reciprocal_rank([doc(1)], {9}) == 0.0


def test_summarize_averages_the_rows():
    rows = [
        {"hit@4": True, "hit@10": True, "reciprocal_rank": 1.0},
        {"hit@4": False, "hit@10": True, "reciprocal_rank": 0.5},
    ]
    assert summarize(rows, (4, 10)) == {
        "recall@4": 0.5,
        "recall@10": 1.0,
        "mrr": 0.75,
        "n": 2,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/evals/test_retrieval_metrics.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`evals/metrics/__init__.py` is empty.

`evals/metrics/retrieval.py`:

```python
"""Deterministic retrieval metrics.

Ground truth is page-level. That is coarser than chunk-level, but it is the
label a human can produce from a PDF without annotating every split.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

_PAGE_RE = re.compile(r"(\d+)")


def page_of(document: Any) -> int | None:
    """Return the 1-based page number, or None when the chunk has no page."""
    metadata = getattr(document, "metadata", {}) or {}
    label = metadata.get("page_label")
    if label:
        match = _PAGE_RE.search(str(label))
        if match:
            return int(match.group(1))
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
```

- [ ] **Step 4: Run the tests and commit**

Run: `python -m pytest tests/evals/test_retrieval_metrics.py -v`

Expected: 4 passed.

```bash
git add evals/metrics tests/evals/test_retrieval_metrics.py
git commit -m "Add deterministic retrieval metrics"
```

---

## Task 4: Retrieval evaluation runner and baseline

**Files:**
- Create: `evals/run_eval.py`
- Create: `tests/evals/test_run_eval.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `ensure_corpus`, `load_golden`, `validate`, `hit_at_k`, `reciprocal_rank`, `summarize`.
- Produces: `run_retrieval(golden, engine) -> list[dict]`
- Produces: `summarise(rows) -> dict`
- Produces: CLI `python -m evals.run_eval [--hybrid] [--reuse-index] [--out PATH]`.

- [ ] **Step 1: Write the failing test**

`tests/evals/test_run_eval.py`:

```python
from types import SimpleNamespace

from evals.run_eval import run_retrieval, summarise


def doc(page):
    return SimpleNamespace(metadata={"page_label": f"Page {page}"})


class FakeEngine:
    def __init__(self, pages):
        self.pages = pages
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
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
        {"type": "fact", "hit@4": True, "hit@10": True, "reciprocal_rank": 1.0, "latency_ms": 10.0},
        {"type": "unanswerable", "hit@4": False, "hit@10": False, "reciprocal_rank": 0.0, "latency_ms": 20.0},
    ]
    summary = summarise(rows)
    assert summary["recall@4"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["unanswerable"] == 1
    assert summary["p95_ms"] == 20.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/evals/test_run_eval.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`evals/run_eval.py`:

```python
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
        "hybrid": args.hybrid,
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
```

- [ ] **Step 4: Keep the CLI runner out of the coverage gate**

In `pyproject.toml`, under both `[tool.coverage.run]` and
`[tool.coverage.report]`, add `"evals/run_eval.py"` to `omit`. The runner is
covered indirectly by `run_retrieval` and `summarise`; the `main()` wiring is
not worth a subprocess test.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/evals/test_run_eval.py -v`

Expected: 2 passed.

- [ ] **Step 6: Produce the baseline before any retrieval change**

This is the most important step in the plan. Do not skip ahead.

```bash
python -m evals.run_eval --out evals/results/baseline.json
```

Expected: a summary printed, for example `recall@4: 0.6`, `mrr: 0.5`. Your
numbers will differ. Record them.

```bash
git add evals/run_eval.py evals/results/baseline.json tests/evals/test_run_eval.py pyproject.toml
git commit -m "Add retrieval evaluation runner and baseline report"
```

---

## Task 5: Hybrid retriever

**Files:**
- Create: `retrieval.py`
- Modify: `config.py` (add `HybridRetrievalConfig`, wire into `RAGConfig`)
- Modify: `requirements.txt` (add `rank-bm25==0.2.2`)
- Modify: `tests/test_config.py` (add hybrid config tests)
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `RAGConfig.hybrid` and `RAGConfig.retriever_k`.
- Produces: `retrieval.HybridRetriever(documents, vector_store, config, reranker=None, bm25_factory=None, ensemble_factory=None)`.
- Produces: `HybridRetriever.retrieve(query: str) -> list[Document]`.
- Produces: `retrieval.CrossEncoderReranker.rerank(query, documents, top_k, min_score) -> list[Document]`.

- [ ] **Step 1: Add the config block**

In `config.py`, after `class MemoryConfig` and before `class RAGConfig`:

```python
class HybridRetrievalConfig(BaseModel):
    """BM25 plus dense fusion, optionally followed by a cross-encoder rerank."""

    enabled: bool = Field(
        default_factory=lambda: _env_str("RAG_HYBRID_ENABLED", "0") == "1",
        description="Use BM25 plus FAISS fusion instead of FAISS alone.",
    )
    bm25_weight: float = Field(
        default_factory=lambda: _env_float("RAG_BM25_WEIGHT", 0.4),
        ge=0.0,
        le=1.0,
        description="Weight of BM25 in the reciprocal rank fusion.",
    )
    candidate_k: int = Field(
        default_factory=lambda: _env_int("RAG_CANDIDATE_K", 20),
        gt=0,
        description="Chunks each retriever contributes before reranking.",
    )
    rerank_model: str = Field(
        default_factory=lambda: _env_str(
            "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        description="Cross-encoder used to reorder the fused candidates.",
    )
    rerank_min_score: float = Field(
        default_factory=lambda: _env_float("RAG_RERANK_MIN_SCORE", 0.0),
        description=(
            "Minimum cross-encoder logit. The model is trained so that a "
            "positive score means the passage is relevant, which is why the "
            "default gate sits at zero."
        ),
    )
```

In `RAGConfig`, next to the other nested configs:

```python
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
```

- [ ] **Step 2: Add the dependency**

In `requirements.txt`, under the vector store section:

```
# ── Hybrid retrieval ─────────────────────────────────────────────────────────
rank-bm25==0.2.2
```

- [ ] **Step 3: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_hybrid_defaults_are_off(monkeypatch):
    monkeypatch.delenv("RAG_HYBRID_ENABLED", raising=False)
    config = RAGConfig()
    assert config.hybrid.enabled is False
    assert config.hybrid.bm25_weight == 0.4
    assert config.hybrid.candidate_k == 20
    assert config.hybrid.rerank_min_score == 0.0


def test_hybrid_reads_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    monkeypatch.setenv("RAG_BM25_WEIGHT", "0.7")
    monkeypatch.setenv("RAG_CANDIDATE_K", "40")
    config = RAGConfig()
    assert config.hybrid.enabled is True
    assert config.hybrid.bm25_weight == 0.7
    assert config.hybrid.candidate_k == 40
```

`tests/test_retrieval.py`:

```python
from langchain_core.documents import Document

from config import RAGConfig
from retrieval import HybridRetriever


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.documents)


class FakeStore:
    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever([])


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k, min_score):
        self.calls.append((query, documents, top_k, min_score))
        return documents[:top_k]


def documents(count=3):
    return [
        Document(page_content=f"chunk {index}", metadata={})
        for index in range(count)
    ]


def build_retriever(reranker=None, bm25_documents=None):
    docs = documents()
    store = FakeStore()
    config = RAGConfig()
    config.retriever_k = 2
    config.hybrid.candidate_k = 20
    retriever = HybridRetriever(
        docs,
        store,
        config,
        reranker=reranker or FakeReranker(),
        bm25_factory=lambda documents, k: FakeRetriever(bm25_documents or documents),
        ensemble_factory=lambda retrievers, weights: FakeRetriever(docs),
    )
    return retriever, store, config


def test_fusion_uses_the_configured_candidate_pool():
    retriever, store, _ = build_retriever()
    assert store.search_kwargs == {"k": 20}
    assert retriever.retrieve("question") == documents()[:2]


def test_reranker_receives_the_fused_candidates_and_final_k():
    reranker = FakeReranker()
    retriever, _, config = build_retriever(reranker)
    retriever.retrieve("question")
    query, candidates, top_k, min_score = reranker.calls[0]
    assert query == "question"
    assert len(candidates) == 3
    assert top_k == config.retriever_k
    assert min_score == config.hybrid.rerank_min_score


def test_empty_documents_are_rejected():
    try:
        HybridRetriever([], FakeStore(), RAGConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_retrieval.py tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval'`.

- [ ] **Step 5: Write the implementation**

`retrieval.py`:

```python
"""Hybrid retrieval: BM25 plus dense fusion, then cross-encoder reranking.

Imports for the optional pieces are delayed until construction so that the
Streamlit app starts without rank-bm25 or a downloaded cross-encoder when
RAG_HYBRID_ENABLED is off.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional, Protocol, Sequence

from langchain_core.documents import Document

from config import RAGConfig

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int, min_score: float
    ) -> List[Document]: ...


def _load_cross_encoder(model_name: str) -> Any:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder '%s' …", model_name)
            _MODEL_CACHE[model_name] = CrossEncoder(model_name)
        return _MODEL_CACHE[model_name]


class CrossEncoderReranker:
    """Orders candidate passages by relevance to the query."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int, min_score: float
    ) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
        scores = _load_cross_encoder(self.model_name).predict(pairs)
        ranked = sorted(
            zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        return [
            document for document, score in ranked if float(score) >= min_score
        ][:top_k]


def _default_bm25(documents: Sequence[Document], k: int) -> Any:
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(list(documents), k=k)


def _default_ensemble(retrievers: Sequence[Any], weights: Sequence[float]) -> Any:
    from langchain.retrievers import EnsembleRetriever

    return EnsembleRetriever(retrievers=list(retrievers), weights=list(weights))


class HybridRetriever:
    """BM25 plus dense retrieval fused with reciprocal rank fusion."""

    def __init__(
        self,
        documents: Sequence[Document],
        vector_store: Any,
        config: RAGConfig,
        reranker: Optional[Reranker] = None,
        bm25_factory: Optional[Callable[[Sequence[Document], int], Any]] = None,
        ensemble_factory: Optional[
            Callable[[Sequence[Any], Sequence[float]], Any]
        ] = None,
    ) -> None:
        if not documents:
            raise ValueError("Hybrid retrieval needs at least one chunk.")
        if vector_store is None:
            raise ValueError("Hybrid retrieval needs a vector store.")
        self.config = config
        self._documents = list(documents)
        self._vector_store = vector_store
        self._reranker = reranker or CrossEncoderReranker(config.hybrid.rerank_model)
        self._bm25_factory = bm25_factory or _default_bm25
        self._ensemble_factory = ensemble_factory or _default_ensemble
        self._ensemble = self._build()

    def _build(self) -> Any:
        hybrid = self.config.hybrid
        bm25 = self._bm25_factory(self._documents, hybrid.candidate_k)
        dense = self._vector_store.as_retriever(
            search_kwargs={"k": hybrid.candidate_k}
        )
        return self._ensemble_factory(
            [bm25, dense], [hybrid.bm25_weight, 1.0 - hybrid.bm25_weight]
        )

    def retrieve(self, query: str) -> List[Document]:
        candidates = self._ensemble.invoke(query)
        return self._reranker.rerank(
            query,
            candidates,
            top_k=self.config.retriever_k,
            min_score=self.config.hybrid.rerank_min_score,
        )
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_retrieval.py tests/test_config.py -v`

Expected: all pass, including the existing config tests.

- [ ] **Step 7: Add the hybrid flag to the runner**

In `evals/run_eval.py`, restore the parameter and set it on the config:

```python
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
```

Add the flag and pass it through:

```python
    parser.add_argument("--hybrid", action="store_true")
```

```python
    engine = build_engine(corpus, args.hybrid, args.index_dir, args.reuse_index)
```

- [ ] **Step 8: Commit**

```bash
git add retrieval.py config.py requirements.txt tests/test_retrieval.py tests/test_config.py
git commit -m "Add hybrid BM25 and dense retriever with cross-encoder reranking"
```

---

## Task 6: Wire hybrid retrieval into RAGEngine

**Files:**
- Modify: `rag_engine.py`
- Modify: `tests/test_rag_engine.py`
- Create: `tests/test_hybrid_engine.py`

**Interfaces:**
- Consumes: `HybridRetriever`, `config.hybrid.enabled`.
- Produces: `RAGEngine.retrieve(question: str) -> List[Document]`.
- Produces: `query()` results gain `contexts: list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid_engine.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from config import RAGConfig, VectorStoreConfig
from rag_engine import RAGEngine


class FakeLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="answer")


class FakeStore:
    def __init__(self, scored=None):
        self.scored = list(scored or [])
        self.index = SimpleNamespace(ntotal=len(self.scored))

    def similarity_search_with_score(self, query, k=None):
        return list(self.scored)


class FakeHybrid:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.documents)


def make_doc(content="chunk", page=1):
    return Document(
        page_content=content,
        metadata={"file_name": "doc.pdf", "page_label": f"Page {page}"},
    )


def make_engine(tmp_path, store=None, documents=None):
    config = RAGConfig()
    config.vector_store = VectorStoreConfig(index_path=tmp_path / "idx")
    with patch("rag_engine.HuggingFaceEmbeddings", return_value=MagicMock()), patch(
        "rag_engine.ChatGroq", return_value=FakeLLM()
    ):
        engine = RAGEngine(config=config)
        engine._models.embeddings()
        engine._models.llm()
    engine._vector_store = store
    engine._docs = list(documents or [])
    return engine


def test_hybrid_is_used_when_enabled(tmp_path, monkeypatch):
    document = make_doc()
    engine = make_engine(tmp_path, store=FakeStore(), documents=[document])
    fake = FakeHybrid([document])
    monkeypatch.setattr("rag_engine.HybridRetriever", lambda *args, **kwargs: fake)
    engine.config.hybrid.enabled = True

    assert engine.retrieve("question") == [document]
    assert fake.queries == ["question"]


def test_dense_path_unchanged_when_hybrid_is_off(tmp_path):
    near = (make_doc("near"), 0.5)
    far = (make_doc("far"), 2.0)
    engine = make_engine(tmp_path, store=FakeStore([near, far]))

    assert engine.retrieve("question") == [near[0]]


def test_query_returns_the_retrieved_contexts(tmp_path):
    document = make_doc("the context")
    engine = make_engine(tmp_path, store=FakeStore([(document, 0.5)]))

    result = engine.query("What is in the document?")

    assert result["contexts"] == ["the context"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_hybrid_engine.py -v`

Expected: FAIL with `AttributeError: 'RAGEngine' object has no attribute 'retrieve'`.

- [ ] **Step 3: Add the import and the instance field**

At the top of `rag_engine.py`, next to the other local imports:

```python
from retrieval import HybridRetriever
```

In `RAGEngine.__init__`, after `self._vector_store`:

```python
        self._hybrid: Optional[HybridRetriever] = None
```

- [ ] **Step 4: Add the retriever factory and the public retrieve method**

Insert before `query`:

```python
    def _get_hybrid(self) -> HybridRetriever:
        """Build the hybrid retriever once per index change."""
        if self._hybrid is None:
            documents = self._get_all_documents()
            if not documents:
                raise RuntimeError(
                    "Cannot build hybrid retrieval without indexed documents."
                )
            self._hybrid = HybridRetriever(
                documents, self._vector_store, self.config
            )
        return self._hybrid

    def retrieve(self, question: str) -> List[Document]:
        """
        Return the chunks that answer ``question``.

        Hybrid retrieval applies its own relevance gate, so an empty list means
        no passage cleared it. The dense path keeps the existing distance
        ceiling, which is what preserves current behaviour when hybrid is off.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "The RAG chain is not ready. "
                "Load or create an index before querying."
            )
        search_query = self._condense(question)
        try:
            if self.config.hybrid.enabled:
                return self._get_hybrid().retrieve(search_query)
            scored = self._vector_store.similarity_search_with_score(
                search_query, k=self.config.retriever_k
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            raise
        return [
            doc
            for doc, distance in scored
            if distance <= self.config.max_distance
        ]
```

- [ ] **Step 5: Replace the retrieval block in query**

In `query`, replace everything from `search_query = self._condense(question)`
through the `if not relevant:` fallback's preceding `relevant = [...]`
comprehension with:

```python
        relevant = self.retrieve(question)
```

The surrounding fallback stays as it is:

```python
        if not relevant:
            logger.info(
                "No chunk scored above the relevance gate; falling back to an overview."
            )
            result = self._summarize_document(question)
            self._remember(question, result["answer"])
            return result
```

- [ ] **Step 6: Add contexts to both result shapes**

In `_summarize_document`, the empty-index return becomes:

```python
        if not docs:
            return {
                "answer": "No documents are indexed yet.",
                "sources": [],
                "contexts": [],
            }
```

The successful return becomes:

```python
        return {
            "answer": answer,
            "sources": sources,
            "contexts": [doc.page_content for doc in sampled],
            "summary": True,
        }
```

The failure return becomes:

```python
            return {
                "answer": "I could not generate an overview right now. "
                          "Please try a more specific question.",
                "sources": [],
                "contexts": [],
            }
```

In `query`, the answer return becomes:

```python
        return {
            "answer": answer,
            "sources": sources,
            "contexts": [doc.page_content for doc in relevant],
        }
```

- [ ] **Step 7: Reset the hybrid index when documents change**

In `add_documents`, inside the lock after `self._docs.extend(documents)`, add:

```python
            self._hybrid = None
```

In `clear_documents`, inside the lock, add:

```python
            self._hybrid = None
```

- [ ] **Step 8: Run the new tests and the existing suite**

Run:

```bash
python -m pytest tests/test_hybrid_engine.py tests/test_rag_engine.py -v
```

Expected: all pass. If the existing suite has a test that asserted the exact
`query()` return dict, update it to include `contexts`.

- [ ] **Step 9: Commit**

```bash
git add rag_engine.py tests/test_rag_engine.py tests/test_hybrid_engine.py
git commit -m "Route RAGEngine retrieval through hybrid search behind a flag"
```

---

## Task 7: DeepSeek faithfulness judge

**Files:**
- Create: `evals/judge.py`
- Create: `evals/metrics/generation.py`
- Create: `tests/evals/test_judge.py`
- Create: `tests/evals/test_generation_metrics.py`

**Interfaces:**
- Produces: `DeepSeekJudge(config: JudgeConfig, cache_dir=None, transport=None)`.
- Produces: `DeepSeekJudge.score_faithfulness(question, answer, context) -> dict` with `claims`, `supported`, `total`, `score`, `model`.
- Produces: `evals.metrics.generation.aggregate(rows) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/evals/test_judge.py`:

```python
import json

import pytest

from evals.judge import DeepSeekJudge, JudgeConfig, JudgeError


def response(claims):
    return {
        "model": "deepseek-chat-2026-10",
        "choices": [
            {
                "message": {
                    "content": json.dumps({"claims": claims})
                }
            }
        ],
    }


def judge(tmp_path, payload, calls=None):
    calls = calls if calls is not None else []

    def transport(_payload):
        calls.append(_payload)
        return payload

    return DeepSeekJudge(
        JudgeConfig(api_key="test-key"),
        cache_dir=tmp_path,
        transport=transport,
    ), calls


def test_faithfulness_is_the_supported_fraction(tmp_path):
    subject, _ = judge(
        tmp_path,
        response(
            [
                {"claim": "A", "supported": True, "evidence": "ctx"},
                {"claim": "B", "supported": False, "evidence": ""},
            ]
        ),
    )
    result = subject.score_faithfulness("q", "a", "ctx")
    assert result["total"] == 2
    assert result["supported"] == 1
    assert result["score"] == 0.5
    assert result["model"] == "deepseek-chat-2026-10"


def test_a_claimless_answer_scores_one(tmp_path):
    subject, _ = judge(tmp_path, response([]))
    assert subject.score_faithfulness("q", "a", "ctx")["score"] == 1.0


def test_cache_prevents_a_second_call(tmp_path):
    subject, calls = judge(
        tmp_path, response([{"claim": "A", "supported": True, "evidence": "ctx"}])
    )
    subject.score_faithfulness("q", "a", "ctx")
    subject.score_faithfulness("q", "a", "ctx")
    assert len(calls) == 1


def test_unparseable_output_raises(tmp_path):
    subject, _ = judge(
        tmp_path,
        {"choices": [{"message": {"content": "not json"}}]},
    )
    with pytest.raises(JudgeError):
        subject.score_faithfulness("q", "a", "ctx")


def test_request_is_deterministic_and_json_mode(tmp_path):
    subject, calls = judge(tmp_path, response([]))
    subject.score_faithfulness("q", "a", "ctx")
    payload = calls[0]
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "deepseek-chat"
```

`tests/evals/test_generation_metrics.py`:

```python
from evals.metrics.generation import aggregate, is_abstention


def test_is_abstention_matches_the_refusal_sentence():
    assert is_abstention("I could not find information about that in the uploaded documents.")
    assert not is_abstention("The answer is on page 4.")


def test_aggregate_combines_faithfulness_and_abstention():
    rows = [
        {"type": "fact", "faithfulness": {"score": 1.0}},
        {"type": "fact", "faithfulness": {"score": 0.5}},
        {"type": "unanswerable", "abstained": True},
        {"type": "unanswerable", "abstained": False},
    ]
    result = aggregate(rows)
    assert result["faithfulness"] == 0.75
    assert result["abstention_accuracy"] == 0.5
    assert result["judged_answers"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/evals/test_judge.py tests/evals/test_generation_metrics.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the judge**

`evals/judge.py`:

```python
"""LLM judge for answer faithfulness.

The judge runs on DeepSeek, a different model family from the GPT-OSS model
that generates the answers. A judge from the generator's own family tends to
prefer its own output, which inflates faithfulness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict evaluator. You return JSON only. "
    "Judge only what the supplied context supports."
)

USER_PROMPT = """Context passages:
{context}

Question: {question}
Answer: {answer}

Break the answer into atomic factual claims. For each claim, decide whether the
context passages support it. A claim not supported by the context is
unsupported, even if it is true in general. Do not reward fluency.

Return JSON with exactly this shape:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "..."}}]}}
"""


class JudgeError(RuntimeError):
    """The judge returned output that could not be parsed."""


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: int = 60


class DeepSeekJudge:
    def __init__(
        self,
        config: JudgeConfig,
        cache_dir: Optional[str | Path] = None,
        transport: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._transport = transport or self._post
        self._lock = threading.Lock()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _cache_key(self, question: str, answer: str, context: str) -> str:
        blob = json.dumps(
            [self.config.model, question, answer, context], sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[dict[str, Any]]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )

    def _payload(self, question: str, answer: str, context: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        context=context, question=question, answer=answer
                    ),
                },
            ],
        }

    def score_faithfulness(
        self, question: str, answer: str, context: str
    ) -> dict[str, Any]:
        key = self._cache_key(question, answer, context)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        raw = self._transport(self._payload(question, answer, context))
        try:
            content = raw["choices"][0]["message"]["content"]
            claims = json.loads(content)["claims"]
            if not isinstance(claims, list):
                raise TypeError("claims is not a list")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise JudgeError(f"judge returned unparseable output: {raw!r}") from exc

        supported = sum(1 for claim in claims if claim.get("supported") is True)
        result = {
            "claims": claims,
            "supported": supported,
            "total": len(claims),
            "score": round(supported / len(claims), 4) if claims else 1.0,
            "model": raw.get("model", self.config.model),
        }
        self._write_cache(key, result)
        return result
```

- [ ] **Step 4: Write the generation metrics**

`evals/metrics/generation.py`:

```python
"""Aggregation for judged answers and abstention."""

from __future__ import annotations

from typing import Any, Sequence

REFUSAL = "could not find information"


def is_abstention(answer: str | None) -> bool:
    return REFUSAL in (answer or "").lower()


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if "faithfulness" in row]
    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    return {
        "faithfulness": (
            round(
                sum(row["faithfulness"]["score"] for row in judged) / len(judged),
                4,
            )
            if judged
            else None
        ),
        "abstention_accuracy": (
            round(
                sum(1 for row in unanswerable if row.get("abstained"))
                / len(unanswerable),
                4,
            )
            if unanswerable
            else None
        ),
        "judged_answers": len(judged),
    }
```

- [ ] **Step 5: Run the tests and commit**

Run: `python -m pytest tests/evals/test_judge.py tests/evals/test_generation_metrics.py -v`

Expected: 7 passed.

```bash
git add evals/judge.py evals/metrics/generation.py tests/evals/test_judge.py tests/evals/test_generation_metrics.py
git commit -m "Add DeepSeek faithfulness judge with caching"
```

---

## Task 8: Full evaluation, regression check, and CI

**Files:**
- Create: `evals/compare.py`
- Create: `evals/thresholds.json`
- Create: `tests/evals/test_compare.py`
- Create: `.github/workflows/eval-nightly.yml`
- Modify: `evals/run_eval.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `evals.compare.compare(baseline, current, thresholds) -> list[str]`.
- Produces: CLI `python -m evals.run_eval --full --hybrid`.
- Produces: CLI `python -m evals.compare --baseline ... --current ... --thresholds ...`.

- [ ] **Step 1: Write the failing test**

`tests/evals/test_compare.py`:

```python
from evals.compare import compare


def test_a_drop_inside_the_threshold_passes():
    baseline = {"summary": {"recall@4": 0.80, "mrr": 0.70}}
    current = {"summary": {"recall@4": 0.79, "mrr": 0.69}}
    thresholds = {"recall@4": -0.02, "mrr": -0.02}
    assert compare(baseline, current, thresholds) == []


def test_a_drop_past_the_threshold_fails():
    baseline = {"summary": {"recall@4": 0.80}}
    current = {"summary": {"recall@4": 0.75}}
    failures = compare(baseline, current, {"recall@4": -0.02})
    assert len(failures) == 1
    assert "recall@4" in failures[0]


def test_a_missing_metric_is_skipped():
    assert compare({"summary": {}}, {"summary": {}}, {"recall@4": -0.02}) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/evals/test_compare.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the comparison module**

`evals/compare.py`:

```python
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
    for metric, allowed_drop in thresholds.items():
        before = baseline.get("summary", {}).get(metric)
        after = current.get("summary", {}).get(metric)
        if before is None or after is None:
            continue
        if after - before < allowed_drop:
            failures.append(
                f"{metric}: {before:.3f} -> {after:.3f} "
                f"(allowed drop {allowed_drop:+.3f})"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--thresholds", required=True)
    args = parser.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    failures = compare(baseline, current, thresholds)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`evals/thresholds.json`:

```json
{
  "recall@4": -0.02,
  "recall@10": -0.02,
  "mrr": -0.02
}
```

- [ ] **Step 4: Extend the runner with judged answers**

In `evals/run_eval.py`, add imports:

```python
import os

from evals.judge import DeepSeekJudge, JudgeConfig
from evals.metrics.generation import aggregate, is_abstention
```

Add:

```python
def run_answers(
    golden: Sequence[dict[str, Any]], engine: RAGEngine, judge: DeepSeekJudge
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in golden:
        result = engine.query(item["question"])
        row: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "answer": result["answer"],
            "sources": result["sources"],
        }
        if item["type"] == "unanswerable":
            row["abstained"] = is_abstention(result["answer"])
        else:
            row["faithfulness"] = judge.score_faithfulness(
                item["question"],
                result["answer"],
                "\n\n".join(result["contexts"]),
            )
        rows.append(row)
    return rows
```

Add the flag:

```python
    parser.add_argument("--full", action="store_true")
```

After the retrieval rows are built, add:

```python
    if args.full:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set; --full requires it.", file=sys.stderr)
            return 2
        judge = DeepSeekJudge(
            JudgeConfig(api_key=api_key), cache_dir="evals/.judge_cache"
        )
        answers = run_answers(golden, engine, judge)
        report["generation"] = aggregate(answers)
        report["answers"] = answers
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/evals -v`

Expected: all pass.

- [ ] **Step 6: Add the push and pull-request job**

Append to `.github/workflows/ci.yml`:

```yaml
  retrieval-eval:
    name: Retrieval evaluation
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Cache corpus
        uses: actions/cache@v4
        with:
          path: evals/corpus/*.pdf
          key: corpus-${{ hashFiles('evals/corpus/lock.json') }}

      - name: Fetch and verify corpus
        run: python -m evals.fetch_corpus

      - name: Run retrieval evaluation
        run: python -m evals.run_eval --out evals/results/current.json

      - name: Compare against baseline
        run: |
          python -m evals.compare \
            --baseline evals/results/baseline.json \
            --current evals/results/current.json \
            --thresholds evals/thresholds.json
```

- [ ] **Step 7: Add the nightly judged job**

`.github/workflows/eval-nightly.yml`:

```yaml
name: Nightly faithfulness evaluation

on:
  schedule:
    - cron: "0 18 * * *"
  workflow_dispatch:

jobs:
  full-eval:
    name: Retrieval plus judge
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Cache corpus
        uses: actions/cache@v4
        with:
          path: evals/corpus/*.pdf
          key: corpus-${{ hashFiles('evals/corpus/lock.json') }}

      - name: Fetch and verify corpus
        run: python -m evals.fetch_corpus

      - name: Run full evaluation
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          python -m evals.run_eval \
            --hybrid \
            --reuse-index \
            --full \
            --out evals/results/latest.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: evals/results/latest.json
```

Note: `--reuse-index` on a fresh runner finds no index and builds one. The flag
only saves time on repeated local runs.

- [ ] **Step 8: Add the secret**

In the GitHub repository settings, add `DEEPSEEK_API_KEY` under Actions
secrets. `GROQ_API_KEY` already exists for the test job.

- [ ] **Step 9: Commit**

```bash
git add evals/compare.py evals/thresholds.json evals/run_eval.py tests/evals/test_compare.py .github/workflows/ci.yml .github/workflows/eval-nightly.yml
git commit -m "Gate CI on retrieval regressions and run nightly faithfulness eval"
```

---

## Task 9: Measure the improvement and document it

**Files:**
- Create: `evals/README.md`
- Modify: `README.md`
- Modify: `evals/results/baseline.json` (unchanged; the comparison is new)
- Create: `evals/results/hybrid.json`

- [ ] **Step 1: Run the improved retrieval**

```bash
python -m evals.run_eval --hybrid --out evals/results/hybrid.json
```

Expected: a summary. Compare it by eye with `evals/results/baseline.json`.

- [ ] **Step 2: Tune only what the numbers justify**

Change one knob at a time, rerunning with `--hybrid --reuse-index`:

- `RAG_BM25_WEIGHT` (0.3, 0.4, 0.5)
- `RAG_CANDIDATE_K` (10, 20, 40)
- `RAG_RERANK_MIN_SCORE` (0.0, 0.5, 1.0)

Stop when `recall@4` and `mrr` stop improving. Do not add features to chase a
number; record what you tried in `evals/README.md`.

- [ ] **Step 3: Write the evaluation README**

`evals/README.md` must contain:

1. What is measured and why.
2. The corpus and how the lock file makes it reproducible.
3. The exact commands to reproduce the baseline and the hybrid run.
4. A results table with your real numbers:

```markdown
| Metric | Baseline (FAISS only) | Hybrid + rerank | Change |
|---|---|---|---|
| recall@4 | 0.000 | 0.000 | +0.000 |
| recall@10 | 0.000 | 0.000 | +0.000 |
| MRR | 0.000 | 0.000 | +0.000 |
| p95 latency | 0 ms | 0 ms | +0 ms |
| faithfulness | 0.000 | 0.000 | +0.000 |
| abstention accuracy | 0.000 | 0.000 | +0.000 |
```

Replace every zero with the value from your runs. Do not commit a table with
invented numbers.

5. Limitations: page-level ground truth, judge alias drift, cross-encoder
   cost, and the 60-question sample size.

- [ ] **Step 4: Add a short section to the root README**

Under "How it works", add:

```markdown
### Evaluation

Intellect ships with a reproducible evaluation harness: 60 hand-written
questions over three pinned public PDFs, deterministic retrieval metrics, and
a cross-family LLM judge for claim-level faithfulness. Run
`python -m evals.run_eval --out evals/results/current.json` to reproduce the
numbers in `evals/README.md`.
```

- [ ] **Step 5: Commit and update the resume**

```bash
git add evals/README.md README.md evals/results/hybrid.json
git commit -m "Document evaluation results and reproduction steps"
```

Now write the three resume bullets from `evals/README.md`, using the real
numbers. If a number is not in the table, it does not go on the resume.

---

## Self-Review

- Spec coverage: corpus (Task 1), golden set (Task 2), retrieval metrics
  (Task 3), runner and baseline (Task 4), hybrid and rerank (Task 5), engine
  integration (Task 6), judge and generation metrics (Task 7), CI and nightly
  (Task 8), documentation and resume output (Task 9). Pre-work claim fixes
  are Task 0.
- No placeholders: the only intentionally open content is the 60 golden
  questions and the final numbers. Both are the user's judgment and are
  produced by named steps with acceptance criteria.
- Type consistency: `retrieve` returns `List[Document]` everywhere;
  `score_faithfulness` returns the `score` key consumed by
  `metrics.generation.aggregate`; `compare` reads `summary` from the report
  written by `run_eval`.
