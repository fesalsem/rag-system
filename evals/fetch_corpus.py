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

    if update_lock:
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
