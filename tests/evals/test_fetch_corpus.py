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
