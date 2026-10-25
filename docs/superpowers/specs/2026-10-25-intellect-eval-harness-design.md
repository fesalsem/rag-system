# Intellect Evaluation Harness and Hybrid Retrieval Design

**Date:** 2026-10-25
**Status:** Approved
**Project:** Intellect (`rag-system`)

## Problem

Intellect has no way to show that retrieval or answer quality is good. Every
change is judged by reading a few answers by eye. That is the same position
every tutorial RAG project is in, so the project cannot support a claim like
"measurably improved retrieval quality."

## Goal

Produce reproducible before-and-after numbers for retrieval and answer
quality, then improve retrieval with hybrid search and reranking, measured
against that baseline. Gate CI so a future change cannot silently regress it.

## Non-goals

- Changing the deployed UI.
- Replacing FAISS or Streamlit.
- Multi-tenant storage or authentication.
- Fine-tuning an embedding model.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Corpus | 2 arXiv papers + 1 annual report, fetched by hash-pinned URL | Public, reproducible, covers prose and tables |
| Golden set | 60 hand-written questions with page labels | Hand-written questions do not echo source wording, so recall is honest |
| Judge model | DeepSeek `deepseek-chat`, temperature 0 | Different model family from the GPT-OSS-20B generator, so no self-preference bias; low cost |
| CI gate | Deterministic retrieval metrics on every push; faithfulness nightly | Judge scores are noisy and cost money on every push |
| Retrieval | BM25 + FAISS reciprocal rank fusion, then cross-encoder rerank | Directly measurable against the baseline |

## Architecture

```
Question
   |
   v
condense (existing, no-op with empty history)
   |
   v
HybridRetriever (only when RAG_HYBRID_ENABLED=1)
   ├── BM25Retriever (top 20)
   ├── FAISS dense retriever (top 20)
   └── EnsembleRetriever (RRF, weights 0.4 / 0.6)
          |
          v
   CrossEncoder rerank -> top 4, drop scores below threshold
   |
   v
RAGEngine.query (existing prompt, overview fallback, sources)
```

With `RAG_HYBRID_ENABLED=0` the engine keeps the current FAISS-only path, so
existing behaviour and tests are unchanged. The flag is the rollback.

The evaluation package is dev-only. `app.py` never imports `evals/`.

## Metric definitions

Retrieval, deterministic:

- `recall@k`: fraction of answerable questions where at least one retrieved
  chunk comes from a gold page.
- `MRR`: mean of `1 / rank` of the first correct page, 0 when absent.
- Unanswerable questions are excluded from retrieval metrics and scored
  separately for abstention.

Generation, judged:

- Faithfulness: the judge extracts claims from the answer, then marks each
  claim supported or unsupported by the retrieved context. Score is
  `supported / total`. A claimless answer scores 1.0 and is flagged.
- Abstention accuracy: fraction of unanswerable questions answered with the
  refusal sentence.

## File layout

```
evals/
├── __init__.py
├── corpus/manifest.json       # URLs, titles, local file names
├── corpus/lock.json           # committed SHA-256 per document
├── fetch_corpus.py            # download and verify
├── golden.jsonl               # 60 questions, hand-authored
├── golden.py                  # schema and validator
├── metrics/__init__.py
├── metrics/retrieval.py       # recall@k, MRR
├── metrics/generation.py      # faithfulness aggregation, abstention
├── judge.py                   # DeepSeek client, rubric, cache
├── run_eval.py                # CLI runner
├── compare.py                 # baseline regression check
├── thresholds.json            # allowed metric drops
└── results/baseline.json      # committed baseline
retrieval.py                   # app-side hybrid retriever
tests/evals/                   # offline tests for all of the above
```

## Risks and limits

- Page-level ground truth is coarser than chunk-level. State it in
  `evals/README.md`; do not hide it.
- `deepseek-chat` is a moving alias. Record the `model` string returned by
  the API in every result file.
- The cross-encoder adds roughly 90 MB and a few hundred milliseconds.
  Lazy-load it, keep hybrid off by default, and bake the model into the
  Docker image only if hybrid is turned on in production.
- Judge output is non-deterministic even at temperature 0. Cache by hash and
  commit the raw judge output alongside the score.

## Success criteria

1. `python -m evals.run_eval` writes a baseline report on unchanged code.
2. Hybrid retrieval is behind a flag and the existing suite stays green.
3. CI fails when `recall@4` or `MRR` drops past the threshold.
4. The README carries a baseline and post-change table with real numbers.
5. Three resume bullets can be written from the output without inventing a
   figure.
