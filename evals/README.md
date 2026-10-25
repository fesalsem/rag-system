# Evaluation

Retrieval and answer quality are measured, not assumed. Every number here is
reproducible from the committed code.

## What is measured

| Metric | What it answers | Set |
|---|---|---|
| recall@4, recall@10 | Did the right page reach the top of the prompt? | 50 answerable |
| MRR | How high did the first correct page rank? | 50 answerable |
| p50, p95 latency | How fast is retrieval? | all 60 |
| faithfulness | Is each claim supported by the retrieved text? | 50 answerable |
| abstention accuracy | Did it refuse when the corpus cannot answer? | 10 unanswerable |

Retrieval is scored by page: a hit means a retrieved chunk came from a gold
page. That is the same page number the user sees in a citation.

Faithfulness is judged by DeepSeek, a different model family from the
GPT-OSS-20B that writes the answers, so the judge is not grading its own
output. It splits an answer into claims and returns the supported fraction.
Refusals make no claims and are not judged.

## Corpus and questions

Three public PDFs, pinned by SHA-256 in `evals/corpus/lock.json`:

| Id | Document | Why |
|---|---|---|
| `attention` | arXiv 1706.03762 | dense technical prose and tables |
| `rag-paper` | arXiv 2005.11401 | the architecture this app implements |
| `berkshire-2023` | Berkshire 2023 annual report | long document, financial statements |

`python -m evals.fetch_corpus` downloads only what is missing and fails if a
byte does not match the lock. The PDFs are gitignored; the lock is the contract.

`evals/golden.jsonl` holds 60 questions: 30 fact, 10 table, 10 multihop, 10
unanswerable. `gold_pages` is the 1-based PDF page index, which is what
`page_label` stores, so the Berkshire balance sheet is page 94 even though the
printed label says K-70. `python -m evals.golden --strict` validates the file.

## Reproduce

```bash
pip install -r requirements.txt

python -m evals.run_eval --out evals/results/baseline.json              # dense
python -m evals.run_eval --hybrid --reuse-index --out evals/results/hybrid.json
python -m evals.run_eval --hybrid --reuse-index --full --out evals/results/full.json
```

`--full` adds faithfulness and abstention. It needs `GROQ_API_KEY` and
`DEEPSEEK_API_KEY`.

## Results

Same index, same 60 questions, same machine.

| Metric | Dense | Hybrid + rerank | Change |
|---|---|---|---|
| recall@4 | 0.78 | **0.86** | +0.08 |
| recall@10 | 0.78 | **0.86** | +0.08 |
| MRR | 0.575 | **0.740** | +0.165 |
| p50 latency | 8 ms | 961 ms | +953 ms |
| p95 latency | 10 ms | 1227 ms | +1217 ms |
| faithfulness | pending | pending | — |
| abstention accuracy | pending | pending | — |

`recall@10` equals `recall@4` for dense because the dense path requests exactly
four chunks. There is nothing ranked between 5 and 10.

Faithfulness and abstention are marked pending. A judged run made before the
empty-hybrid fallback fix (`df5e0b0`) scored 0.84 and 5/10, but it answered
several questions from a whole-document overview, so it does not describe this
code. The rerun is blocked on the free Groq tier's daily token cap.

## Tuning

One knob at a time, same index and questions:

| Knob | Setting | recall@4 | MRR | p50 |
|---|---|---|---|---|
| `RAG_BM25_WEIGHT` | 0.3 | 0.86 | 0.740 | 974 ms |
| | 0.4 (default) | 0.86 | 0.740 | 961 ms |
| | 0.5 | 0.86 | 0.740 | 911 ms |
| `RAG_CANDIDATE_K` | 10 | 0.80 | 0.693 | 431 ms |
| | 20 (default) | 0.86 | 0.740 | 961 ms |
| | 40 | 0.86 | 0.753 | 1830 ms |

The fusion weight changes nothing because the cross-encoder sets the final
order. The candidate pool does: 10 loses 6 points of recall, 40 buys 0.013 MRR
for double the latency. Defaults stayed at 0.4 and 20.

## Limitations

- **Page-level truth.** A hit means the right page, not the right passage.
- **Small sample.** One question moves recall by 2 points; treat gaps under
  0.04 as noise.
- **Judge alias drift.** Cached verdicts are keyed by the configured model
  name, so a provider-side model swap is invisible unless the cache is cleared.
- **Reranker cost.** About one second per query on CPU. That is why
  `RAG_HYBRID_ENABLED` defaults to `0` in the deployed app.
- **Phrase-based abstention.** A refusal counts when it matches one of two
  refusal phrases.

## CI

Every push reruns retrieval and fails if `recall@4`, `recall@10`, or `mrr`
drops more than 0.02 below `evals/results/baseline.json`
(`evals/compare.py`, `evals/thresholds.json`). A nightly workflow runs the full
judged evaluation and uploads the JSON.
