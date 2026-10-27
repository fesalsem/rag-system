# Evaluation

Retrieval and answer quality are measured, not assumed. Every number here is
reproducible from the committed code.

## What is measured

| Metric | What it answers | Set |
|---|---|---|
| recall@4, recall@10 | Did the right page reach the top of the prompt? | 52 answerable |
| MRR | How high did the first correct page rank? | 52 answerable |
| p50, p95 latency | How fast is retrieval? | all 62 |
| faithfulness | Is each claim supported by the retrieved text? | 52 answerable |
| abstention accuracy | Did it refuse when the corpus cannot answer? | 10 unanswerable |
| false abstention rate | Did it refuse when the corpus can answer? | 52 answerable |
| partial answers | How many answers refused part of the question? | 52 answerable |

Retrieval is scored by page: a hit means a retrieved chunk came from a gold
page. That is the same page number the user sees in a citation.

Faithfulness is judged by DeepSeek, a different model family from the
GPT-OSS-20B that writes the answers, so the judge is not grading its own
output. It splits an answer into claims and returns the supported fraction.
Refusals make no claims and are not judged.

A refusal is scored per sentence: an answer counts as a refusal only when
every sentence refuses. An answer that gives one fact and refuses the rest is
a partial answer, counted separately, because treating it as a refusal would
hide work the system did. Retrieval is scored at k=10 even though the app
answers from 4 chunks; scoring at 4 made recall@10 a copy of recall@4.

## Corpus and questions

Three public PDFs, pinned by SHA-256 in `evals/corpus/lock.json`:

| Id | Document | Why |
|---|---|---|
| `attention` | arXiv 1706.03762 | dense technical prose and tables |
| `rag-paper` | arXiv 2005.11401 | the architecture this app implements |
| `berkshire-2023` | Berkshire 2023 annual report | long document, financial statements |

`python -m evals.fetch_corpus` downloads only what is missing and fails if a
byte does not match the lock. The PDFs are gitignored; the lock is the contract.

`evals/golden.jsonl` holds 62 questions: 32 fact, 10 table, 10 multihop, 10
unanswerable. Two rows that were labelled unanswerable were corrected on
2026-10-27 after checking the corpus: page 57 states that no cash dividend has
been declared since 1967, and page 30 states BNSF Railway had about 37,000
employees. The system had answered both correctly and was being scored wrong
for it. Two new unanswerable questions replaced them, so the set still holds
ten. `gold_pages` is the 1-based PDF page index, which is what
`page_label` stores, so the Berkshire balance sheet is page 94 even though the
printed label says K-70. `python -m evals.golden --strict` validates the file.

## Reproduce

```bash
pip install -r requirements.txt

python -m evals.run_eval --out evals/results/baseline.json              # dense
python -m evals.run_eval --hybrid --reuse-index --full --out evals/results/full.json
```

`--full` adds faithfulness and abstention. It needs `GROQ_API_KEY` and
`DEEPSEEK_API_KEY`.

## Results

Same index, same 62 questions, same machine, both runs after the metric fixes
of 2026-10-27.

| Metric | Dense | Hybrid + rerank | Change |
|---|---|---|---|
| recall@4 | 0.789 | **0.865** | +0.077 |
| recall@10 | 0.865 | **0.885** | +0.019 |
| MRR | 0.592 | **0.744** | +0.152 |
| p50 latency | 13 ms | 1028 ms | +1015 ms |
| p95 latency | 21 ms | 1472 ms | +1451 ms |
| faithfulness | not run | **0.91** | — |
| abstention accuracy | not run | **0.80** (8 of 10) | — |
| false abstention rate | not run | 0.13 (7 of 52) | — |

Retrieval is scored at k=10 for both paths, which the app does not do: it
answers from six chunks. That is why recall@10 now differs from recall@4 and
why the latency rows are about 60 ms higher than an answering query costs.

Faithfulness and abstention come from `evals/results/full.json`, rerun on
2026-10-27 with `--hybrid --reuse-index --full`. The judge scores each claim
against the retrieved context: 0.907 over the 45 answers that made a claim.
Abstention accuracy is 8 of 10. One answer gave one half of a two-part
question and refused the other, and is counted as a partial answer rather than
a refusal.

That judged run used the four-chunk default the app answered from at the time.
A rerun at the current six-chunk default was started later on 2026-10-27 and
stopped after 22 of 62 answers when Groq's 200k daily token cap was reached.
Its checkpoint is kept in `evals/.answers.jsonl`, so the same command continues
from there; until it completes, the judged numbers above describe the
four-chunk configuration.

Two limitations of the remaining abstention misses. `unans-007` (return on
equity) and `unans-012` (BNSF operating ratio) were answered by computing a
ratio from two figures that are in the corpus; for `unans-012` the arithmetic
checks out. The golden set treats a derived figure as not stated. That is a
policy choice to make explicit, not a hallucination.

## Tuning

One knob at a time, same index and questions. These sweeps ran on the earlier
60-question set, before two labels were corrected; the direction holds but the
last digit is stale.

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
- **Sentence-based abstention.** An answer counts as a refusal when every
  sentence refuses. A refusal is still recognised by wording, so a new
  phrasing needs a new pattern, and the model's own summary of a table can
  read as a claim even when it refuses the question. Seven of the 52
  answerable questions were refused. Four followed a retrieval miss, two
  refused with the evidence already in the context, and one (`fact-029`) had
  the right page returned but a chunk from that page that did not hold the
  figure.
- **Derived answers are scored as wrong.** `unans-007` and `unans-012` were
  answered by computing a ratio from figures in the corpus. The golden set
  says the corpus does not state the figure. Decide the policy before
  counting these as failures.
- **Judge wants one statement.** Two claims scored 0 because no single
  passage states the figure: `table-001` and `table-009` summed two line items
  that do add up to it. Faithfulness is a lower bound there, not a wrong
  answer.

## CI

Every push reruns retrieval and fails if `recall@4`, `recall@10`, or `mrr`
drops more than 0.02 below `evals/results/baseline.json`
(`evals/compare.py`, `evals/thresholds.json`). A weekly workflow runs the full
judged evaluation and uploads the JSON. It is weekly because a full judged run
spends about 150k of the free tier's 200k daily tokens.
