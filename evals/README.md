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

The answer pass clears conversation memory before each question. The judged
runs recorded below did not: the engine condensed question N against the answer
to question N-1, so some answers respond to a rewritten question, and the run
order decided what that rewrite was. A source mismatch makes it visible. The
run of 2026-10-28 answered table-010 from pages 125, 97, 60, 98, 66 and 91,
while that same run's own retrieval rows for table-010 list pages 25, 30, 13,
66, 98 and 125. Judge rows from before the fix describe the system it was, not
the system it is, and are not comparable to a new run row by row.

## Results

Same index, same 62 questions, same machine, both runs after the metric fixes
of 2026-10-27.

| Metric | Dense | Hybrid + rerank | Change |
|---|---|---|---|
| recall@4 | 0.808 | **0.865** | +0.058 |
| recall@10 | **0.923** | 0.904 | -0.019 |
| MRR | 0.609 | **0.751** | +0.143 |
| p50 latency | 13 ms | 1212 ms | +1199 ms |
| p95 latency | 24 ms | 1451 ms | +1427 ms |
| faithfulness | not run | **0.96** | — |
| abstention accuracy | not run | **0.80** (8 of 10) | — |
| false abstention rate | not run | 0.019 (1 of 52) | — |

Both paths keep at most one chunk per page, and each retrieved chunk now
brings a window of three chunks from its page. Dense still finds more gold
pages by rank 10, while hybrid ranks the right page first more often: MRR 0.751
against 0.609. The judged rows come from the run of 2026-11-03, with memory
cleared between questions.

Retrieval is scored at k=10 for both paths, which the app does not do: it
answers from six chunks. That is why recall@10 now differs from recall@4 and
why the latency rows are about 60 ms higher than an answering query costs.

Faithfulness and abstention come from `evals/results/full.json`, rerun on
2026-11-03 with `--hybrid --reuse-index --full` at the current defaults,
`RAG_MAX_PER_PAGE=1`, `RAG_PAGE_CHUNKS=3` and `max_tokens=8192`, with memory
cleared between questions. The judge scores each claim against the retrieved
context, and the table reports mean support per judged answer. Abstention
accuracy is 8 of 10. Two answers gave one half of a two-part question and
refused the other; they are counted as partial answers rather than refusals.
No generation came back empty, and the single false abstention, `table-008`,
followed a retrieval miss.

Two changes moved these rows. The previous run, window off and a 4096-token
budget, scored 0.955 over 44 judged answers with 8 false abstentions. The page
window is the larger factor: on the 30-question pair it alone cut false
abstentions from 4 of 26 to 1 of 26. The 1024-token default also truncated the
hidden reasoning on two two-part questions and returned empty answers, so the
budget is now 8192.

Older runs are kept for context. The uncapped run of 2026-10-28 scored 0.952
over 49 judged answers with 3 false abstentions; it predates the page cap, the
page window and the memory reset, so it is not comparable row by row.

Two limitations of the remaining abstention misses. `unans-007` (return on
equity) and `unans-012` (BNSF operating ratio) were answered by computing a
ratio from two figures that are in the corpus; for `unans-012` the arithmetic
checks out. The golden set treats a derived figure as not stated. That is a
policy choice to make explicit, not a hallucination.

## Page cap sweep

Dense retrieval returned near-duplicate chunks from the same page: 81% of the
52 answerable questions had a repeated page in their top six, and 94% in their
top ten. `RAG_MAX_PER_PAGE` caps how many chunks one page may contribute. The
sweep below is dense-only, same index and questions, retrieval at k=10.

| Cap | recall@4 | recall@10 | MRR | top-6 lists with a repeat | table recall@10 |
|---|---|---|---|---|---|
| 0 (off) | 0.788 | 0.865 | 0.592 | 81% | 0.700 |
| 1 | **0.808** | **0.923** | **0.609** | **6%** | **0.900** |
| 2 | 0.788 | 0.885 | 0.594 | 81% | 0.800 |
| 3 | 0.788 | 0.865 | 0.592 | 81% | 0.700 |

At one chunk per page, one question gained at rank 4 and three gained at rank
10; none lost. Hybrid benefits less because its reranker already spread the
list: recall@10 rose 0.885 to 0.904, MRR 0.744 to 0.751, and the share of top-six
lists with a repeat fell from 53% to 0%. One is now the default on both paths,
and the judged results table above now covers the capped system.


## Page windows, paired run

`RAG_PAGE_CHUNKS` widens each retrieved chunk to a window of chunks from the
same page. This pair isolates the knob: same index, same 30-question subset,
hybrid and capped at one chunk per page on both sides, with only the window
changed. The subset is 16 fact, 5 table, 5 multi-hop and 4 unanswerable
questions, cut from the 62 to fit the free tier's daily token budget. Both runs
clear memory between questions.

| Metric | Window 0 | Window 3 |
|---|---|---|
| recall@4 | 0.808 | 0.808 |
| recall@10 | 0.846 | 0.846 |
| MRR | 0.722 | 0.722 |
| false abstention | 0.154 (4 of 26) | **0.038 (1 of 26)** |
| abstention accuracy | 0.75 (3 of 4) | 0.75 (3 of 4) |
| claim-level support | 0.900 (36 of 40) | **0.920 (46 of 50)** |
| judged answers | 22 | 25 |

Retrieval rows are identical by construction: the window changes what reaches
the prompt, not what is retrieved. Three of the four false abstentions
(`fact-001`, `fact-003`, `fact-028`) were answered at window 3, and all three
were judged fully supported. The fourth, `table-008`, refused on both sides.

The two runs abstained on different questions, so the totals above cover
different answer sets. Restricted to the 22 questions judged on both sides,
claim support is 36 of 40 against 42 of 46. With 26 answerable questions, one
refusal is 3.8 points, so read the abstention gap as a direction. Three is now
the default. The full run behind the results table recorded one false
abstention; the window-off run recorded eight, though it also used the smaller
token budget.

Two reading caveats for this pair. The control run returned three empty
answers (`table-001`, `table-005`, `multihop-005`) and the window run returned
one (`multihop-005`). An answer with no claims scores 1.0 under the judge's
default, so per-answer means are inflated on both sides and are not quoted
here; claim totals ignore empty answers. Latency is not quoted either: the
window adds prompt tokens, and the two medians (978 ms and 908 ms) differ in
the wrong direction, which makes the difference noise at this sample. The
`unans-012` miss is present in both runs and is the derived-answer policy
question noted under Limitations.

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

## Threshold calibration

`RAG_MAX_DISTANCE` is the only relevance gate on the dense path, and the dense
path is what the deployed app runs. It was set to 1.4 by hand, from cosine 0.3.
Sweeping it over the 62 questions on the saved index, capped at one chunk per
page and scored at k=10:

| `RAG_MAX_DISTANCE` | answerable with context | unanswerable with context | recall@4 | recall@10 |
|---|---|---|---|---|
| 1.0 | 41/52 | 8/10 | 0.577 | 0.635 |
| 1.1 | 49/52 | 9/10 | 0.692 | 0.769 |
| 1.2 | 52/52 | 10/10 | 0.769 | 0.846 |
| 1.3 | 52/52 | 10/10 | 0.788 | 0.885 |
| **1.4** | **52/52** | **10/10** | **0.808** | **0.923** |
| 1.5 | 52/52 | 10/10 | 0.808 | 0.923 |
| 1.6 | 52/52 | 10/10 | 0.808 | 0.923 |
| 1.7 | 52/52 | 10/10 | 0.808 | 0.923 |

1.4 is the lowest cutoff that reaches the best recall@10; above it nothing
changes, and below it answerable questions start losing their context. The
classes never separate: every cutoff loose enough to keep all 52 answerable
questions in context also keeps all 10 unanswerable ones. A distance gate
therefore cannot improve abstention here. The remaining misses are a ranking
miss (`table-008`) and two derived answers (`unans-007`, `unans-012`).

The hybrid gate behaves the same way once the engine's dense fallback is
counted. At 0.0 the reranker rejects every candidate for 2 of 52 answerable
questions and 2 of 10 unanswerable ones, but `RAGEngine.retrieve` falls back to
dense on an empty rerank, and the recorded run has non-empty context for all 62.
Sweeping the gate moves the ordering, not whether context exists: -1.0 keeps
the same unanswerable set and moves recall@10 by 0.038, this sample's noise
floor, while 1.0 removes one question from each class. The default is the
cross-encoder's trained relevance boundary, and no neighbouring value beats it
outside noise, so both defaults stay unchanged.

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
  read as a claim even when it refuses the question. One of the 52
  answerable questions was refused (`table-008`), after a retrieval miss.
- **Derived answers are scored as wrong.** `unans-007` and `unans-012` were
  answered by computing a ratio from figures in the corpus. The golden set
  says the corpus does not state the figure. Decide the policy before
  counting these as failures.
- **Failure modes that flatter or punish the numbers.** Four unsupported
  claims were page numbers written into the answer text ("page 4", "page 97")
  that the judge cannot see in the context, so they are judge artefacts
  (`fact-011`, `fact-030`, `table-010`). One (`multihop-005`) was a real gap:
  the judge did not accept the answer's claim that the retriever is held fixed.
  Read faithfulness as a lower bound until the prompt stops writing page
  numbers into prose.

## CI

Every push reruns retrieval and fails if `recall@4`, `recall@10`, or `mrr`
drops more than 0.02 below `evals/results/baseline.json`
(`evals/compare.py`, `evals/thresholds.json`). A weekly workflow runs the full
judged evaluation and uploads the JSON. It is weekly because a full judged run
spends about 150k of the free tier's 200k daily tokens.
