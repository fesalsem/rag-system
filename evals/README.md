# Evaluation

Retrieval quality and answer faithfulness are measured here, not assumed. Every
number in this file comes from the committed code, the pinned corpus, and the
60-question golden set, and any of them can be reproduced with the commands
below.

## What is measured

| Question | Metric | Set |
|---|---|---|
| Did the right page reach the prompt? | recall@4, recall@10 | 50 answerable questions |
| How high did the first correct page rank? | MRR | 50 answerable questions |
| How fast is retrieval? | p50, p95 latency | all 60 questions |
| Is every claim in the answer supported by the retrieved text? | faithfulness | 50 answerable questions |
| Did the system refuse when the corpus cannot answer? | abstention accuracy | 10 unanswerable questions |

Retrieval is scored at page level: a hit means a retrieved chunk came from a
gold page. That is coarse on purpose. Chunk-level ground truth would need a
human to mark the exact passage for every question, and the page is what the
user sees in the citation.

Faithfulness is judged by DeepSeek, a different model family from the
GPT-OSS-20B that writes the answers. A judge from the generator's own family
tends to score its own output higher. The judge splits the answer into atomic
claims and marks each one supported or not; the score is the supported
fraction. Refusals make no claims and are not judged.

## The corpus

Three public PDFs, chosen to cover prose, formulas, and financial tables:

| Document | Source | Why |
|---|---|---|
| `attention` | arXiv 1706.03762 | dense technical prose, tables |
| `rag-paper` | arXiv 2005.11401 | the architecture this app implements |
| `berkshire-2023` | berkshirehathaway.com 2023 annual report | long document, financial statements |

`evals/corpus/lock.json` stores each file's SHA-256. `python -m evals.fetch_corpus`
downloads only what is missing, verifies every byte against the lock, and
fails rather than scoring a different document. The PDFs themselves are
gitignored; the lock file is the contract.

## The golden set

`evals/golden.jsonl` holds 60 questions: 30 fact, 10 table, 10 multihop, 10
unanswerable. `gold_pages` uses the 1-based PDF page index, which is what
`document_processor` writes into each chunk's `page_label`, so a question about
Berkshire's balance sheet points at PDF page 94 even though the printed label
says K-70.

`python -m evals.golden --strict` validates the file: required fields, unique
ids, known document ids, and the 30/10/10/10 split.

## Reproduce

```bash
pip install -r requirements.txt

# Dense baseline
python -m evals.run_eval --out evals/results/baseline.json

# Hybrid BM25 + FAISS with cross-encoder reranking
python -m evals.run_eval --hybrid --reuse-index --out evals/results/hybrid.json

# Add claim-level faithfulness and abstention (needs DEEPSEEK_API_KEY and GROQ_API_KEY)
python -m evals.run_eval --hybrid --reuse-index --full --out evals/results/full.json

# Fail if retrieval moved backwards
python -m evals.compare \
  --baseline evals/results/baseline.json \
  --current evals/results/hybrid.json \
  --thresholds evals/thresholds.json
```

## Results

Same index, same 60 questions, same machine.

| Metric | Dense (FAISS only) | Hybrid + rerank | Change |
|---|---|---|---|
| recall@4 | 0.78 | **0.86** | +0.08 |
| recall@10 | 0.78 | **0.86** | +0.08 |
| MRR | 0.575 | **0.740** | +0.165 |
| p50 latency | 8.18 ms | 961 ms | +953 ms |
| p95 latency | 10.47 ms | 1227 ms | +1216 ms |
| faithfulness | pending | pending | — |
| abstention accuracy | pending | pending | — |

`recall@10` equals `recall@4` for dense retrieval because the dense path asks
FAISS for exactly four chunks; there is nothing to rank between 5 and 10.

The judged rows are marked pending on purpose. A judged run made before the
empty-hybrid fallback fix (`df5e0b0`) scored 0.84 faithfulness over 50 answers
and 5/10 strict abstentions, but that run answered several questions from a
whole-document overview after the reranker rejected every candidate. The fix
changes those answers, so the old numbers do not describe this code. The rerun
is blocked on the free Groq tier's daily token cap.

### What changed, question by question

Dense-only misses clustered in the Berkshire financial statements and in
questions whose answer is a definition stated once in prose. BM25 matches the
number or name the embedding smeared. Before the fallback fix, hybrid fixed 5
questions (`fact-013`, `fact-023`, `fact-027`, `table-001`, `table-007`) and
broke 2 (`fact-003`, `multihop-001`).

### Tuning

One knob at a time, `--reuse-index`, 60 questions:

| Knob | Setting | recall@4 | MRR | p50 |
|---|---|---|---|---|
| `RAG_BM25_WEIGHT` | 0.3 | 0.86 | 0.740 | 974 ms |
| | 0.4 (default) | 0.86 | 0.740 | 961 ms |
| | 0.5 | 0.86 | 0.740 | 911 ms |
| `RAG_CANDIDATE_K` | 10 | 0.80 | 0.693 | 431 ms |
| | 20 (default) | 0.86 | 0.740 | 961 ms |
| | 40 | 0.86 | 0.753 | 1830 ms |

The fusion weight does not move the metrics because the cross-encoder sets the
final order once both retrievers have contributed candidates. The candidate
pool matters in both directions: 10 loses six points of recall, while 40 buys
0.013 MRR for double the latency. Defaults stayed at 0.4 and 20.

## Limitations

- **Page-level ground truth.** A hit means the right page, not necessarily the
  right passage. Two chunks from the same page both count.
- **Judge alias drift.** The judge reports the model alias DeepSeek returns
  (for example `deepseek-chat-2026-10`). If the alias changes, cached verdicts
  are still keyed by the configured model name, so a provider-side swap is
  invisible unless the cache is cleared.
- **Cross-encoder cost.** The reranker adds roughly one second per query on
  CPU. That is why `RAG_HYBRID_ENABLED` defaults to `0` and the deployed app
  keeps the dense path.
- **Sample size.** 60 questions means one question moves recall by two
  percentage points. Differences under about 0.04 are noise.
- **Phrase-based abstention.** A refusal counts when it contains one of two
  refusal phrases. A more creative refusal would be scored as a failure.

## CI

`.github/workflows/ci.yml` reruns the retrieval evaluation on every push and
fails when `recall@4`, `recall@10`, or `mrr` drops more than 0.02 below
`evals/results/baseline.json`. `.github/workflows/eval-nightly.yml` runs the
full judged evaluation daily and uploads `evals/results/latest.json`.
