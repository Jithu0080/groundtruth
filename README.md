# Groundtruth

**An evaluation harness that measures whether a retrieval change actually helped a legal research assistant, and blocks the changes that make it worse.**

> **Headline:** hybrid search + a cross-encoder reranker raises recall@10 from **0.513 to 0.676** on 74 held-out legal questions (**+0.162, 95% CI +0.041 to +0.284**), raises MRR from 0.291 to 0.491, and sends **half as much text** to the LLM. A CI gate fails any pull request that drops recall@10 by more than 1 point.

## Results

Test split of `data/golden_v1.jsonl` (74 questions). Corpus: 700 Supreme Court of India judgments from 2023. Embedding model `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 50 first-stage results. Latency measured on a Colab T4 GPU.

| Config | Recall@10 | Recall@5 | MRR | nDCG@10 | Avg chars in top 10 | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| dense, 800-char chunks (baseline) | 0.513 | 0.419 | 0.291 | 0.344 | 7,916 | 27 |
| dense, 400 | 0.540 | 0.460 | 0.329 | 0.380 | 3,952 | 31 |
| dense, 1600 | 0.622 | 0.460 | 0.302 | 0.376 | 15,825 | 20 |
| BM25, 400 | 0.446 | 0.351 | 0.276 | 0.315 | 3,956 | 7 |
| hybrid (dense + BM25, RRF), 400 | 0.581 | 0.486 | 0.324 | 0.385 | 3,955 | 48 |
| **hybrid + reranker, 400 (production)** | **0.676** | **0.595** | **0.491** | **0.535** | 3,958 | 132 |
| dense + reranker, 400 (ablation) | 0.703 | 0.608 | 0.500 | 0.549 | 3,958 | 134 |

Paired bootstrap against the baseline (10,000 resamples of queries, recall@10):

| Config | Δ recall@10 | 95% CI | Verdict |
| --- | --- | --- | --- |
| dense, 400 | +0.027 | −0.095 to +0.149 | noise |
| dense, 1600 | +0.108 | +0.000 to +0.216 | borderline |
| BM25, 400 | −0.068 | −0.203 to +0.068 | noise |
| hybrid, 400 | +0.068 | −0.054 to +0.189 | noise |
| **hybrid + reranker, 400** | **+0.162** | **+0.041 to +0.284** | **real** |
| dense + reranker, 400 | +0.189 | +0.068 to +0.311 | real |

Recall@10 by question category:

| Category (test n) | Baseline | Production | Change |
| --- | --- | --- | --- |
| factual (16) | 0.812 | 0.812 | 0 |
| holding (15) | 0.533 | 0.600 | +0.067 |
| paraphrase (14) | 0.357 | 0.571 | +0.214 |
| procedural (11) | 0.364 | 0.727 | +0.364 |
| statute (18) | 0.444 | 0.667 | +0.222 |

Each category has 11–18 test questions, so one question moves a category score by 6–9 points. Read the category table as patterns, not precise measurements.

## Recommendation

**Ship hybrid + reranker with 400-character chunks.** It is the only change chosen before seeing test results that beats the baseline beyond noise: recall@10 +0.162 (95% CI +0.041 to +0.284), MRR 0.291 → 0.491. It also halves the text sent to the LLM (7,916 → 3,958 characters per query). The cost is about **105 ms of extra p95 latency** (27 → 132 ms), acceptable for a research tool where a missed precedent costs far more than a tenth of a second.

If the latency budget were under ~50 ms, no fast configuration beat the baseline beyond noise; the reasonable choice would be dense with 400-character chunks, which matches baseline recall at half the context cost.

**For the next version:** an ablation found that **BM25 adds nothing measurable once the reranker is in place** (dense + reranker 0.703 vs hybrid + reranker 0.676; the gap is within noise). Dropping BM25 would simplify the system at no cost, but that ablation was run on the test split, so it should be confirmed on fresh questions before switching. I kept the configuration chosen in advance rather than letting the test set pick production.

## What the results taught me

1. **The reranker does the work.** The baseline already had 77% of answers in its top 50 but only 51% in its top 10. Re-scoring those 50 with a cross-encoder recovered most of the gap.
2. **Keyword search was the weakest retriever.** BM25 scored lowest overall (0.446). I expected it to dominate statute questions, but its edge there was small: 0.389 vs 0.333 for dense at the same chunk size, a single question's difference. The questions were deliberately written in plain language rather than copied from the passages, which leaves few exact terms to match. An eval set that copies passage wording would make BM25 look far better than it is for real users.
3. **Large chunks inflate recall.** 1600-character chunks scored 0.622, but they send four times as much text, and a big chunk "covers" an answer more easily. Recall has to be read next to context size.
4. **A 30-question dev split is noisy.** Dev picked 400-character chunks (1600 was worst on dev); on test, 1600 was second best. I kept the dev choice and report the disagreement, because switching after seeing test results is exactly the overfitting the split exists to prevent.

## Error analysis

The production config misses the labelled passage on 24 of 74 test questions. I read every one:

| Root cause | Count | Example | A real retrieval failure? |
| --- | --- | --- | --- |
| Answer retrieved from **elsewhere in the same judgment** | 10 | q073: the top result repeats the gold sentence word for word, because the headnote copies the judgment body | Mostly no: the gold set marks one passage when several answer the question |
| **Near-identical text in another judgment** | 7 | q101: the top result is the same sentence in a different file, so the corpus contains the same judgment twice; q022/q035: many land acquisition judgments repeat the same *Indore Development Authority* paragraph | Mostly no: the label is stricter than the question |
| **Vocabulary mismatch** | 4 | q070: "accepting a benefit, then challenging it" never reached the Latin term *approbate and reprobate* | Yes |
| **Question too vague** | 3 | q081: "the earlier compensation case discussed" does not identify a case | No: a labelling weakness |

**Document-level recall** supports this: the right *judgment* is in the top 10 for about 85% of questions in every configuration (0.851 for production). Only about 5% of test questions (4 of 74) are genuine vocabulary misses, so **the strict passage-level metric under-states the system.**

**Known label errors** (kept as-is, because `golden_v1` is frozen): q049's question starts with a typo ("f a court…"); q073 starts with a stray ": "; q033 is tagged *statute* but is a paraphrase question.

## How it works

1. **Corpus.** 700 English judgments sampled with a fixed seed from the 2023 volume (`data/corpus_sample.txt` lists them), cleaned into plain text: page numbers and Supreme Court Reports margin letters removed, and words split by PDF extraction re-joined using corpus statistics. 35.9 million characters.
2. **Golden set.** 104 question–passage pairs in five categories (holding 21, statute 25, factual 23, procedural 15, paraphrase 20), split 30 dev / 74 test, stratified by category. Gold passages are stored as **character spans**, not chunk IDs, so they stay valid under any chunking. Questions were drafted with an LLM from 150 randomly sampled passages, then reviewed, edited and accepted one by one by me; 16 passages were rejected as boilerplate, citation lists or garbled text. Every answer span was checked automatically against the corpus.
3. **Retrieval metrics.** Recall@5, recall@10, MRR and nDCG@10, overall and per category. A chunk counts as a hit if it covers at least 50% of a gold span, and each gold span is credited once, so overlapping chunks cannot double-count.
4. **Experiments as configs.** Chunk size, dense vs BM25 vs hybrid (reciprocal rank fusion, c = 60), and a reranker on or off. Each is a YAML file in `configs/experiments/`; settings were chosen on dev and reported on test.
5. **Generation metrics.** Faithfulness (a claim-level LLM judge) and answer relevance (a 1–5 rubric), reported separately from retrieval. Implemented and unit-tested, but not yet run on this corpus.
6. **CI gate.** On every pull request, GitHub Actions re-runs `configs/production.yaml` on the test split. It fails if recall@10 drops more than 1 point below `results/baseline.json`, or any category drops more than 5 points. Retrieval is deterministic (pinned model, exact search), so the gate is not flaky; the slow, paid generation eval stays out of it.

**The gate in action:** a pull request that cut the results passed to the LLM from 10 to 3 "to save tokens" dropped recall@10 from 0.676 to 0.622 and was blocked. _[Add the PR link and screenshot here.]_

## Limitations

- **One gold passage per question.** The biggest weakness, as the error analysis shows. `golden_v2` should allow several correct spans.
- **Duplicate and repeated text in the corpus:** the same judgment appears twice, and boilerplate paragraphs repeat across land acquisition cases. Deduplicating would make scores more meaningful.
- **Running page headers** ("SUPREME COURT REPORTS [2023] …", case names) remain inside the text and add noise. Removing them means re-ingesting and remapping the golden set.
- **Small per-category samples** (11–18 test questions), and a 30-question dev split too small to choose chunk size reliably.
- **One annotator**, with LLM-drafted questions. No inter-annotator agreement was measured.
- **The generation eval** has not been run yet, so no faithfulness numbers are reported.

## Future work

Multi-span gold labels and corpus deduplication (`golden_v2`); confirming dense + reranker on fresh questions; a stronger embedding model (`bge-base`/`bge-large`) and a reranker trained beyond web search; paragraph-aware chunking; query rewriting for plain-language questions; running the generation eval, with the judge calibrated on hand labels.

## Reproduce

Everything runs in Google Colab on a free T4 GPU, using `notebooks/groundtruth_full.ipynb`. It covers setup, labelling, experiments, the gate and the GitHub push.

Locally (Python 3.10+):

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src
pytest -q                        # unit tests + the regression gate
python -m groundtruth evaluate --config configs/production.yaml
python -m groundtruth compare --baseline dense_800 --runs dense_400 dense_1600 bm25_400 hybrid_400 hybrid_rerank_400
python -m groundtruth failures --run hybrid_rerank_400
```

## Data and attribution

Judgments from the [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/) open dataset ([source repository](https://github.com/vanga/indian-supreme-court-judgments)), licensed **CC-BY-4.0**. This project uses a cleaned 700-document sample of the 2023 English judgments. Changes made: text extracted from PDF, page furniture removed, and split words re-joined.

## Project documents

`docs/PRD.md` · `docs/DESIGN.md` · `docs/ARCHITECTURE.md` · `docs/DATABASE.md` · `docs/API.md` · `docs/IMPLEMENTATION_PLAN.md` · `AGENTS.md`

Tests use a small **fictional** corpus in `tests/fixtures/` and an offline LSA embedding stand-in. Neither is ever reported as a result.
