# Groundtruth

**An evaluation harness that proves whether a retrieval change helped a legal research assistant, and blocks the ones that hurt.**

> Headline result: _fill in after your real run: "Hybrid + reranker raises recall@10 from X to Y on 70 held-out legal queries (95% CI +a to +b)."_

## Results

_Replace with `results/experiments.md` after running the experiments. Every number must come from a committed config and `data/golden_v1.jsonl`._

| Config | Recall@10 | Recall@5 | MRR | nDCG@10 | Avg chars in top 10 | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| dense_400 | | | | | | |
| dense_800 (baseline) | | | | | | |
| dense_1600 | | | | | | |
| bm25_800 | | | | | | |
| hybrid_800 | | | | | | |
| hybrid_rerank_800 | | | | | | |

**Recommendation.** _3–5 sentences: the decision, the evidence with its confidence interval, the cost (latency, characters retrieved), and what you would ship under a tighter latency budget._

## The problem

A 60-attorney firm runs a research assistant over 20 years of case files. People keep changing chunk size, embedding models and rerankers, and nobody can say whether last month's version was better. A missed precedent is a malpractice risk, so opinion is not evidence. Groundtruth turns each change into numbers.

## How it works

1. **Golden set:** 100 human-reviewed query–passage pairs from real judgments, in 5 categories, split 30 dev / 70 test. Gold passages are stored as **character spans**, so they stay valid under any chunking.
2. **Retrieval metrics:** recall@5, recall@10, MRR, nDCG@10, per category. A chunk counts as a hit when it covers at least 50% of a gold span, and each gold span earns credit once.
3. **Generation metrics,** kept separate: faithfulness (claim-level LLM judge) and answer relevance (1–5 rubric), plus the refusal rate on unanswerable questions.
4. **Experiments as configs:** chunk size, dense vs BM25 vs hybrid (reciprocal rank fusion), and a cross-encoder reranker on or off. Comparisons use a paired bootstrap 95% CI.
5. **CI gate:** every pull request re-runs `configs/production.yaml` on the test split. It fails if recall@10 drops more than 1 point, or any category drops more than 5 points. _Link your deliberately failing PR here._

## Dataset

_Source, license, number of documents, legal area. Example: N Supreme Court of India judgments on cheque dishonour, from [source], under [license]._

## Error analysis

_The five queries that still fail (`python -m groundtruth failures --run <best>`), each with its root cause: vocabulary mismatch, chunk boundary, multi-hop, ambiguous query or label error._

| Query ID | Category | What happened | Root cause | Next fix to try |
| --- | --- | --- | --- | --- |

## Limitations

One legal domain; 100 queries labelled by one person; LLM-judge bias (calibrated on 20 hand labels); exact search only (no approximate-index loss measured).

---

## Run it

Works on Windows, macOS and Linux with Python 3.10+. Commands below are for Windows (Anaconda Prompt or PowerShell); on macOS/Linux use `source .venv/bin/activate`.

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements.lock     # record exact versions for reproducibility
set PYTHONPATH=src                 # PowerShell: $env:PYTHONPATH="src"
pytest -q                          # 56 passed, 1 skipped (the gate, until you add real data)
```

The first real run downloads `BAAI/bge-small-en-v1.5` (~130 MB) and the reranker (~90 MB) from Hugging Face.

### 2. Build the corpus

Put 300–1,000 judgment PDFs or TXT files from one legal area in `raw/`. Check the source's license first.

```bash
python -m groundtruth ingest --src raw --out data/corpus
```

### 3. Build the golden set (the most important step)

```bash
python -m groundtruth sample --n 150          # writes data/candidates.jsonl
```

For each candidate passage, write the question a lawyer would ask **in different words from the passage**. Keep the best 100, aim for 15–20 per category, and mark 30 as `dev` and 70 as `test`. Save one JSON object per line to `data/golden_v1.jsonl`:

```json
{"qid": "q017", "query": "Is notice to the drawer mandatory before prosecution for a bounced cheque?", "category": "holding", "relevant": [{"doc_id": "SC_2019_1432", "start": 18840, "end": 19410}], "split": "test", "reviewed_by": "your-name"}
```

Optionally add about 10 unanswerable probes (`"answerable": false, "relevant": []`) for the generation eval. Then validate:

```bash
python -m groundtruth validate
```

Once frozen, never edit it. A fix becomes `golden_v2.jsonl`.

### 4. Run the experiments

```bash
python -m groundtruth evaluate --config configs/experiments/dense_400.yaml
python -m groundtruth evaluate --config configs/experiments/dense_800.yaml
python -m groundtruth evaluate --config configs/experiments/dense_1600.yaml
```

Set `chunk_size` in `bm25_800`, `hybrid_800` and `hybrid_rerank_800` to the chunk-size winner (rename the files if it changes), then:

```bash
python -m groundtruth evaluate --config configs/experiments/bm25_800.yaml
python -m groundtruth evaluate --config configs/experiments/hybrid_800.yaml
python -m groundtruth evaluate --config configs/experiments/hybrid_rerank_800.yaml
python -m groundtruth compare --baseline dense_800 --runs dense_400 dense_1600 bm25_800 hybrid_800 hybrid_rerank_800
python -m groundtruth failures --run hybrid_rerank_800
```

Tune on `--split dev` while developing; report numbers from the default `test` split.

### 5. Generation eval (optional, costs API calls)

Copy `.env.example` values into your environment. For a free local option, use Ollama with `GT_LLM_PROVIDER=openai` and `OPENAI_BASE_URL=http://localhost:11434/v1`.

```bash
python -m groundtruth generate-eval --config configs/production.yaml --limit 20
```

Hand-label those 20 answers yourself and report how often the judge agrees with you.

### 6. Turn on the CI gate

1. Copy the winning settings into `configs/production.yaml`, then run `python -m groundtruth evaluate --config configs/production.yaml` and `python -m groundtruth set-baseline --run production`.
2. Commit `data/`, `results/baseline.json` and `results/experiments.md`, and push to GitHub.
3. Under Settings → Branches, protect `main` and require the `retrieval-eval` check.
4. Open a PR that copies `configs/experiments/bad_tiny_chunks.yaml` into `production.yaml`. Screenshot the red check and link it above.

## Project docs

`docs/PRD.md` · `docs/DESIGN.md` · `docs/ARCHITECTURE.md` · `docs/DATABASE.md` · `docs/API.md` · `docs/IMPLEMENTATION_PLAN.md` · `AGENTS.md`

Tests use a small **fictional** corpus in `tests/fixtures/` and an offline LSA embedding stand-in. Neither is ever reported as a result.
