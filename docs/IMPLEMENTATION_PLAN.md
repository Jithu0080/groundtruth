# Implementation plan — Groundtruth

Each phase is small enough to implement and verify on its own. Status: ✅ = built and tested in this repo; 🧑 = needs you (real data or accounts).

| # | Phase | Goal | Files | Tests | Done when | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Setup | Runnable package, configs, tooling | `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `configs/` | import + lint | `python -m groundtruth --help` works | ✅ |
| 1 | Corpus | Ingest + clean PDFs/TXT; stable doc_ids | `corpus.py` | header/footer removal, loading | corpus folder + metadata.csv produced | ✅ code · 🧑 real corpus |
| 2 | Chunking | Char chunks with offsets | `chunking.py` | coverage, overlap, offsets match text | every char covered; spans exact | ✅ |
| 3 | Retrievers | Dense, BM25, hybrid RRF, reranker | `retrievers/*`, `pipeline.py` | BM25 ranking, RRF math, pipeline smoke | each returns sorted hits | ✅ |
| 4 | Golden set tooling | Sampler + validator | `golden.py` | validator catches each error type | invalid sets rejected | ✅ code · 🧑 write 100 queries |
| 5 | Metrics | recall@k, MRR, nDCG, gold matching | `metrics.py` | worked example by hand, double-count guard | tests match hand math | ✅ |
| 6 | Evaluate | Config → per-query + summary | `evaluate.py`, `cli.py` | end-to-end on fixture corpus | summary.json written | ✅ |
| 7 | Experiments | 6 configs + compare + bootstrap | `configs/experiments/`, `compare.py` | bootstrap sanity | table + CI printed | ✅ code · 🧑 real run |
| 8 | Generation eval | Grounded answers + judge | `llm.py`, `generation.py` | prompt/parse tests with a fake LLM | scores + error counts reported | ✅ code · 🧑 API key |
| 9 | CI gate | Regression test + workflow | `tests/test_retrieval_regression.py`, `.github/workflows/eval.yml` | regression test fails on a bad config | red check on a bad PR | ✅ code · 🧑 GitHub + branch protection |
| 10 | Write-up | Results, recommendation, error analysis | `README.md`, `results/` | numbers trace to runs | ship gate met | 🧑 after real runs |

**Recommended first task for you:** Phase 1 with real data — choose the corpus (PRD D1), put the files in `raw/`, and run `python -m groundtruth ingest --src raw --out data/corpus`.
