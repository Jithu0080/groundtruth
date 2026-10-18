# PRD — Groundtruth: Retrieval Evaluation Harness for a Legal Research Assistant

## 1. Problem

A 60-attorney firm runs a research assistant over 20 years of case files. Every week someone changes chunk size, swaps an embedding model or adds a reranker, and nobody can say whether retrieval got better or worse. A missed precedent is a malpractice risk, so opinion is not acceptable evidence.

## 2. Value proposition

Groundtruth turns "it feels better" into numbers. It measures retrieval and generation separately against a human-reviewed golden set, compares configurations with confidence intervals, and blocks any change that makes retrieval worse.

## 3. Target users

| User | Need |
| --- | --- |
| ML / AI engineer (primary) | Run experiments, see which config wins, ship with evidence |
| Reviewer of a pull request | See instantly whether a change regressed retrieval |
| Hiring manager / interviewer (portfolio audience) | Read the results table, recommendation and error analysis in under 2 minutes |

## 4. Core user journeys

1. **Ingest a corpus.** Engineer points the tool at a folder of judgments (PDF or TXT) and gets a cleaned corpus with stable document IDs.
2. **Build the golden set.** Engineer samples candidate passages, writes questions, reviews them and freezes `golden_v1.jsonl`. The validator rejects broken records.
3. **Evaluate one config.** Engineer runs one command with a YAML config and gets overall + per-category retrieval metrics and per-query rows.
4. **Compare configs.** Engineer runs the experiment configs and gets one results table with a bootstrap confidence interval against the baseline.
5. **Evaluate generation.** Engineer runs the generation eval and gets faithfulness and answer relevance, reported separately from retrieval.
6. **Gate a change.** A pull request that drops recall@10 by more than 1 point fails CI automatically.
7. **Analyse failures.** Engineer lists the queries that still fail, with the retrieved chunks, to write the error analysis.

## 5. User stories

- As an engineer, I want chunking, retriever and reranker set in a config file, so experiments need no code edits.
- As an engineer, I want gold passages stored as character spans, so the golden set survives any chunking change.
- As an engineer, I want metrics per query category, so I can see which kinds of question fail.
- As an engineer, I want a paired bootstrap interval, so I know whether a gain is real or noise.
- As a reviewer, I want a red CI check with the exact recall drop, so I can block a regression without rerunning anything.
- As an engineer, I want generation eval kept out of the per-PR gate, so CI stays fast, free and deterministic.

## 6. Functional requirements

| ID | Requirement |
| --- | --- |
| F1 | Ingest PDF and TXT files; strip repeated headers/footers and page numbers; write `{doc_id}.txt` + `metadata.csv` |
| F2 | Fixed-size character chunking with overlap; every chunk keeps `doc_id`, `start`, `end` |
| F3 | Retrievers: dense (sentence-transformers), BM25, hybrid (reciprocal rank fusion) |
| F4 | Optional cross-encoder reranker over N first-stage candidates |
| F5 | Golden set in JSONL with `qid, query, category, relevant[{doc_id,start,end}], split`; validator checks IDs, spans, duplicates, category counts |
| F6 | Candidate sampler that draws passages across many documents for human question writing |
| F7 | Metrics: recall@5, recall@10, recall@first-stage-candidates, MRR, nDCG@10; hit = chunk covers ≥ 50% of a gold span; a gold span earns credit once |
| F8 | Evaluation CLI: per-query CSV, summary JSON (overall + by category), latency p50/p95, average characters in top-k |
| F9 | Comparison CLI: markdown results table + paired bootstrap 95% CI against a baseline |
| F10 | Generation: top-5 chunks → LLM answer with numbered citations; refuses with a fixed sentence when unsupported |
| F11 | Generation eval: faithfulness (claim-level LLM judge) and answer relevance (1–5 rubric), reported separately |
| F12 | Failure report: queries with recall@10 < 1 and their top retrieved chunks |
| F13 | Regression tests + GitHub Actions workflow that fail when recall@10 drops > 0.01 or any category drops > 0.05 |
| F14 | Embedding cache keyed by model, chunk size and overlap |

## 7. Non-functional requirements

- **Deterministic:** same code + same data = same retrieval numbers (exact search, pinned versions, temperature 0 for LLM calls).
- **Runs on a laptop CPU:** a 500–1,000 document corpus indexes in minutes.
- **No secrets in the repo:** LLM keys only from environment variables.
- **Reproducible:** every reported number comes from a committed config and a committed golden set version.
- **Offline-testable:** unit tests run without network or model downloads.

## 8. Pages / screens

None. This is a command-line tool. Outputs are terminal summaries, CSV/JSON files, a markdown results table and the README.

## 9. Roles and permissions

Single-user local tool. No accounts. In CI, write access to `main` is controlled by GitHub branch protection (required check: `retrieval-eval`).

## 10. Data requirements

- Public legal corpus (license recorded in README).
- Golden set: 100 human-reviewed queries, ≥ 15 per category, 30 dev / 70 test.
- Results: per-query rows, summary JSON, `results/baseline.json` (the gate's reference).
- No personal or client data. See DATABASE.md.

## 11. Integrations

| Integration | Use | Required? |
| --- | --- | --- |
| Hugging Face model hub | Download embedding and reranker models | Yes (first run) |
| LLM API (Anthropic, or any OpenAI-compatible endpoint incl. local Ollama) | Generation + judge | Only for generation eval |
| GitHub Actions | CI gate | Yes for the gate |

## 12. Loading, empty and error states (CLI equivalents)

| State | Behaviour |
| --- | --- |
| Long-running indexing | Progress bar; cached embeddings reused on rerun |
| Empty corpus folder | Exit with a clear message naming the folder |
| Golden set invalid | Exit non-zero, list every bad record |
| Golden doc_id missing from corpus | Validation error, not a silent zero |
| Missing baseline.json | Regression test is skipped with a message, never silently passed |
| LLM key missing | Generation eval exits with the variable name to set |
| LLM returns malformed JSON | That item is recorded as a judge error and counted separately, not scored |

## 13. MVP scope

F1–F14, three experiments (chunk size, hybrid vs dense, reranker on/off), results table, written recommendation, error analysis of 5 queries, CI gate shown failing on a deliberately bad PR.

## 14. Out of scope

Web UI, user accounts, database server, approximate vector index, multiple legal domains, fine-tuning, query rewriting, production deployment of the assistant itself.

## 15. Success criteria (the ship gate)

1. Three configurations compared on the test split.
2. Results table + recommendation with the tradeoff stated.
3. CI gate demonstrated failing on a deliberately bad change.
4. Error analysis naming five failing queries and why.

## 16. Assumptions

- **A1:** Corpus is Indian Supreme Court judgments in one area of law (alternative: a US Caselaw Access Project slice).
- **A2:** The embedding model is `BAAI/bge-small-en-v1.5`; reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **A3:** You write and review the golden set; no other annotator is available.

## 17. Decisions requiring your approval

- **D1:** Which corpus and legal area (license must allow use).
- **D2:** Which LLM provider for generation/judge (paid API vs local Ollama).
- **D3:** Whether the corpus is committed to the repo (must be < 100 MB per file) or downloaded in CI.
