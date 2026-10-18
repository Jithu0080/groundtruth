# API — Groundtruth interfaces

Groundtruth exposes no HTTP API. Its interfaces are a command-line interface, internal Python module contracts and one outbound LLM integration.

## 1. CLI (`python -m groundtruth <command>`)

| Command | Purpose | Key arguments | Output |
| --- | --- | --- | --- |
| `ingest` | Clean PDFs/TXTs into the corpus | `--src DIR --out data/corpus` | `{doc_id}.txt`, `metadata.csv` |
| `sample` | Draw candidate passages for question writing | `--n 150 --seed 0` | `data/candidates.jsonl` |
| `validate` | Check the golden set | `--golden PATH` | Report; exit 1 on errors |
| `evaluate` | Run one config | `--config PATH --split test` | `results/runs/<name>/` |
| `compare` | Results table + bootstrap CI | `--runs A B ... --baseline A` | Markdown table to stdout and `results/experiments.md` |
| `failures` | Queries that miss gold at k | `--run NAME --k 10` | Markdown report |
| `generate-eval` | Answers + faithfulness + relevance | `--config PATH --split test --limit N` | `results/runs/<name>/generation.csv`, `generation_summary.json` |
| `set-baseline` | Copy a run's summary to `results/baseline.json` | `--run NAME` | baseline file |

Exit codes: 0 ok, 1 usage/validation error, 2 regression.

## 2. Internal contracts

- `Retriever.index(chunks: list[Chunk]) -> None`
- `Retriever.search(query: str, k: int) -> list[Hit]` where `Hit = (chunk: Chunk, score: float)`, sorted by score descending.
- `Reranker.rerank(query, hits, k) -> list[Hit]`
- `match_gold(chunks, gold_spans, min_cover=0.5) -> list[set[int]]` (gold spans each ranked chunk covers)
- Metric functions take that list and return floats in [0, 1].

## 3. Validation conventions

- Configs are validated on load: unknown keys rejected; `retriever ∈ {dense, bm25, hybrid}`; `overlap < chunk_size`; `k ≤ candidates`.
- Golden records validated as in DATABASE.md; all errors collected and reported together.

## 4. External integration: LLM provider

| Setting | Env variable | Default |
| --- | --- | --- |
| Provider | `GT_LLM_PROVIDER` | `anthropic` (`openai` = any OpenAI-compatible endpoint, incl. Ollama) |
| Generator model | `GT_LLM_MODEL` | none — must be set |
| Judge model | `GT_JUDGE_MODEL` | falls back to `GT_LLM_MODEL` with a warning |
| Anthropic key | `ANTHROPIC_API_KEY` | — |
| OpenAI-compatible key | `OPENAI_API_KEY` | — (Ollama ignores it) |
| OpenAI-compatible base URL | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |

- Endpoints used: Anthropic `POST /v1/messages`; OpenAI-compatible `POST {base}/chat/completions`.
- Request: one user message, temperature 0, bounded max tokens.
- Timeouts: 60 s per call.
- Retries: only on HTTP 429, 5xx and network timeouts; exponential backoff, max 3 attempts. Never on 4xx (a bad request will not fix itself).
- Judge responses must be JSON; a malformed response is recorded as `judge_error` and excluded from averages, and the error count is reported.

## 5. Rate limiting

Sequential calls with backoff on 429. `--limit` caps how many queries are sent, to control cost.

## 6. Webhooks

None.

## 7. Privacy and failure handling

- API keys are read only from the environment, never written to files or logs; error messages show the HTTP status and a truncated body, never headers.
- Only public corpus text and questions are sent to the LLM.
- A failed generation call for one query is recorded and the run continues; the summary reports how many failed.
