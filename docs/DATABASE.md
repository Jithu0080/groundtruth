# DATABASE — Groundtruth data model

## 1. Technology

No database server (ARCHITECTURE §6). Data is stored as versioned files in the repository: plain text, CSV, JSONL and JSON. This keeps every number reproducible from a git commit. A database is unnecessary at 1,000 documents and 100 queries.

## 2. Entities

```
Document 1 ──< Chunk               (derived, never stored as truth)
Document 1 ──< GoldSpan >── 1 Query
Query    1 ──< QueryResult >── 1 Run
Run      1 ──  1 Summary
```

### Document — `data/corpus/{doc_id}.txt` + `data/corpus/metadata.csv`

| Column | Type | Constraint |
| --- | --- | --- |
| doc_id | string | PK; `^[A-Za-z0-9._-]+$`; stable forever, never renumbered |
| title | string | required |
| court | string | optional |
| year | integer | optional, 1900–2100 |
| source_file | string | original file name |

### Chunk — in memory only

| Field | Type | Constraint |
| --- | --- | --- |
| chunk_id | string | `{doc_id}:{start}`, unique per config |
| doc_id | string | FK → Document |
| start, end | integer | 0 ≤ start < end ≤ len(document) |
| text | string | document[start:end] |

### Query + GoldSpan — `data/golden_v1.jsonl` (one JSON object per line)

| Field | Type | Constraint |
| --- | --- | --- |
| qid | string | PK, unique |
| query | string | non-empty, unique (case-insensitive) |
| category | string | required; each category ≥ 15 queries (warning below) |
| relevant | list of {doc_id, start, end} | ≥ 1 item; doc_id exists; 0 ≤ start < end ≤ doc length |
| split | "dev" \| "test" | required |
| reviewed_by | string | required, non-empty |
| answerable | boolean | default true; false = unanswerable probe (generation eval only, `relevant` empty) |
| notes | string | optional |

### Candidate — `data/candidates.jsonl`

Sampled passages awaiting a question: `cid, doc_id, start, end, text`. Working file; not used by evaluation.

### QueryResult — `results/runs/<run>/per_query.csv`

qid, category, split, n_gold, recall@5, recall@10, recall@candidates, rr, ndcg@10, latency_ms, chars_topk, retrieved (top-k chunk ids, `|`-separated).

### Summary — `results/runs/<run>/summary.json` and `results/baseline.json`

```
{ "config": str, "golden": str, "split": str, "n_queries": int,
  "recall@5": float, "recall@10": float, "recall@candidates": float, "mrr": float, "ndcg@10": float,
  "latency_p50_ms": float, "latency_p95_ms": float, "avg_chars_topk": float,
  "by_category": { "<category>": { "n": int, "recall@10": float, "mrr": float, "ndcg@10": float } } }
```

## 3. Indexes (based on real queries)

- Golden validation looks up documents by `doc_id` → in-memory dict.
- Gold matching groups chunks by `doc_id` → in-memory dict.
- Embedding cache file keyed by model + chunk size + overlap + corpus hash.

## 4. Ownership and authorization

Single owner (the repo). Changes to the golden set and baseline go through pull requests and are reviewed with the code. Row-level security: not applicable.

## 5. Sensitive data

Public court judgments only. No client data, no personal accounts. Judgments can name individuals; they are public record, but do not add private annotations about people. Secrets are never stored in data files.

## 6. Migration strategy

- **Golden set:** immutable once frozen. A label fix creates `golden_v2.jsonl` and a README changelog line. Numbers are only compared within one version.
- **Corpus:** adding documents changes the corpus hash, which invalidates the embedding cache automatically; doc_ids of existing documents never change.
- **Baseline:** updated only in the same PR as the change that earned it.

## 7. Seed data

`tests/fixtures/mini_corpus/` + `tests/fixtures/golden_mini.jsonl`: a tiny handwritten corpus for tests only. Never reported as results.

## 8. Backup and recovery

Git is the backup. Push to GitHub. Cached embeddings are disposable and regenerated on demand.
