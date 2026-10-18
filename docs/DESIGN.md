# DESIGN — Groundtruth

Groundtruth has no graphical interface (PRD §8). Its "design" is how results are presented to two readers: the engineer at the terminal and the hiring manager skimming the README. These rules keep that output consistent.

## 1. Principles

1. **The number first.** Every output leads with the headline metric (recall@10), then detail.
2. **Separate what is separate.** Retrieval and generation metrics never share a table.
3. **Show the cost beside the gain.** Recall is always printed next to characters retrieved and p95 latency.
4. **No false precision.** Metrics to 3 decimals, latency to whole milliseconds, percentages to 1 decimal.
5. **Plain text that survives copy-paste.** Markdown tables only; no colour-only meaning, no emoji.

## 2. Terminal output

- One summary block per run: config name, split, query count, then a metrics table.
- Progress bars only for steps over ~2 seconds (embedding, LLM calls).
- Errors: one line stating what failed and what to do, e.g. `golden set: q017 doc_id SC_2019_1432 not in corpus`.
- Exit codes: 0 success, 1 validation/usage error, 2 regression detected.

## 3. Tables (reusable formats)

**Results table** — columns in this order: Config · Recall@10 · Recall@5 · MRR · nDCG@10 · Avg chars in top 10 · p95 latency (ms). Baseline row marked `(baseline)`. Best value per metric is bolded.

**Per-category table** — rows = categories, columns = configs, cell = recall@10.

**Comparison line** — `hybrid vs baseline: recall@10 +0.043 (95% CI +0.012 to +0.074)`.

**Generation table** — Faithfulness · Answer relevance · Refusal rate (unanswerable) · Judge errors.

## 4. README layout

1. One-line summary with the headline number.
2. Results table (above the fold).
3. Recommendation paragraph.
4. Problem, dataset, golden set, metrics, error analysis, CI gate, limitations, how to run.

## 5. Accessibility

- Tables have header rows; no information carried by colour or position alone.
- Charts (if added) include the underlying numbers in a table.
- Terminal output avoids box-drawing characters that break screen readers.
