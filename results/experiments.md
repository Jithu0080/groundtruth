## Results

| Config | Recall@10 | Recall@5 | MRR | nDCG@10 | Avg chars in top 10 | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| dense_800 (baseline) | 0.513 | 0.419 | 0.291 | 0.344 | 7916 | 27 |
| dense_400 | 0.540 | 0.460 | 0.329 | 0.380 | 3952 | 31 |
| dense_1600 | 0.622 | 0.460 | 0.302 | 0.376 | 15825 | 20 |
| bm25_400 | 0.446 | 0.351 | 0.276 | 0.315 | 3956 | 7 |
| hybrid_400 | 0.581 | 0.486 | 0.324 | 0.385 | 3955 | 48 |
| hybrid_rerank_400 | 0.676 | 0.595 | 0.491 | 0.535 | 3958 | 132 |
| dense_rerank_400 | **0.703** | **0.608** | **0.500** | **0.549** | 3958 | 134 |

## recall@10 by category

| Category | dense_800 | dense_400 | dense_1600 | bm25_400 | hybrid_400 | hybrid_rerank_400 | dense_rerank_400 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factual | 0.812 | 0.812 | 0.875 | 0.625 | 0.812 | 0.812 | 0.812 |
| holding | 0.533 | 0.467 | 0.533 | 0.400 | 0.467 | 0.600 | 0.733 |
| paraphrase | 0.357 | 0.429 | 0.643 | 0.357 | 0.500 | 0.571 | 0.571 |
| procedural | 0.364 | 0.727 | 0.455 | 0.455 | 0.636 | 0.727 | 0.727 |
| statute | 0.444 | 0.333 | 0.556 | 0.389 | 0.500 | 0.667 | 0.667 |

## recall@10 vs dense_800 (paired bootstrap, 95% CI)

- dense_400: +0.027 (95% CI -0.095 to +0.149), not distinguishable from noise
- dense_1600: +0.108 (95% CI +0.000 to +0.216), not distinguishable from noise
- bm25_400: -0.068 (95% CI -0.203 to +0.068), not distinguishable from noise
- hybrid_400: +0.068 (95% CI -0.054 to +0.189), not distinguishable from noise
- hybrid_rerank_400: +0.162 (95% CI +0.041 to +0.284), likely real
- dense_rerank_400: +0.189 (95% CI +0.068 to +0.311), likely real

_Golden set: data/golden_v1.jsonl, split: test, 74 queries._
