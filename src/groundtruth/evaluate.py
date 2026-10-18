"""Run one retrieval config over the golden set."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groundtruth.config import Config, load_config
from groundtruth.golden import load_golden
from groundtruth.metrics import match_gold, ndcg_at_k, recall_at_k, reciprocal_rank
from groundtruth.pipeline import Pipeline, build_pipeline

RUNS_DIR = Path("results/runs")


def evaluate_pipeline(pipe: Pipeline, golden_path: str | Path, split: str) -> pd.DataFrame:
    """One row per answerable golden query: metrics, latency and what was retrieved."""
    cfg = pipe.config
    queries = [q for q in load_golden(golden_path, pipe.docs, split) if q.answerable]
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        first, final = pipe.search(q.query)
        latency_ms = (time.perf_counter() - t0) * 1000
        final_m = match_gold([c for c, _ in final], q.relevant)
        first_m = match_gold([c for c, _ in first], q.relevant)
        n = len(q.relevant)
        rows.append(
            {
                "qid": q.qid,
                "category": q.category,
                "split": q.split,
                "n_gold": n,
                "recall@5": recall_at_k(final_m, n, 5),
                "recall@10": recall_at_k(final_m, n, 10),
                "recall@candidates": recall_at_k(first_m, n, cfg.candidates),
                "rr": reciprocal_rank(final_m),
                "ndcg@10": ndcg_at_k(final_m, n, 10),
                "latency_ms": latency_ms,
                "chars_topk": sum(len(c.text) for c, _ in final[: cfg.k]),
                "retrieved": "|".join(f"{c.doc_id}:{c.start}-{c.end}" for c, _ in final[: cfg.k]),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cfg: Config, golden_path: str | Path, split: str) -> dict[str, Any]:
    by_cat = {
        str(cat): {
            "n": int(len(g)),
            "recall@10": round(float(g["recall@10"].mean()), 4),
            "mrr": round(float(g["rr"].mean()), 4),
            "ndcg@10": round(float(g["ndcg@10"].mean()), 4),
        }
        for cat, g in df.groupby("category", sort=True)
    }
    return {
        "config": cfg.name,
        "golden": Path(golden_path).as_posix(),  # same string on Windows and Linux CI
        "split": split,
        "n_queries": int(len(df)),
        "recall@5": round(float(df["recall@5"].mean()), 4),
        "recall@10": round(float(df["recall@10"].mean()), 4),
        "recall@candidates": round(float(df["recall@candidates"].mean()), 4),
        "candidates": cfg.candidates,
        "mrr": round(float(df["rr"].mean()), 4),
        "ndcg@10": round(float(df["ndcg@10"].mean()), 4),
        "latency_p50_ms": round(float(np.percentile(df["latency_ms"], 50)), 1),
        "latency_p95_ms": round(float(np.percentile(df["latency_ms"], 95)), 1),
        "avg_chars_topk": round(float(df["chars_topk"].mean()), 1),
        "by_category": by_cat,
    }


def run_retrieval_eval(
    config_path: str | Path | Config,
    golden_path: str | Path = "data/golden_v1.jsonl",
    split: str = "test",
    docs: dict[str, str] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a config. Writes per_query.csv and summary.json when `out_dir` is given."""
    cfg = config_path if isinstance(config_path, Config) else load_config(config_path)
    pipe = build_pipeline(cfg, docs)
    df = evaluate_pipeline(pipe, golden_path, split)
    summary = summarize(df, cfg, golden_path, split)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_query.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(s: dict[str, Any]) -> str:
    lines = [
        f"config: {s['config']}   split: {s['split']}   queries: {s['n_queries']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recall@10 | {s['recall@10']:.3f} |",
        f"| Recall@5 | {s['recall@5']:.3f} |",
        f"| Recall@{s['candidates']} (first stage) | {s['recall@candidates']:.3f} |",
        f"| MRR | {s['mrr']:.3f} |",
        f"| nDCG@10 | {s['ndcg@10']:.3f} |",
        f"| Avg chars in top k | {s['avg_chars_topk']:.0f} |",
        f"| Latency p50 / p95 (ms) | {s['latency_p50_ms']:.0f} / {s['latency_p95_ms']:.0f} |",
        "",
        "| Category | n | Recall@10 | MRR | nDCG@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['n']} | {v['recall@10']:.3f} | {v['mrr']:.3f} | {v['ndcg@10']:.3f} |")
    return "\n".join(lines)
