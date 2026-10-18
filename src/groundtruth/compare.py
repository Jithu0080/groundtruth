"""Compare runs: results table, per-category table, paired bootstrap CI, failure report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groundtruth import GroundtruthError
from groundtruth.corpus import load_corpus
from groundtruth.golden import load_golden

COLUMNS = [
    ("recall@10", "Recall@10", True),
    ("recall@5", "Recall@5", True),
    ("mrr", "MRR", True),
    ("ndcg@10", "nDCG@10", True),
    ("avg_chars_topk", "Avg chars in top 10", None),
    ("latency_p95_ms", "p95 latency (ms)", None),
]


def load_run(runs_dir: Path, name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    d = runs_dir / name
    if not (d / "summary.json").is_file():
        raise GroundtruthError(f"run {name!r} not found in {runs_dir} (run `groundtruth evaluate`)")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return summary, pd.read_csv(d / "per_query.csv")


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) and its 95% CI by resampling queries with replacement."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired bootstrap needs two equal, non-empty arrays")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float((b - a).mean()), float(low), float(high)


def compare_runs(runs_dir: str | Path, names: list[str], baseline: str, metric: str = "recall@10") -> str:
    rdir = Path(runs_dir)
    if baseline not in names:
        names = [baseline, *names]
    runs = {n: load_run(rdir, n) for n in names}
    splits = {s["split"] for s, _ in runs.values()}
    goldens = {s["golden"] for s, _ in runs.values()}
    if len(splits) > 1 or len(goldens) > 1:
        raise GroundtruthError("runs used different splits or golden sets; numbers are not comparable")

    out = ["## Results", "", "| Config | " + " | ".join(c[1] for c in COLUMNS) + " |"]
    out.append("| --- |" + " --- |" * len(COLUMNS))
    best = {key: max(s[key] for s, _ in runs.values()) for key, _, higher in COLUMNS if higher}
    for name, (s, _) in runs.items():
        cells = []
        for key, _, higher in COLUMNS:
            v = s[key]
            txt = f"{v:.3f}" if higher else f"{v:.0f}"
            cells.append(f"**{txt}**" if higher and v == best[key] and len(runs) > 1 else txt)
        label = f"{name} (baseline)" if name == baseline else name
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    cats = sorted({c for s, _ in runs.values() for c in s["by_category"]})
    out += ["", f"## {metric} by category", "", "| Category | " + " | ".join(runs) + " |"]
    out.append("| --- |" + " --- |" * len(runs))
    for cat in cats:
        vals = [s["by_category"].get(cat, {}).get(metric) for s, _ in runs.values()]
        out.append(f"| {cat} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")

    col = {"mrr": "rr"}.get(metric, metric)
    base_df = runs[baseline][1].set_index("qid")[col]
    out += ["", f"## {metric} vs {baseline} (paired bootstrap, 95% CI)", ""]
    for name, (_, df) in runs.items():
        if name == baseline:
            continue
        joined = pd.concat([base_df, df.set_index("qid")[col]], axis=1, join="inner").to_numpy()
        mean, lo, hi = paired_bootstrap(joined[:, 0], joined[:, 1])
        verdict = "likely real" if lo > 0 or hi < 0 else "not distinguishable from noise"
        out.append(f"- {name}: {mean:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), {verdict}")
    s0 = runs[baseline][0]
    out += ["", f"_Golden set: {s0['golden']}, split: {s0['split']}, {s0['n_queries']} queries._"]
    return "\n".join(out)


def failure_report(runs_dir: str | Path, name: str, corpus_dir: str, k: int = 10, show: int = 3) -> str:
    """Queries whose gold passages are not all in the top k, with gold text and top retrieved text."""
    summary, df = load_run(Path(runs_dir), name)
    docs = load_corpus(corpus_dir)
    golden = {q.qid: q for q in load_golden(summary["golden"], docs, "all")}
    col = f"recall@{k}"
    if col not in df:
        raise GroundtruthError(f"run has no {col} column; use k=5 or k=10")
    failing = df[df[col] < 1.0].sort_values([col, "rr"])
    out = [f"# Failures for {name}: {len(failing)} of {len(df)} queries miss gold in top {k}", ""]
    for _, row in failing.iterrows():
        q = golden[row["qid"]]
        out += [f"## {q.qid} [{q.category}] {col}={row[col]:.2f}", "", f"**Query:** {q.query}", ""]
        for g in q.relevant:
            snippet = docs[g["doc_id"]][g["start"] : g["end"]][:400]
            out.append(f"**Gold** ({g['doc_id']}:{g['start']}-{g['end']}): {snippet}")
        out.append("")
        for i, ref in enumerate(str(row["retrieved"]).split("|")[:show], start=1):
            doc_id, span = ref.rsplit(":", 1)
            s, e = (int(x) for x in span.split("-"))
            out.append(f"{i}. ({ref}) {docs[doc_id][s:e][:300]}")
        out += ["", "Root cause: _vocabulary mismatch / chunk boundary / multi-hop / ambiguous / label error_", ""]
    return "\n".join(out)
