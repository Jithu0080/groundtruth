"""Command-line interface: python -m groundtruth <command> [options]."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from groundtruth import GroundtruthError
from groundtruth.evaluate import RUNS_DIR

BASELINE = Path("results/baseline.json")


def _cmd_ingest(a: argparse.Namespace) -> int:
    from groundtruth.corpus import ingest

    n = ingest(a.src, a.out)
    print(f"ingested {n} documents into {a.out}")
    return 0


def _cmd_sample(a: argparse.Namespace) -> int:
    from groundtruth.corpus import load_corpus
    from groundtruth.golden import sample_candidates

    cands = sample_candidates(load_corpus(a.corpus), n=a.n, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if len(cands) < a.n:
        print(f"warning: only {len(cands)} of {a.n} passages could be sampled (corpus too small or docs too short)")
    print(f"wrote {len(cands)} candidate passages to {a.out}; write one question per passage, then review")
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    from collections import Counter

    from groundtruth.corpus import load_corpus
    from groundtruth.golden import read_jsonl, validate_records

    records = read_jsonl(a.golden)
    rep = validate_records(records, load_corpus(a.corpus))
    for w in rep.warnings:
        print(f"warning: {w}")
    for e in rep.errors:
        print(f"error: {e}")
    splits = Counter(r.get("split") for r in records)
    cats = Counter(r.get("category") for r in records)
    print(f"{len(records)} records | splits {dict(splits)} | categories {dict(cats)}")
    print("OK" if rep.ok else f"INVALID: {len(rep.errors)} errors")
    return 0 if rep.ok else 1


def _cmd_evaluate(a: argparse.Namespace) -> int:
    from groundtruth.config import load_config
    from groundtruth.evaluate import format_summary, run_retrieval_eval

    cfg = load_config(a.config)
    if cfg.embedding_backend == "lsa":
        print("note: LSA test backend - do not report these numbers")
    summary = run_retrieval_eval(cfg, a.golden, a.split, out_dir=Path(a.runs_dir) / cfg.name)
    print(format_summary(summary))
    print(f"\nsaved to {Path(a.runs_dir) / cfg.name}")
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    from groundtruth.compare import compare_runs

    report = compare_runs(a.runs_dir, a.runs, a.baseline, a.metric)
    print(report)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nsaved to {a.out}")
    return 0


def _cmd_failures(a: argparse.Namespace) -> int:
    from groundtruth.compare import failure_report

    print(failure_report(a.runs_dir, a.run, a.corpus, k=a.k))
    return 0


def _cmd_generate_eval(a: argparse.Namespace) -> int:
    from groundtruth.config import load_config
    from groundtruth.generation import run_generation_eval
    from groundtruth.llm import LLMClient
    from groundtruth.pipeline import build_pipeline

    cfg = load_config(a.config)
    generator = LLMClient.from_env("GT_LLM_MODEL")
    judge = LLMClient.from_env("GT_JUDGE_MODEL")
    df, summary = run_generation_eval(build_pipeline(cfg), a.golden, a.split, generator, judge, a.limit)
    out = Path(a.runs_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "generation.csv", index=False)
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved to {out}")
    return 0


def _cmd_set_baseline(a: argparse.Namespace) -> int:
    src = Path(a.runs_dir) / a.run / "summary.json"
    if not src.is_file():
        raise GroundtruthError(f"no summary for run {a.run!r} at {src}")
    summary = json.loads(src.read_text(encoding="utf-8"))
    if summary.get("split") != "test":
        raise GroundtruthError("the baseline must come from the test split")
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, BASELINE)
    print(f"baseline set from {a.run}: recall@10 = {summary['recall@10']:.3f}. Commit it with the change.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groundtruth", description="Retrieval evaluation harness")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="clean PDFs/TXTs into the corpus folder")
    s.add_argument("--src", required=True)
    s.add_argument("--out", default="data/corpus")
    s.set_defaults(fn=_cmd_ingest)

    s = sub.add_parser("sample", help="sample candidate passages for writing golden questions")
    s.add_argument("--corpus", default="data/corpus")
    s.add_argument("--out", default="data/candidates.jsonl")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_sample)

    s = sub.add_parser("validate", help="validate the golden set against the corpus")
    s.add_argument("--golden", default="data/golden_v1.jsonl")
    s.add_argument("--corpus", default="data/corpus")
    s.set_defaults(fn=_cmd_validate)

    for name, fn, helptext in [
        ("evaluate", _cmd_evaluate, "evaluate one retrieval config"),
        ("generate-eval", _cmd_generate_eval, "generate answers and score faithfulness/relevance"),
    ]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True)
        s.add_argument("--golden", default="data/golden_v1.jsonl")
        s.add_argument("--split", default="test", choices=["dev", "test", "all"])
        s.add_argument("--runs-dir", default=str(RUNS_DIR))
        if name == "generate-eval":
            s.add_argument("--limit", type=int, default=None, help="cap the number of queries (cost control)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("compare", help="results table + bootstrap CI across runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--baseline", required=True)
    s.add_argument("--metric", default="recall@10", choices=["recall@10", "mrr", "ndcg@10"])
    s.add_argument("--runs-dir", default=str(RUNS_DIR))
    s.add_argument("--out", default="results/experiments.md")
    s.set_defaults(fn=_cmd_compare)

    s = sub.add_parser("failures", help="list queries that miss gold in the top k")
    s.add_argument("--run", required=True)
    s.add_argument("--k", type=int, default=10, choices=[5, 10])
    s.add_argument("--corpus", default="data/corpus")
    s.add_argument("--runs-dir", default=str(RUNS_DIR))
    s.set_defaults(fn=_cmd_failures)

    s = sub.add_parser("set-baseline", help="make a test-split run the CI gate's reference")
    s.add_argument("--run", required=True)
    s.add_argument("--runs-dir", default=str(RUNS_DIR))
    s.set_defaults(fn=_cmd_set_baseline)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except GroundtruthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
