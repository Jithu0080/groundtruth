"""End-to-end on the fictional fixture corpus (offline). Checks plumbing, not quality."""

import json

import pytest

from conftest import FIXTURES
from groundtruth.cli import main
from groundtruth.compare import compare_runs, failure_report, paired_bootstrap
from groundtruth.config import load_config
from groundtruth.evaluate import run_retrieval_eval
from groundtruth.gate import check_regression

GOLDEN = FIXTURES / "golden_mini.jsonl"
CFG = FIXTURES / "configs"


@pytest.mark.parametrize("name", ["smoke_bm25", "smoke_dense", "smoke_hybrid"])
def test_each_retriever_runs(name, tmp_path):
    s = run_retrieval_eval(CFG / f"{name}.yaml", GOLDEN, "test", out_dir=tmp_path / name)
    assert s["n_queries"] == 8  # 9 test queries minus 1 unanswerable probe
    for key in ["recall@5", "recall@10", "mrr", "ndcg@10", "recall@candidates"]:
        assert 0.0 <= s[key] <= 1.0
    assert s["recall@5"] <= s["recall@10"] <= s["recall@candidates"]
    assert (tmp_path / name / "per_query.csv").is_file()
    assert set(s["by_category"]) == {"holding", "factual", "procedural"}


def test_bm25_finds_lexical_fixture_queries(tmp_path):
    s = run_retrieval_eval(CFG / "smoke_bm25.yaml", GOLDEN, "test")
    assert s["recall@10"] >= 0.8  # tiny corpus: BM25 should find nearly everything


def test_compare_and_failures(tmp_path):
    for n in ["smoke_bm25", "smoke_hybrid"]:
        run_retrieval_eval(CFG / f"{n}.yaml", GOLDEN, "test", out_dir=tmp_path / n)
    report = compare_runs(tmp_path, ["smoke_hybrid"], baseline="smoke_bm25")
    assert "smoke_bm25 (baseline)" in report
    assert "95% CI" in report
    fr = failure_report(tmp_path, "smoke_bm25", str(FIXTURES / "mini_corpus"), k=5)
    assert fr.startswith("# Failures for smoke_bm25")


def test_bootstrap_detects_clear_gain_and_noise():
    import numpy as np

    a = np.zeros(50)
    mean, lo, hi = paired_bootstrap(a, np.ones(50))
    assert mean == 1.0 and lo > 0
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 2, 50).astype(float)
    _, lo, hi = paired_bootstrap(noise, rng.permutation(noise))
    assert lo < 0 < hi


def test_gate_fails_on_a_bad_config(tmp_path):
    """The CI gate's logic, demonstrated offline: tiny chunks break gold coverage."""
    from dataclasses import replace

    good = run_retrieval_eval(CFG / "smoke_bm25.yaml", GOLDEN, "test")
    bad_cfg = replace(load_config(CFG / "smoke_bm25.yaml"), name="bad", chunk_size=50, overlap=0)
    bad = run_retrieval_eval(bad_cfg, GOLDEN, "test")
    assert check_regression(good, good) == []
    problems = check_regression(bad, good)
    assert any("recall@10 fell" in p for p in problems), problems


def test_gate_refuses_mismatched_golden():
    s = {"golden": "a", "split": "test", "recall@10": 1.0, "by_category": {}}
    assert check_regression({**s, "golden": "b"}, s)


def test_cli_evaluate_and_set_baseline(tmp_path, capsys, monkeypatch):
    runs = tmp_path / "runs"
    rc = main(["evaluate", "--config", str(CFG / "smoke_bm25.yaml"), "--golden", str(GOLDEN), "--runs-dir", str(runs)])
    assert rc == 0
    assert "Recall@10" in capsys.readouterr().out
    monkeypatch.setattr("groundtruth.cli.BASELINE", tmp_path / "baseline.json")
    assert main(["set-baseline", "--run", "smoke_bm25", "--runs-dir", str(runs)]) == 0
    assert json.loads((tmp_path / "baseline.json").read_text())["config"] == "smoke_bm25"


def test_cli_reports_errors_cleanly(capsys):
    assert main(["evaluate", "--config", "does/not/exist.yaml"]) == 1
    assert "config file not found" in capsys.readouterr().err
    assert main(["validate", "--golden", str(GOLDEN), "--corpus", str(FIXTURES / "mini_corpus")]) == 0


def test_config_validation(tmp_path):
    from groundtruth import GroundtruthError

    bad = tmp_path / "bad.yaml"
    bad.write_text("retriever: magic\nchunk_size: 100\noverlap: 200\n", encoding="utf-8")
    with pytest.raises(GroundtruthError) as e:
        load_config(bad)
    assert "retriever" in str(e.value) and "overlap" in str(e.value)
    typo = tmp_path / "typo.yaml"
    typo.write_text("chunksize: 100\n", encoding="utf-8")
    with pytest.raises(GroundtruthError):
        load_config(typo)


def test_summary_golden_path_is_platform_independent():
    """A baseline made on Windows must match a CI run on Linux."""
    from pathlib import PureWindowsPath

    import pandas as pd

    from groundtruth.evaluate import summarize

    df = pd.DataFrame(
        [
            {
                "category": "c",
                "recall@5": 1,
                "recall@10": 1,
                "recall@candidates": 1,
                "rr": 1,
                "ndcg@10": 1,
                "latency_ms": 1,
                "chars_topk": 1,
            }
        ]
    )
    cfg = load_config(CFG / "smoke_bm25.yaml")
    win = summarize(df, cfg, PureWindowsPath("data\\golden_v1.jsonl"), "test")
    assert win["golden"] == "data/golden_v1.jsonl"
