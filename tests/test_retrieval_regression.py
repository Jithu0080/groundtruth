"""THE CI GATE. Runs the production config on the real golden set and compares with the baseline.

Skipped (never silently passed) until data/golden_v1.jsonl, data/corpus and results/baseline.json exist.
"""

import json
from functools import lru_cache
from pathlib import Path

import pytest

from groundtruth.evaluate import run_retrieval_eval
from groundtruth.gate import check_regression

GOLDEN = Path("data/golden_v1.jsonl")
BASELINE = Path("results/baseline.json")
CORPUS = Path("data/corpus")

missing = [str(p) for p in (GOLDEN, BASELINE, CORPUS) if not p.exists()]
pytestmark = pytest.mark.skipif(bool(missing), reason=f"gate inactive, missing: {missing}")


@lru_cache(maxsize=1)
def _current() -> dict:
    return run_retrieval_eval("configs/production.yaml", GOLDEN, "test")


def test_recall_did_not_regress():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = check_regression(_current(), baseline)
    assert not problems, "retrieval regressed:\n" + "\n".join(problems)
