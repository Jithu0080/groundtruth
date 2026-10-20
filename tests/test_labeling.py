import pytest

from groundtruth import GroundtruthError
from groundtruth.golden import assign_splits, locate_answer

DOC = "Intro text. The court held that notice is mandatory before prosecution.  It also said more."
CAND = {"doc_id": "D", "start": 12, "end": len(DOC), "text": DOC[12:]}


def test_locate_exact_and_whitespace_insensitive():
    s, e = locate_answer(CAND, "notice is mandatory before prosecution")
    assert DOC[s:e] == "notice is mandatory before prosecution"
    s, e = locate_answer(CAND, "  prosecution.\n  It   also ")
    assert DOC[s:e] == "prosecution.  It also"


def test_locate_empty_means_whole_passage_and_missing_raises():
    assert locate_answer(CAND, "") == (CAND["start"], CAND["end"])
    with pytest.raises(GroundtruthError):
        locate_answer(CAND, "this sentence is not there")


def _recs():
    cats = ["holding"] * 20 + ["factual"] * 20 + ["statute"] * 15 + ["procedural"] * 15 + ["paraphrase"] * 30
    rs = [{"qid": f"q{i:03d}", "category": c} for i, c in enumerate(cats)]
    return rs + [{"qid": "u1", "category": "unanswerable", "answerable": False}]


def test_assign_splits_counts_stratified_and_deterministic():
    out = assign_splits(_recs(), n_dev=30, seed=0)
    dev = [r for r in out if r["split"] == "dev"]
    assert len(dev) == 30
    assert {r["category"] for r in dev} == {"holding", "factual", "statute", "procedural", "paraphrase"}
    assert next(r for r in out if r["qid"] == "u1")["split"] == "test"
    assert out == assign_splits(_recs(), n_dev=30, seed=0)
    assert out != assign_splits(_recs(), n_dev=30, seed=1)


def test_assign_splits_bad_n():
    with pytest.raises(GroundtruthError):
        assign_splits(_recs(), n_dev=1000)


def test_cli_finalize_writes_once(tmp_path, capsys):
    import json

    from conftest import FIXTURES
    from groundtruth.cli import main

    drafts = tmp_path / "drafts.jsonl"
    rows = [json.loads(line) for line in (FIXTURES / "golden_mini.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("split")
    drafts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "golden_v1.jsonl"
    args = [
        "finalize",
        "--drafts",
        str(drafts),
        "--out",
        str(out),
        "--corpus",
        str(FIXTURES / "mini_corpus"),
        "--dev",
        "3",
    ]
    assert main(args) == 0
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert sum(r["split"] == "dev" for r in written) == 3
    assert main(args) == 1  # refuses to overwrite a frozen golden set
    assert "immutable" in capsys.readouterr().err
