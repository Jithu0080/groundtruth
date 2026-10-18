import pytest

from conftest import FIXTURES
from groundtruth import GroundtruthError
from groundtruth.corpus import load_corpus
from groundtruth.golden import load_golden, read_jsonl, sample_candidates, validate_records

DOCS = load_corpus(FIXTURES / "mini_corpus")
GOOD = {
    "qid": "q1",
    "query": "Is notice mandatory?",
    "category": "holding",
    "relevant": [{"doc_id": "FIX_001_cheque", "start": 10, "end": 90}],
    "split": "test",
    "reviewed_by": "me",
}


def _errors(records):
    return validate_records(records, DOCS).errors


def test_fixture_golden_is_valid():
    rep = validate_records(read_jsonl(FIXTURES / "golden_mini.jsonl"), DOCS)
    assert rep.ok, rep.errors


def test_good_record_passes():
    assert _errors([GOOD]) == []


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"relevant": [{"doc_id": "NOPE", "start": 0, "end": 50}]}, "not in corpus"),
        ({"relevant": [{"doc_id": "FIX_001_cheque", "start": 50, "end": 10}]}, "outside document"),
        ({"relevant": [{"doc_id": "FIX_001_cheque", "start": 0, "end": 10**7}]}, "outside document"),
        ({"relevant": []}, "at least one relevant"),
        ({"split": "train"}, "split"),
        ({"reviewed_by": " "}, "reviewed_by"),
        ({"query": ""}, "empty query"),
    ],
)
def test_each_error_type_is_caught(change, fragment):
    errs = _errors([{**GOOD, **change}])
    assert any(fragment in e for e in errs), errs


def test_duplicates_caught():
    errs = _errors([GOOD, {**GOOD, "query": "  is NOTICE mandatory? "}])
    assert any("duplicate qid" in e for e in errs)
    assert any("same query text" in e for e in errs)


def test_missing_fields():
    assert any("missing fields" in e for e in _errors([{"qid": "x"}]))


def test_unanswerable_rules():
    ok = {**GOOD, "answerable": False, "relevant": []}
    assert _errors([ok]) == []
    assert _errors([{**ok, "relevant": GOOD["relevant"]}])


def test_load_golden_raises_and_filters(tmp_path):
    test_split = load_golden(FIXTURES / "golden_mini.jsonl", DOCS, "test")
    assert all(q.split == "test" for q in test_split)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"qid": "x"}\n', encoding="utf-8")
    with pytest.raises(GroundtruthError):
        load_golden(bad, DOCS)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(GroundtruthError):
        read_jsonl(broken)


def test_sample_candidates_spread_and_exact():
    cands = sample_candidates(DOCS, n=6, passage_chars=200, max_per_doc=1, seed=1)
    assert len({c["doc_id"] for c in cands}) == len(cands)
    for c in cands:
        assert DOCS[c["doc_id"]][c["start"] : c["end"]] == c["text"]
