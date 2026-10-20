import json

import pytest

from groundtruth import GroundtruthError
from groundtruth.labeling import LabelStore, wording_overlap

TEXT = "The court held that a demand notice to the drawer is mandatory before any prosecution is launched."


def _cands(tmp_path):
    p = tmp_path / "candidates.jsonl"
    rows = [
        {"cid": "c001", "doc_id": "D1", "start": 100, "end": 100 + len(TEXT), "text": TEXT},
        {"cid": "c002", "doc_id": "D2", "start": 0, "end": len(TEXT), "text": TEXT},
        {"cid": "c003", "doc_id": "D3", "start": 0, "end": len(TEXT), "text": TEXT},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_overlap_flags_copied_wording():
    assert wording_overlap("Is a demand notice to the drawer mandatory before prosecution?", TEXT) > 0.6
    assert wording_overlap("Must a cheque issuer be warned before a criminal case?", TEXT) < 0.6


def test_save_skip_resume(tmp_path):
    drafts = tmp_path / "golden_drafts.jsonl"
    store = LabelStore(_cands(tmp_path), drafts, "jithin")
    c = store.next_candidate()
    rec = store.save(
        c,
        "Must a cheque issuer be warned before a criminal case?",
        "holding",
        "a demand notice to the drawer is mandatory",
    )
    assert rec["qid"] == "q001"
    s, e = rec["relevant"][0]["start"], rec["relevant"][0]["end"]
    assert s == 100 + TEXT.index("a demand notice") and e - s == len("a demand notice to the drawer is mandatory")
    store.skip(store.next_candidate())
    again = LabelStore(_cands(tmp_path), drafts, "jithin")  # simulates a Colab reconnect
    assert again.next_candidate()["cid"] == "c003"
    assert again.progress()["holding"] == 1


def test_save_rejects_bad_input(tmp_path):
    store = LabelStore(_cands(tmp_path), tmp_path / "d.jsonl", "me")
    c = store.next_candidate()
    with pytest.raises(GroundtruthError):
        store.save(c, "short?", "holding")
    with pytest.raises(GroundtruthError):
        store.save(c, "A perfectly fine question here?", "nonsense")
    with pytest.raises(GroundtruthError):
        store.save(c, "A perfectly fine question here?", "holding", "text that is not in the passage")
    store.save(c, "A perfectly fine question here?", "holding")
    with pytest.raises(GroundtruthError):
        store.save(store.next_candidate(), "a perfectly  fine question here?", "factual")
    with pytest.raises(GroundtruthError):
        LabelStore(_cands(tmp_path), tmp_path / "d.jsonl", "  ")


def test_unanswerable_then_finalize_validates(tmp_path):
    from groundtruth.golden import assign_splits, validate_records

    store = LabelStore(_cands(tmp_path), tmp_path / "d.jsonl", "me")
    store.save(store.next_candidate(), "Must a cheque issuer be warned before a criminal case?", "holding")
    assert store.add_unanswerable(["What is the limitation period for defamation?", ""]) == 1
    docs = {"D1": "x" * 100 + TEXT, "D2": TEXT, "D3": TEXT}
    recs = assign_splits(store.drafts, n_dev=0)
    rep = validate_records(recs, docs)
    assert rep.ok, rep.errors
