import pytest

from groundtruth import GroundtruthError
from groundtruth.chunking import chunk_document
from groundtruth.corpus import clean_pages, corpus_hash, ingest, load_corpus, make_doc_id

TEXT = " ".join(f"word{i}" for i in range(600))


def test_offsets_match_text_and_cover_everything():
    chunks = chunk_document("D", TEXT, chunk_size=200, overlap=40)
    covered = [False] * len(TEXT)
    for c in chunks:
        assert TEXT[c.start : c.end] == c.text
        assert c.chunk_id == f"D:{c.start}"
        for i in range(c.start, c.end):
            covered[i] = True
    assert all(covered)


def test_chunks_overlap_and_respect_size():
    chunks = chunk_document("D", TEXT, chunk_size=200, overlap=40)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start < a.end  # overlap
        assert b.start > a.start  # progress
    assert all(len(c.text) <= 200 for c in chunks)


def test_words_not_cut():
    for c in chunk_document("D", TEXT, chunk_size=200, overlap=40)[:-1]:
        assert TEXT[c.end] == " "


def test_short_and_spaceless_text():
    assert chunk_document("D", "short", 100, 10)[0].text == "short"
    long_token = "x" * 450  # no spaces: must still terminate
    chunks = chunk_document("D", long_token, 100, 10)
    assert chunks[-1].end == 450


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document("D", TEXT, 100, 100)


def _page(i: int, middle: str) -> str:
    body = [f"Body line {j} of page {i}." for j in range(8)]
    body[4] = middle
    return "\n".join(["SUPREME COURT REPORTS", "CASE NO. 12 OF 2019", *body, "Printed for research use", str(i)])


def test_clean_pages_removes_headers_footers_and_numbers():
    out = clean_pages([_page(i, f"about the appel-\nlant on page {i}.") for i in range(1, 5)])
    assert "SUPREME COURT REPORTS" not in out
    assert "CASE NO. 12" not in out
    assert "Printed for research use" not in out
    assert "appellant on page 3" in out
    assert "Body line 7 of page 4." in out
    assert not out.endswith("4")


def test_clean_pages_keeps_repeated_body_text():
    out = clean_pages([_page(i, "The appeal is dismissed.") for i in range(1, 5)])
    assert out.count("The appeal is dismissed.") == 4


def test_make_doc_id():
    assert make_doc_id("SC 2019 (1432)") == "SC_2019_1432"
    with pytest.raises(GroundtruthError):
        make_doc_id("()")


def test_ingest_txt_roundtrip(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    (src / "case one.txt").write_text("Holding: " + "the notice is mandatory. " * 20, encoding="utf-8")
    (src / "tiny.txt").write_text("too short", encoding="utf-8")
    n = ingest(src, tmp_path / "corpus")
    assert n == 1
    docs = load_corpus(tmp_path / "corpus")
    assert list(docs) == ["case_one"]
    assert (tmp_path / "corpus" / "metadata.csv").is_file()


def test_ingest_empty_folder(tmp_path):
    with pytest.raises(GroundtruthError):
        ingest(tmp_path, tmp_path / "out")


def test_corpus_hash_changes_with_content():
    assert corpus_hash({"a": "x"}) != corpus_hash({"a": "y"})
    assert corpus_hash({"a": "x", "b": "y"}) == corpus_hash({"b": "y", "a": "x"})


def test_margin_letters_removed():
    from groundtruth.corpus import clean_pages

    out = clean_pages(["A\nThe appeal is allowed.\nB\nC D E\nCosts follow.\nF G H the end"])
    assert out == "The appeal is allowed. Costs follow. the end"


def test_repair_split_words_uses_corpus_statistics():
    from groundtruth.corpus import repair_split_words

    common = "The forest officer recorded the facts and the appellants declared the land. " * 10
    docs = {
        "d1": common,
        "d2": "On fa cts, the For est land was declare d reserved and the appe llants were heard.",
        "d3": "He went in to the room and a part of the land was sold. " * 3 + "into apart " * 20,
    }
    out, n = repair_split_words(docs)
    assert out["d2"] == "On facts, the Forest land was declared reserved and the appellants were heard."
    assert n == 4
    assert out["d3"] == docs["d3"]  # "in to" / "a part": both halves are common words, never joined
    assert out["d1"] == docs["d1"]
