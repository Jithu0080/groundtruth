import math

from groundtruth.chunking import Chunk
from groundtruth.retrievers.bm25 import BM25Retriever
from groundtruth.retrievers.dense import DenseRetriever, LSAEncoder
from groundtruth.retrievers.hybrid import rrf


def _chunks(texts):
    return [Chunk(f"D{i}:0", f"D{i}", 0, len(t), t) for i, t in enumerate(texts)]


CHUNKS = _chunks(
    [
        "dishonour of cheque under section 138 requires notice",
        "tenancy terminated by fifteen days notice",
        "force majeure clause excuses impossible performance",
        "bail is the rule and jail the exception",
    ]
)


def test_bm25_ranks_exact_terms_first():
    r = BM25Retriever()
    r.index(CHUNKS)
    hits = r.search("section 138 cheque", k=4)
    assert hits[0][0].doc_id == "D0"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_bm25_score_matches_formula():
    r = BM25Retriever(k1=1.5, b=0.75)
    r.index(CHUNKS)
    (chunk, score), *_ = r.search("bail", k=1)
    n, df = 4, 1
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    lens = [len(c.text.split()) for c in CHUNKS]
    dl, avgdl = lens[3], sum(lens) / 4
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert chunk.doc_id == "D3"
    assert abs(score - expected) < 1e-9


def test_bm25_unknown_words_score_zero():
    r = BM25Retriever()
    r.index(CHUNKS)
    assert all(s == 0 for _, s in r.search("zzzz", k=4))


def test_rrf_math_and_order():
    a, b, c = CHUNKS[:3]
    fused = rrf([[(a, 9.0), (b, 5.0)], [(b, 0.9), (c, 0.5)]], k=3, c=60)
    assert fused[0][0] is b  # ranked by both lists
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-12
    assert {h[0].doc_id for h in fused} == {"D0", "D1", "D2"}


def test_rrf_deterministic_ties():
    a, b = CHUNKS[:2]
    one = rrf([[(a, 1.0)], [(b, 1.0)]], k=2)
    two = rrf([[(b, 1.0)], [(a, 1.0)]], k=2)
    assert [h[0].chunk_id for h in one] == [h[0].chunk_id for h in two]


def test_dense_lsa_returns_k_sorted(tmp_path):
    r = DenseRetriever(LSAEncoder(n_components=3), cache_path=tmp_path / "e.npy")
    r.index(CHUNKS)
    hits = r.search("cheque notice", k=10)
    assert len(hits) == 4
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert (tmp_path / "e.npy").is_file()
    r2 = DenseRetriever(LSAEncoder(n_components=3), cache_path=tmp_path / "e.npy")
    r2.index(CHUNKS)  # loads from cache
    assert [h[0].chunk_id for h in r2.search("cheque notice", 4)] == [h[0].chunk_id for h in hits]
