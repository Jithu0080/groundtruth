from groundtruth.chunking import Chunk
from groundtruth.metrics import coverage, match_gold, ndcg_at_k, recall_at_k, reciprocal_rank

E: set[int] = set()


def test_worked_example():
    """Gold passages at ranks 3 and 7 (hand-computed in the study guide)."""
    m = [E, E, {0}, E, E, E, {1}, E, E, E]
    assert recall_at_k(m, n_gold=2, k=5) == 0.5
    assert recall_at_k(m, n_gold=2, k=10) == 1.0
    assert abs(reciprocal_rank(m) - 1 / 3) < 1e-9
    # DCG = 1/log2(4) + 1/log2(8) = 0.8333 ; IDCG = 1 + 1/log2(3) = 1.6309
    assert abs(ndcg_at_k(m, n_gold=2, k=10) - 0.5110) < 1e-3


def test_no_hits():
    m = [E] * 10
    assert recall_at_k(m, 1, 10) == 0.0
    assert reciprocal_rank(m) == 0.0
    assert ndcg_at_k(m, 1, 10) == 0.0


def test_perfect_ranking_scores_one():
    m = [{0}, {1}, E]
    assert ndcg_at_k(m, 2, 10) == 1.0
    assert reciprocal_rank(m) == 1.0


def test_same_gold_span_counted_once():
    """Two overlapping chunks covering the same passage must not double-count."""
    m = [{0}, {0}, E]
    assert recall_at_k(m, 2, 10) == 0.5
    assert abs(ndcg_at_k(m, 2, 10) - 1 / (1 + 1 / 1.584962500721156)) < 1e-6


def test_one_chunk_covering_two_spans_credits_both_for_recall():
    m = [{0, 1}]
    assert recall_at_k(m, 2, 10) == 1.0
    assert ndcg_at_k(m, 2, 10) <= 1.0


def test_coverage_and_50_percent_rule():
    gold = {"doc_id": "D", "start": 100, "end": 200}
    assert coverage(Chunk("D:0", "D", 0, 150, ""), gold) == 0.5
    assert coverage(Chunk("X:0", "X", 0, 500, ""), gold) == 0.0
    chunks = [
        Chunk("D:0", "D", 0, 140, ""),  # covers 40% -> not a hit
        Chunk("D:120", "D", 120, 400, ""),  # covers 80% -> hit
    ]
    assert match_gold(chunks, [gold]) == [set(), {0}]


def test_k_larger_than_list():
    assert recall_at_k([{0}], 1, 10) == 1.0
