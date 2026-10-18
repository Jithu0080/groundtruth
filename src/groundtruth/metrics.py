"""Retrieval metrics over gold character spans.

`match_gold` turns a ranked list of chunks into a list where position i holds the set of gold
span indices that the rank-(i+1) chunk covers (empty set = not relevant). All metrics read that
list, so a chunking change never invalidates the golden set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from groundtruth.chunking import Chunk

Matches = Sequence[set[int]]


class GoldSpan(TypedDict):
    doc_id: str
    start: int
    end: int


def coverage(chunk: Chunk, span: GoldSpan) -> float:
    """Fraction of the gold span's characters that fall inside the chunk (0..1)."""
    if chunk.doc_id != span["doc_id"]:
        return 0.0
    overlap = min(chunk.end, span["end"]) - max(chunk.start, span["start"])
    return max(overlap, 0) / (span["end"] - span["start"])


def match_gold(chunks: Sequence[Chunk], gold: Sequence[GoldSpan], min_cover: float = 0.5) -> list[set[int]]:
    """For each chunk in rank order, the gold spans it covers by at least `min_cover`.

    The >= 50% rule stops a huge chunk from "hitting" a passage it barely touches.
    """
    return [{gi for gi, g in enumerate(gold) if coverage(ch, g) >= min_cover} for ch in chunks]


def recall_at_k(matches: Matches, n_gold: int, k: int) -> float:
    """Share of gold spans covered by at least one of the top k chunks."""
    if n_gold <= 0:
        raise ValueError("n_gold must be positive")
    found: set[int] = set().union(*matches[:k]) if matches[:k] else set()
    return len(found) / n_gold


def reciprocal_rank(matches: Matches) -> float:
    """1 / rank of the first relevant chunk, 0 if none."""
    for rank, m in enumerate(matches, start=1):
        if m:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(matches: Matches, n_gold: int, k: int) -> float:
    """Binary-relevance nDCG@k.

    A rank earns gain 1 only if it covers a gold span not already credited higher up, so
    overlapping chunks covering the same passage cannot inflate the score, and nDCG <= 1.
    """
    if n_gold <= 0:
        raise ValueError("n_gold must be positive")
    seen: set[int] = set()
    dcg = 0.0
    for rank, m in enumerate(matches[:k], start=1):
        new = m - seen
        if new:
            seen |= new
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_gold, k) + 1))
    return dcg / idcg
